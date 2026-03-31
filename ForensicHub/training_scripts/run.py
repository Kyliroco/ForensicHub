import os
import json
import time
import datetime
import argparse
import shutil
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import ForensicHub.training_scripts.utils.misc as misc
from ForensicHub.registry import DATASETS, MODELS, POSTFUNCS, TRANSFORMS, EVALUATORS, build_from_registry
from ForensicHub.common.evaluation import *  # noqa: F401,F403 - triggers evaluator registration
from ForensicHub.common.utils.yaml import load_yaml_config, split_run_config, add_attr
from ForensicHub.common.wrapper.sliding_window_merge import merge_batch_predictions
from colorama import Fore, Style


def get_args_parser():
    parser = argparse.ArgumentParser('ForensicHub run (inference) launch!', add_help=True)
    parser.add_argument("--config", type=str, help="Path to YAML config file", required=True)

    args = parser.parse_args()
    config = load_yaml_config(args.config)

    args, model_args, run_dataset_args, transform_args, evaluator_args = split_run_config(config)
    add_attr(args, output_dir=args.log_dir)
    return args, model_args, run_dataset_args, transform_args, evaluator_args


def save_prediction_image(pred, save_path):
    """Save a prediction (numpy array) as a PNG image.

    Handles 2D (HW) and 3D (CHW or HWC) arrays.
    Values are expected in [0, 1] range and will be scaled to [0, 255].
    """
    if pred.ndim == 3:
        # If CHW, convert to HWC
        if pred.shape[0] in (1, 3):
            pred = np.transpose(pred, (1, 2, 0))
        if pred.shape[2] == 1:
            pred = pred.squeeze(2)

    # Clamp and scale to uint8
    pred = np.clip(pred, 0.0, 1.0)
    pred = (pred * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = Image.fromarray(pred)
    img.save(save_path)


def apply_threshold(pred, threshold):
    """Zero out pixels below threshold value."""
    pred = pred.copy()
    pred[pred < threshold] = 0.0
    return pred


def compute_per_image_metric(pred_mask, gt_mask, pred_label, evaluator):
    """Compute a per-image metric score using the evaluator's type.

    For image-level evaluators (ImageAP, ImageMCC, ImageTPR, ImageTNR):
        The score is the pred_label confidence value.

    For pixel-level evaluators (PixelF1, PixelIOU):
        Computes the metric on this single image's pred_mask vs gt_mask.
        Uses the evaluator's own threshold to binarize the prediction.

    Returns the score as a float.
    """
    ev_name = evaluator.name.lower()

    # Image-level: score = model confidence
    if 'image' in ev_name or 'ap' in ev_name:
        if pred_label is not None:
            return float(pred_label)
        # Fallback to mask mean if no pred_label
        if pred_mask is not None:
            return float(np.mean(pred_mask))
        return 0.0

    # Pixel-level: compute F1 or IoU per-image
    if pred_mask is None or gt_mask is None:
        return float('nan')

    ev_pixel_threshold = getattr(evaluator, 'threshold', 0.5)

    # Binarize prediction
    if isinstance(pred_mask, np.ndarray):
        pred_flat = (pred_mask.flatten() >= ev_pixel_threshold).astype(np.float32)
    else:
        pred_flat = (pred_mask.flatten().cpu().numpy() >= ev_pixel_threshold).astype(np.float32)

    # Binarize ground truth
    if isinstance(gt_mask, np.ndarray):
        gt_flat = (gt_mask.flatten() > 0.5).astype(np.float32)
    else:
        gt_flat = (gt_mask.flatten().cpu().numpy() > 0.5).astype(np.float32)

    tp = np.sum(pred_flat * gt_flat)
    fp = np.sum(pred_flat * (1 - gt_flat))
    fn = np.sum((1 - pred_flat) * gt_flat)

    if 'iou' in ev_name:
        return float(tp / (tp + fp + fn + 1e-9))
    else:
        # F1 (default for pixel-level)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        return float(2 * precision * recall / (precision + recall + 1e-9))


def _score_str(score):
    if np.isnan(score) or np.isinf(score):
        return str(score)
    return f"{score:.6f}"


def _classify(score, threshold):
    if threshold is None:
        return "N/A"
    if np.isnan(score) or np.isinf(score):
        return "nan"
    return "above" if score >= threshold else "below"


def main(args, model_args, run_dataset_args, transform_args, evaluator_args):
    # Init distributed mode
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Build transforms (optional)
    if transform_args:
        transform = build_from_registry(TRANSFORMS, transform_args)
        test_transform = transform.get_test_transform()
        post_transform = transform.get_post_transform()
    else:
        test_transform = None
        post_transform = None

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
    else:
        global_rank = 0

    print("Transform: ", test_transform)
    print("Post transform: ", post_transform)

    # Init model
    model = build_from_registry(MODELS, model_args)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=getattr(args, 'find_unused_parameters', False)
        )
        model_without_ddp = model.module

    # Load checkpoint
    checkpoint_path = args.checkpoint_path
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        model_without_ddp.load_state_dict(ckpt['model'])
    else:
        model_without_ddp.load_state_dict(ckpt)
    print("Checkpoint loaded successfully.")

    # Get post function (if any)
    post_function_name = f"{model_args['name']}_post_func".lower()
    if model_args.get('post_func_name') is not None:
        post_function_name = f"{model_args['post_func_name']}_post_func".lower()
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None
    print(f"Post function: {post_function}")

    # ---- Build evaluators ----
    # Global evaluators + extract score_threshold from the first one that has it
    evaluator_list = []
    global_score_threshold = None
    for eva_args in evaluator_args:
        evaluator_list.append(build_from_registry(EVALUATORS, eva_args))
        if global_score_threshold is None and "score_threshold" in eva_args:
            global_score_threshold = eva_args["score_threshold"]
    if evaluator_list:
        print(f"Global evaluators: {[e.name for e in evaluator_list]}")
        if global_score_threshold is not None:
            print(f"Global score_threshold (from 1st evaluator): {global_score_threshold}")

    # Per-dataset evaluators
    per_dataset_evaluators = {}
    per_dataset_score_threshold = {}
    for ds_args_item in run_dataset_args:
        ds_name = ds_args_item["dataset_name"]
        if "evaluator" in ds_args_item:
            ds_evals = []
            ds_st = None
            for eva_args_item in ds_args_item["evaluator"]:
                ds_evals.append(build_from_registry(EVALUATORS, eva_args_item))
                if ds_st is None and "score_threshold" in eva_args_item:
                    ds_st = eva_args_item["score_threshold"]
            per_dataset_evaluators[ds_name] = ds_evals
            per_dataset_score_threshold[ds_name] = ds_st
            print(f"Evaluators for {ds_name}: {[e.name for e in ds_evals]}, score_threshold={ds_st}")
        else:
            per_dataset_evaluators[ds_name] = evaluator_list
            per_dataset_score_threshold[ds_name] = global_score_threshold

    # Global settings
    output_base_dir = getattr(args, 'output_base_dir', './run_output')
    threshold = getattr(args, 'threshold', None)
    predict_mask = getattr(args, 'if_predict_mask', False)
    predict_label = getattr(args, 'if_predict_label', True)

    # Set model to eval mode
    no_model_eval = getattr(args, 'no_model_eval', False)
    if not no_model_eval:
        model.eval()

    start_time = time.time()

    # Process each dataset
    for ds_args in run_dataset_args:
        dataset_name = ds_args["dataset_name"]
        output_subdir = ds_args.get("output_dir", dataset_name)
        output_dir = os.path.join(output_base_dir, output_subdir)

        # Per-dataset pixel threshold override
        ds_threshold = ds_args.get("threshold", threshold)

        # Evaluators & score_threshold for this dataset
        ds_evaluator_list = per_dataset_evaluators.get(dataset_name, evaluator_list)
        ds_score_threshold = per_dataset_score_threshold.get(dataset_name, global_score_threshold)
        for ev in ds_evaluator_list:
            ev.recovery()

        # The first evaluator is used for per-image scoring
        primary_evaluator = ds_evaluator_list[0] if ds_evaluator_list else None

        print(
            Fore.CYAN + f"\n{'='*60}\n"
            f"  RUN >> {dataset_name}\n"
            f"  Output: {output_dir}\n"
            f"  Pixel threshold: {ds_threshold}\n"
            f"  Evaluators: {[e.name for e in ds_evaluator_list]}\n"
            f"  Score threshold (1st evaluator): {ds_score_threshold}\n"
            f"{'='*60}" + Style.RESET_ALL
        )

        # Build dataset
        ds_args["init_config"].update({
            "post_funcs": post_function,
            "common_transform": test_transform,
            "post_transform": post_transform
        })
        dataset = build_from_registry(DATASETS, ds_args)
        print(f"  Dataset: {dataset} ({len(dataset)} samples)")

        # Create dataloader - NO shuffle for inference
        if args.distributed:
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
                drop_last=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        batch_size = getattr(args, 'batch_size', 8)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=getattr(args, 'num_workers', 4),
            pin_memory=getattr(args, 'pin_mem', True),
            drop_last=False,
            collate_fn=misc.common_keys_collate,
        )

        # Check if this is a sliding window dataset
        is_sliding_window = isinstance(dataset, torch.utils.data.Dataset) and \
            hasattr(dataset, '_meta')  # SlidingWindowWrapper has _meta

        # Accumulate predictions for sliding window merge
        sw_all_names = []
        sw_all_preds = []
        sw_all_metas = []
        sw_all_labels = {}  # original_name -> label

        # Prepare output dirs
        os.makedirs(output_dir, exist_ok=True)
        above_dir = os.path.join(output_dir, "above_score")
        below_dir = os.path.join(output_dir, "below_score")
        nan_dir = os.path.join(output_dir, "nan_score")
        if ds_score_threshold is not None:
            os.makedirs(above_dir, exist_ok=True)
            os.makedirs(below_dir, exist_ok=True)

        # Open score CSV for streaming writes
        score_csv_path = os.path.join(output_dir, "score_details.csv")
        score_csv_f = open(score_csv_path, 'w')
        # We'll write the header after we know if labels exist; track state
        csv_header_written = False
        count_above = 0
        count_below = 0
        count_nan = 0

        # Per-folder CSV file handles (opened lazily)
        above_csv_f = None
        below_csv_f = None
        nan_csv_f = None

        def _write_score_row(f, name, score, label_val, has_labels, write_classification=True):
            """Write a single row to a score CSV."""
            s_str = _score_str(score)
            if write_classification:
                cls = _classify(score, ds_score_threshold)
                if has_labels:
                    gt = label_val if label_val is not None else ""
                    f.write(f"{name},{s_str},{gt},{cls}\n")
                else:
                    f.write(f"{name},{s_str},{cls}\n")
            else:
                if has_labels:
                    gt = label_val if label_val is not None else ""
                    f.write(f"{name},{s_str},{gt}\n")
                else:
                    f.write(f"{name},{s_str}\n")
            f.flush()

        def _get_folder_csv(folder, has_labels):
            """Open a predictions.csv in the given folder, write header."""
            os.makedirs(folder, exist_ok=True)
            f = open(os.path.join(folder, "predictions.csv"), 'w')
            if has_labels:
                f.write("name,score,ground_truth_label\n")
            else:
                f.write("name,score\n")
            return f

        def process_single_image(name, pred_mask_np, gt_mask, pred_label_val, label_val, has_labels):
            """Process one image: compute evaluator scores, write to CSVs, save to correct folder.

            pred_mask_np: numpy array or None (mask prediction)
            gt_mask: tensor or None (ground truth mask)
            pred_label_val: float or None (model's label confidence)
            label_val: int or None (ground truth label)
            """
            nonlocal csv_header_written, count_above, count_below, count_nan
            nonlocal above_csv_f, below_csv_f, nan_csv_f

            # Compute per-image score for ALL evaluators (for the CSV detail)
            scores_dict = {}
            for ev in ds_evaluator_list:
                s = compute_per_image_metric(pred_mask_np, gt_mask, pred_label_val, ev)
                scores_dict[ev.name] = s

            # The primary evaluator's score determines the above/below split
            if primary_evaluator is not None:
                primary_score = scores_dict[primary_evaluator.name]
            elif pred_label_val is not None:
                primary_score = pred_label_val
            else:
                primary_score = float(np.mean(pred_mask_np)) if pred_mask_np is not None else 0.0

            # Write CSV header on first row
            if not csv_header_written:
                ev_cols = ",".join(ev.name for ev in ds_evaluator_list) if ds_evaluator_list else ""
                if has_labels:
                    header = f"name,score,ground_truth_label,classification"
                else:
                    header = f"name,score,classification"
                if ev_cols:
                    header += f",{ev_cols}"
                score_csv_f.write(header + "\n")
                csv_header_written = True

            # Build row
            s_str = _score_str(primary_score)
            cls = _classify(primary_score, ds_score_threshold)
            if has_labels:
                gt = label_val if label_val is not None else ""
                row = f"{name},{s_str},{gt},{cls}"
            else:
                row = f"{name},{s_str},{cls}"
            # Append per-evaluator scores
            for ev in ds_evaluator_list:
                row += f",{_score_str(scores_dict[ev.name])}"
            score_csv_f.write(row + "\n")
            score_csv_f.flush()

            # Save mask image to filtered folder (streaming)
            if ds_score_threshold is not None:
                is_nan = np.isnan(primary_score) or np.isinf(primary_score)
                if is_nan:
                    target_dir = nan_dir
                    count_nan += 1
                    if nan_csv_f is None:
                        nan_csv_f = _get_folder_csv(nan_dir, has_labels)
                    _write_score_row(nan_csv_f, name, primary_score, label_val, has_labels, write_classification=False)
                elif primary_score >= ds_score_threshold:
                    target_dir = above_dir
                    count_above += 1
                    if above_csv_f is None:
                        above_csv_f = _get_folder_csv(above_dir, has_labels)
                    _write_score_row(above_csv_f, name, primary_score, label_val, has_labels, write_classification=False)
                else:
                    target_dir = below_dir
                    count_below += 1
                    if below_csv_f is None:
                        below_csv_f = _get_folder_csv(below_dir, has_labels)
                    _write_score_row(below_csv_f, name, primary_score, label_val, has_labels, write_classification=False)

                # Save mask prediction image directly to filtered folder
                if pred_mask_np is not None:
                    clean_name = os.path.splitext(os.path.basename(str(name)))[0]
                    save_path = os.path.join(target_dir, f"{clean_name}.png")
                    save_prediction_image(pred_mask_np, save_path)

        # ========== Inference loop ==========
        with torch.no_grad():
            for batch_idx, data_dict in enumerate(tqdm(
                dataloader, desc=f"  Inference {dataset_name}",
                disable=(global_rank != 0)
            )):
                # Move tensors to device
                for key in data_dict:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(device, non_blocking=True)

                # Forward pass
                output_dict = model(**data_dict)

                # Extract predictions
                preds = None
                pred_labels = None
                if predict_mask and 'pred_mask' in output_dict:
                    preds = output_dict['pred_mask']
                    if preds.min() < 0 or preds.max() > 1:
                        preds = torch.sigmoid(preds)
                    preds = preds.cpu().numpy()
                if predict_label and 'pred_label' in output_dict:
                    pred_labels = output_dict['pred_label']
                    if pred_labels.min() < 0 or pred_labels.max() > 1:
                        pred_labels = torch.sigmoid(pred_labels)
                    pred_labels = pred_labels.cpu().numpy()

                if preds is None and pred_labels is None:
                    # Fallback
                    for k in ('pred_mask', 'pred_label', 'pred'):
                        if k in output_dict:
                            preds = output_dict[k]
                            if isinstance(preds, torch.Tensor):
                                preds = preds.cpu().numpy()
                            break
                    else:
                        print(f"  Warning: No prediction found in output keys: {list(output_dict.keys())}")
                        continue

                n_samples = len(preds) if preds is not None else len(pred_labels)
                names = data_dict.get('name', [f"img_{batch_idx}_{i}" for i in range(n_samples)])
                labels = data_dict.get('label', None)
                gt_masks = data_dict.get('mask', None)
                has_labels = labels is not None

                # Feed evaluators with batch data (for aggregate metrics)
                if pred_labels is not None and labels is not None:
                    pl_tensor = torch.as_tensor(pred_labels).to(device)
                    lbl_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.as_tensor(labels).to(device)
                    for ev in ds_evaluator_list:
                        ev.batch_update(pl_tensor, lbl_tensor)

                # Handle sliding window: accumulate for merge later
                if is_sliding_window and 'sw_meta' in data_dict:
                    sw_metas = data_dict['sw_meta']
                    for i, name in enumerate(names):
                        sw_all_names.append(name)
                        sw_all_preds.append(preds[i])
                        meta = {}
                        for k in sw_metas:
                            val = sw_metas[k]
                            if isinstance(val, torch.Tensor):
                                meta[k] = val[i].item()
                            elif isinstance(val, list):
                                meta[k] = val[i]
                            else:
                                meta[k] = val
                        sw_all_metas.append(meta)
                        if labels is not None:
                            orig_name = meta.get('original_name', name)
                            label_val = labels[i].item() if isinstance(labels[i], torch.Tensor) else int(labels[i])
                            if orig_name not in sw_all_labels:
                                sw_all_labels[orig_name] = label_val
                            else:
                                sw_all_labels[orig_name] = max(sw_all_labels[orig_name], label_val)
                    continue

                # ---- Process each image immediately (streaming) ----
                for i, name in enumerate(names):
                    pred_mask_np = preds[i] if preds is not None else None
                    gt_mask_i = gt_masks[i] if gt_masks is not None else None

                    # Apply pixel threshold
                    if pred_mask_np is not None and ds_threshold is not None:
                        pred_mask_np = apply_threshold(pred_mask_np, ds_threshold)

                    # Get pred_label for this image
                    pred_label_val = None
                    if pred_labels is not None:
                        pl_i = pred_labels[i]
                        pred_label_val = float(pl_i.item() if hasattr(pl_i, 'item') and pl_i.size == 1 else np.asarray(pl_i).flatten()[0])

                    label_val = None
                    if labels is not None:
                        label_val = labels[i].item() if isinstance(labels[i], torch.Tensor) else int(labels[i])

                    # Process: compute scores, write CSV, save to folder
                    process_single_image(
                        name=str(name),
                        pred_mask_np=pred_mask_np,
                        gt_mask=gt_mask_i,
                        pred_label_val=pred_label_val,
                        label_val=label_val,
                        has_labels=has_labels,
                    )

        # ---- Merge sliding window predictions ----
        if is_sliding_window and sw_all_names:
            merge_mode = getattr(dataset, 'merge_mode', 'gaussian')
            print(f"  Merging {len(sw_all_names)} patches with mode='{merge_mode}'...")
            origin_sizes = {}
            for meta in sw_all_metas:
                if meta.get("split", False):
                    orig_name = meta["original_name"]
                    origin_sizes[orig_name] = (meta["origin_h"], meta["origin_w"])

            predictions = {}
            for name, pred in zip(sw_all_names, sw_all_preds):
                predictions[name] = pred

            from ForensicHub.common.wrapper.sliding_window_merge import merge_predictions
            merged = merge_predictions(predictions, origin_sizes, mode=merge_mode)
            print(f"  Merged into {len(merged)} full images.")

            for img_name, pred in merged.items():
                if ds_threshold is not None:
                    pred = apply_threshold(pred, ds_threshold)

                label_val = sw_all_labels.get(img_name, None)
                has_labels = label_val is not None

                process_single_image(
                    name=str(img_name),
                    pred_mask_np=pred,
                    gt_mask=None,
                    pred_label_val=None,
                    label_val=label_val,
                    has_labels=has_labels,
                )

            print(f"  Saved merged predictions to {output_dir}")

        # ---- Close CSV files ----
        score_csv_f.close()
        if above_csv_f is not None:
            above_csv_f.close()
        if below_csv_f is not None:
            below_csv_f.close()
        if nan_csv_f is not None:
            nan_csv_f.close()
        print(f"  Score details saved to {score_csv_path}")

        # ---- Evaluator aggregate metrics ----
        if ds_evaluator_list:
            print(Fore.YELLOW + f"\n  Evaluator results for {dataset_name}:" + Style.RESET_ALL)
            eval_results = {}
            for ev in ds_evaluator_list:
                try:
                    metric_val = ev.epoch_update()
                    if isinstance(metric_val, torch.Tensor):
                        metric_val = metric_val.item()
                    eval_results[ev.name] = metric_val
                    print(f"    {ev.name}: {metric_val:.6f}")
                except Exception as e:
                    print(f"    {ev.name}: could not compute ({e})")

            if eval_results:
                eval_json_path = os.path.join(output_dir, "evaluator_results.json")
                with open(eval_json_path, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                print(f"  Evaluator results saved to {eval_json_path}")

        # ---- Summary ----
        if ds_score_threshold is not None:
            msg = (
                Fore.GREEN + f"  Score filtering (score_threshold={ds_score_threshold}):\n"
                f"    Above: {count_above} samples -> {above_dir}\n"
                f"    Below: {count_below} samples -> {below_dir}"
            )
            if count_nan > 0:
                msg += f"\n    NaN/Inf: {count_nan} samples -> {nan_dir}"
            print(msg + Style.RESET_ALL)

        ds_time = time.time() - start_time
        print(f"  Done with {dataset_name} in {str(datetime.timedelta(seconds=int(ds_time)))}")

    total_time = time.time() - start_time
    print(f'\nTotal inference time: {str(datetime.timedelta(seconds=int(total_time)))}')
    exit(0)


if __name__ == '__main__':
    args, model_args, run_dataset_args, transform_args, evaluator_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args, run_dataset_args, transform_args, evaluator_args)
