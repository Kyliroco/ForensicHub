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

    # Init global (default) evaluators
    evaluator_list = []
    for eva_args in evaluator_args:
        evaluator_list.append(build_from_registry(EVALUATORS, eva_args))
    if evaluator_list:
        print(f"Global evaluators: {evaluator_list}")

    # Init per-dataset evaluators (if specified in YAML, else fallback to global)
    per_dataset_evaluators = {}
    for ds_args_item in run_dataset_args:
        ds_name = ds_args_item["dataset_name"]
        if "evaluator" in ds_args_item:
            ds_evaluators = []
            for eva_args_item in ds_args_item["evaluator"]:
                ds_evaluators.append(build_from_registry(EVALUATORS, eva_args_item))
            per_dataset_evaluators[ds_name] = ds_evaluators
            print(f"Evaluators for {ds_name}: {ds_evaluators}")
        else:
            per_dataset_evaluators[ds_name] = evaluator_list

    # Global settings
    output_base_dir = getattr(args, 'output_base_dir', './run_output')
    threshold = getattr(args, 'threshold', None)
    score_threshold = getattr(args, 'score_threshold', None)
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

        # Per-dataset threshold override
        ds_threshold = ds_args.get("threshold", threshold)
        ds_score_threshold = ds_args.get("score_threshold", score_threshold)

        # Evaluators for this dataset
        ds_evaluator_list = per_dataset_evaluators.get(dataset_name, evaluator_list)
        for ev in ds_evaluator_list:
            ev.recovery()

        # Track per-image scores and names
        all_names = []
        all_scores = []
        all_labels = []

        print(
            Fore.CYAN + f"\n{'='*60}\n"
            f"  RUN >> {dataset_name}\n"
            f"  Output: {output_dir}\n"
            f"  Threshold: {ds_threshold}\n"
            f"  Score threshold: {ds_score_threshold}\n"
            f"  Evaluators: {[e.name for e in ds_evaluator_list]}\n"
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

        os.makedirs(output_dir, exist_ok=True)

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
                if predict_mask and 'pred_mask' in output_dict:
                    preds = output_dict['pred_mask']
                    # Sigmoid if not already in [0, 1]
                    if preds.min() < 0 or preds.max() > 1:
                        preds = torch.sigmoid(preds)
                    preds = preds.cpu().numpy()
                elif predict_label and 'pred_label' in output_dict:
                    pred_labels = output_dict['pred_label']
                    if pred_labels.min() < 0 or pred_labels.max() > 1:
                        pred_labels = torch.sigmoid(pred_labels)
                    pred_labels = pred_labels.cpu().numpy()
                else:
                    # Fallback: try any prediction key
                    for k in ('pred_mask', 'pred_label', 'pred'):
                        if k in output_dict:
                            preds = output_dict[k]
                            if isinstance(preds, torch.Tensor):
                                preds = preds.cpu().numpy()
                            break
                    else:
                        print(f"  Warning: No prediction found in output keys: {list(output_dict.keys())}")
                        continue

                names = data_dict.get('name', [f"img_{batch_idx}_{i}" for i in range(len(preds) if 'preds' in dir() else len(pred_labels))])
                labels = data_dict.get('label', None)

                # Handle label-only prediction (no spatial output to save as image)
                if predict_label and not predict_mask and 'pred_mask' not in output_dict:
                    # Feed evaluators
                    pred_labels_tensor = torch.as_tensor(pred_labels).to(device) if not isinstance(pred_labels, torch.Tensor) else pred_labels.to(device)
                    if labels is not None:
                        labels_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.as_tensor(labels).to(device)
                        for ev in ds_evaluator_list:
                            ev.batch_update(pred_labels_tensor, labels_tensor)

                    for i, name in enumerate(names):
                        pred_val = pred_labels[i] if isinstance(pred_labels, np.ndarray) else pred_labels
                        if isinstance(pred_val, np.ndarray):
                            pred_val = pred_val.item() if pred_val.size == 1 else pred_val.flatten()[0]
                        elif isinstance(pred_val, torch.Tensor):
                            pred_val = pred_val.item()

                        label_val = None
                        if labels is not None:
                            label_val = labels[i].item() if isinstance(labels[i], torch.Tensor) else int(labels[i])

                        all_names.append(str(name))
                        all_scores.append(float(pred_val))
                        all_labels.append(label_val)
                    continue

                # For mask predictions - handle sliding window or direct save
                if is_sliding_window and 'sw_meta' in data_dict:
                    # Accumulate for later merge
                    sw_metas = data_dict['sw_meta']
                    for i, name in enumerate(names):
                        sw_all_names.append(name)
                        sw_all_preds.append(preds[i])
                        # Extract meta for this item
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

                        # Track labels per original image
                        if labels is not None:
                            orig_name = meta.get('original_name', name)
                            label_val = labels[i].item() if isinstance(labels[i], torch.Tensor) else int(labels[i])
                            # Keep max label (if any patch is tampered, image is tampered)
                            if orig_name not in sw_all_labels:
                                sw_all_labels[orig_name] = label_val
                            else:
                                sw_all_labels[orig_name] = max(sw_all_labels[orig_name], label_val)
                else:
                    # Feed evaluators (use pred_label if available for image-level score)
                    if 'pred_label' in output_dict and labels is not None:
                        ev_preds = output_dict['pred_label']
                        if ev_preds.min() < 0 or ev_preds.max() > 1:
                            ev_preds = torch.sigmoid(ev_preds)
                        labels_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.as_tensor(labels).to(device)
                        for ev in ds_evaluator_list:
                            ev.batch_update(ev_preds.to(device), labels_tensor)

                    # Direct save (no sliding window)
                    for i, name in enumerate(names):
                        pred = preds[i]

                        # Apply threshold if specified
                        if ds_threshold is not None:
                            pred = apply_threshold(pred, ds_threshold)

                        # Compute per-image score (use pred_label if available, else mask mean)
                        if 'pred_label' in output_dict:
                            ev_preds_np = output_dict['pred_label']
                            if isinstance(ev_preds_np, torch.Tensor):
                                ev_preds_np = ev_preds_np.detach().cpu().numpy()
                            score_val = float(ev_preds_np[i].item() if ev_preds_np[i].size == 1 else ev_preds_np[i].flatten()[0])
                        else:
                            score_val = float(np.mean(pred))

                        label_val = None
                        if labels is not None:
                            label_val = labels[i].item() if isinstance(labels[i], torch.Tensor) else int(labels[i])

                        all_names.append(str(name))
                        all_scores.append(score_val)
                        all_labels.append(label_val)

                        # Determine subfolder by label if available
                        if labels is not None:
                            subfolder = os.path.join(output_dir, f"label_{label_val}")
                        else:
                            subfolder = output_dir

                        # Clean name for filename
                        clean_name = os.path.splitext(os.path.basename(str(name)))[0]
                        save_path = os.path.join(subfolder, f"{clean_name}.png")
                        save_prediction_image(pred, save_path)

        # Merge sliding window predictions
        if is_sliding_window and sw_all_names:
            merge_mode = getattr(dataset, 'merge_mode', 'gaussian')
            print(f"  Merging {len(sw_all_names)} patches with mode='{merge_mode}'...")
            # Build origin_sizes from metas
            origin_sizes = {}
            for meta in sw_all_metas:
                if meta.get("split", False):
                    orig_name = meta["original_name"]
                    origin_sizes[orig_name] = (meta["origin_h"], meta["origin_w"])

            # Build predictions dict
            predictions = {}
            for name, pred in zip(sw_all_names, sw_all_preds):
                predictions[name] = pred

            from ForensicHub.common.wrapper.sliding_window_merge import merge_predictions
            merged = merge_predictions(predictions, origin_sizes, mode=merge_mode)

            print(f"  Merged into {len(merged)} full images.")

            # Save merged results
            for img_name, pred in merged.items():
                # Apply threshold if specified
                if ds_threshold is not None:
                    pred = apply_threshold(pred, ds_threshold)

                mask_score = float(np.mean(pred))
                label_val = sw_all_labels.get(img_name, None)

                all_names.append(str(img_name))
                all_scores.append(mask_score)
                all_labels.append(label_val)

                # Determine subfolder by label if available
                if label_val is not None:
                    subfolder = os.path.join(output_dir, f"label_{label_val}")
                else:
                    subfolder = output_dir

                clean_name = os.path.splitext(os.path.basename(str(img_name)))[0]
                save_path = os.path.join(subfolder, f"{clean_name}.png")
                save_prediction_image(pred, save_path)

            print(f"  Saved merged predictions to {output_dir}")

        # ---- Evaluator metrics ----
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

            # Save evaluator results to JSON
            if eval_results:
                eval_json_path = os.path.join(output_dir, "evaluator_results.json")
                with open(eval_json_path, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                print(f"  Evaluator results saved to {eval_json_path}")

        # ---- Score details CSV + score-based folder separation ----
        if all_names:
            has_labels = any(l is not None for l in all_labels)

            # Write detailed score CSV
            score_csv_path = os.path.join(output_dir, "score_details.csv")
            with open(score_csv_path, 'w') as f:
                if has_labels:
                    f.write("name,score,ground_truth_label,classification\n")
                else:
                    f.write("name,score,classification\n")
                for name, score, label_val in zip(all_names, all_scores, all_labels):
                    if ds_score_threshold is not None:
                        classification = "above" if score >= ds_score_threshold else "below"
                    else:
                        classification = "N/A"
                    if has_labels:
                        gt = label_val if label_val is not None else ""
                        f.write(f"{name},{score:.6f},{gt},{classification}\n")
                    else:
                        f.write(f"{name},{score:.6f},{classification}\n")
            print(f"  Score details saved to {score_csv_path}")

            # Score-based folder separation
            if ds_score_threshold is not None:
                above_dir = os.path.join(output_dir, "above_score")
                below_dir = os.path.join(output_dir, "below_score")
                os.makedirs(above_dir, exist_ok=True)
                os.makedirs(below_dir, exist_ok=True)

                count_above = 0
                count_below = 0

                # Write per-folder CSV with details
                above_entries = []
                below_entries = []
                for name, score, label_val in zip(all_names, all_scores, all_labels):
                    if score >= ds_score_threshold:
                        above_entries.append((name, score, label_val))
                        count_above += 1
                    else:
                        below_entries.append((name, score, label_val))
                        count_below += 1

                for folder, entries in [(above_dir, above_entries), (below_dir, below_entries)]:
                    csv_path = os.path.join(folder, "predictions.csv")
                    with open(csv_path, 'w') as f:
                        if has_labels:
                            f.write("name,score,ground_truth_label\n")
                            for n, s, l in entries:
                                gt = l if l is not None else ""
                                f.write(f"{n},{s:.6f},{gt}\n")
                        else:
                            f.write("name,score\n")
                            for n, s, _ in entries:
                                f.write(f"{n},{s:.6f}\n")

                # Copy mask images to filtered folders if mask prediction
                if predict_mask:
                    for name, score, label_val in zip(all_names, all_scores, all_labels):
                        clean_name = os.path.splitext(os.path.basename(str(name)))[0]
                        target_dir = above_dir if score >= ds_score_threshold else below_dir

                        # Find source image in output_dir
                        if label_val is not None:
                            src_path = os.path.join(output_dir, f"label_{label_val}", f"{clean_name}.png")
                        else:
                            src_path = os.path.join(output_dir, f"{clean_name}.png")

                        if os.path.exists(src_path):
                            dst_path = os.path.join(target_dir, f"{clean_name}.png")
                            shutil.copy2(src_path, dst_path)

                print(
                    Fore.GREEN + f"  Score filtering (threshold={ds_score_threshold}):\n"
                    f"    Above: {count_above} samples -> {above_dir}\n"
                    f"    Below: {count_below} samples -> {below_dir}" + Style.RESET_ALL
                )

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
