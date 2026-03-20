import os
import time
import datetime
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import ForensicHub.training_scripts.utils.misc as misc
from ForensicHub.registry import DATASETS, MODELS, POSTFUNCS, TRANSFORMS, build_from_registry
from ForensicHub.common.utils.yaml import load_yaml_config, split_run_config, add_attr
from ForensicHub.common.wrapper.sliding_window_merge import merge_batch_predictions
from colorama import Fore, Style


def get_args_parser():
    parser = argparse.ArgumentParser('ForensicHub run (inference) launch!', add_help=True)
    parser.add_argument("--config", type=str, help="Path to YAML config file", required=True)

    args = parser.parse_args()
    config = load_yaml_config(args.config)

    args, model_args, run_dataset_args, transform_args = split_run_config(config)
    add_attr(args, output_dir=args.log_dir)
    return args, model_args, run_dataset_args, transform_args


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


def main(args, model_args, run_dataset_args, transform_args):
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

    # Global settings
    output_base_dir = getattr(args, 'output_base_dir', './run_output')
    threshold = getattr(args, 'threshold', None)
    merge_mode = getattr(args, 'merge_mode', 'gaussian')
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

        print(
            Fore.CYAN + f"\n{'='*60}\n"
            f"  RUN >> {dataset_name}\n"
            f"  Output: {output_dir}\n"
            f"  Threshold: {ds_threshold}\n"
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
                    # Save label predictions to a text/csv file
                    for i, name in enumerate(names):
                        pred_val = pred_labels[i] if isinstance(pred_labels, np.ndarray) else pred_labels
                        if isinstance(pred_val, np.ndarray):
                            pred_val = pred_val.item() if pred_val.size == 1 else pred_val.flatten()[0]

                        # Determine subfolder by label if available
                        if labels is not None:
                            label_val = labels[i].item() if isinstance(labels[i], torch.Tensor) else int(labels[i])
                            subfolder = os.path.join(output_dir, f"label_{label_val}")
                        else:
                            subfolder = output_dir

                        os.makedirs(subfolder, exist_ok=True)
                        # Append to predictions file
                        pred_file = os.path.join(output_dir, "predictions.csv")
                        with open(pred_file, 'a') as f:
                            if os.path.getsize(pred_file) == 0 if os.path.exists(pred_file) else True:
                                f.write("name,pred_label\n")
                            f.write(f"{name},{pred_val:.6f}\n")
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
                    # Direct save (no sliding window)
                    for i, name in enumerate(names):
                        pred = preds[i]

                        # Apply threshold if specified
                        if ds_threshold is not None:
                            pred = apply_threshold(pred, ds_threshold)

                        # Determine subfolder by label if available
                        if labels is not None:
                            label_val = labels[i].item() if isinstance(labels[i], torch.Tensor) else int(labels[i])
                            subfolder = os.path.join(output_dir, f"label_{label_val}")
                        else:
                            subfolder = output_dir

                        # Clean name for filename
                        clean_name = os.path.splitext(os.path.basename(str(name)))[0]
                        save_path = os.path.join(subfolder, f"{clean_name}.png")
                        save_prediction_image(pred, save_path)

        # Merge sliding window predictions
        if is_sliding_window and sw_all_names:
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

                # Determine subfolder by label if available
                if img_name in sw_all_labels:
                    label_val = sw_all_labels[img_name]
                    subfolder = os.path.join(output_dir, f"label_{label_val}")
                else:
                    subfolder = output_dir

                clean_name = os.path.splitext(os.path.basename(str(img_name)))[0]
                save_path = os.path.join(subfolder, f"{clean_name}.png")
                save_prediction_image(pred, save_path)

            print(f"  Saved merged predictions to {output_dir}")

        ds_time = time.time() - start_time
        print(f"  Done with {dataset_name} in {str(datetime.timedelta(seconds=int(ds_time)))}")

    total_time = time.time() - start_time
    print(f'\nTotal inference time: {str(datetime.timedelta(seconds=int(total_time)))}')
    exit(0)


if __name__ == '__main__':
    args, model_args, run_dataset_args, transform_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args, run_dataset_args, transform_args)
