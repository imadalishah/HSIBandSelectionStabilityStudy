import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import cohen_kappa_score
import torch
import json
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torch.cuda.amp import GradScaler
from misc import (
    select_dataset, calculate_classwise_metrics,
    calculate_per_image_iou, calculate_metrics,
    CombinedLoss, seed_everything, cleanup,
)


SEEDS = [42, 43, 44]
ABLATION = [3, 5, 7, 9, 11]

DATASET_LIST = ['HSI_City']

MODEL_CONFIGS = [
    {'arch': 'Unet', 'backbone': 'efficientnet-b0'},
    {'arch': 'DeepLabV3Plus', 'backbone': 'efficientnet-b0'},
    {'arch': 'Segformer', 'backbone': 'mit_b0'},
]

# Hyperparameters
NUM_WORKERS = 4
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 50
IGNORE_CLASS = 255
CUDA_DEVICE = 'cuda:0'
DEVICE = torch.device(CUDA_DEVICE if torch.cuda.is_available() else 'cpu')

BASE_OUTPUT_DIR = Path('./experiments/')


# DATASET
class HSIBandDataset(Dataset):
    def __init__(self, data_dir: str, mask_dir: str, num_classes: int, raw_in_channels: int,
                 selected_bands: list[int]):
        """
        Parameters
        ----------
        data_dir        : directory containing raw HSI cubes as .npy files
        mask_dir        : directory containing segmentation masks
        num_classes     : number of segmentation classes
        raw_in_channels : total number of channels in the raw cube
                          (used to normalise the (H,W,C) vs (C,H,W) ambiguity)
        selected_bands  : list of band indices to keep
        """
        self.raw_in_channels = raw_in_channels
        self.selected_bands = selected_bands
        self.num_classes = num_classes

        all_cubes = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
        cube_map = {Path(f).stem: f for f in all_cubes}

        self.cube_files = []
        self.mask_files = []

        for stem, cube_path in cube_map.items():
            candidates = [
                f for f in glob.glob(os.path.join(mask_dir, '*'))
                if Path(f).stem.lower() == stem.lower()
                   and Path(f).suffix.lower() in ('.png', '.tif', '.npy')
            ]
            if candidates:
                if len(candidates) > 1:
                    print(f"Warning: multiple masks for '{stem}'. Using {candidates[0]}")
                self.cube_files.append(cube_path)
                self.mask_files.append(candidates[0])
            else:
                print(f"Warning: no mask for '{stem}' in {mask_dir} — skipping.")

        assert len(self.cube_files) == len(self.mask_files)

    def __len__(self):
        return len(self.cube_files)

    @staticmethod
    def _load_mask(path: str) -> np.ndarray:
        p = Path(path)
        if p.suffix.lower() == '.npy':
            m = np.load(str(p)).astype(np.int64)
        else:
            m = np.array(Image.open(p)).astype(np.int64)
            if m.ndim == 3:
                m = np.argmax(m, axis=2) if m.shape[2] > 1 else m[:, :, 0]
        if m.ndim != 2:
            raise ValueError(f"Mask from {path} is not 2D — shape: {m.shape}")
        return m

    def __getitem__(self, idx):
        # Load & orient cube
        cube = np.load(self.cube_files[idx]).astype(np.float32)

        if cube.ndim != 3:
            raise ValueError(f"Unexpected cube shape {cube.shape} in {self.cube_files[idx]}")

        # Detect (H, W, C) vs (C, H, W)
        if cube.shape[-1] == self.raw_in_channels and cube.shape[0] != self.raw_in_channels:
            cube = np.transpose(cube, (2, 0, 1))  # → (C, H, W)
        elif cube.shape[0] != self.raw_in_channels:
            cube = np.transpose(cube, (2, 0, 1))  # fallback

        # Select bands
        image = cube[self.selected_bands, :, :]  # (len(selected_bands), H, W)

        _, H, W = image.shape
        original_shape = (H, W)

        # Pad to multiple of 32
        pad_H = (32 - H % 32) % 32
        pad_W = (32 - W % 32) % 32
        if pad_H > 0 or pad_W > 0:
            image = np.pad(image, ((0, 0), (0, pad_H), (0, pad_W)), mode='reflect')

        # # Per-channel normalisation
        # mean  = image.mean(axis=(1, 2), keepdims=True)
        # std   = image.std(axis=(1, 2),  keepdims=True) + 1e-6
        # image = (image - mean) / std

        # Load mask
        mask = self._load_mask(self.mask_files[idx])
        if mask.shape != original_shape:
            raise ValueError(
                f"Mask shape {mask.shape} != image spatial shape {original_shape} "
                f"for {self.cube_files[idx]}"
            )
        if pad_H > 0 or pad_W > 0:
            mask = np.pad(mask, ((0, pad_H), (0, pad_W)),
                          mode='constant', constant_values=IGNORE_CLASS)

        stem = Path(self.cube_files[idx]).stem
        return (torch.from_numpy(image).float(), torch.from_numpy(mask).long(),
                stem, original_shape, (pad_H, pad_W))


# TRAINING / EVALUATION
def run_experiment(
        dataset_name: str, num_classes: int, batch_size: int,
        train_data_dir: str, val_data_dir: str,
        train_mask_dir: str, val_mask_dir: str,
        raw_in_channels: int, selected_bands: list[int],
        band_set_name: str,
        ablation: int,
        model_config: dict, all_results: list,
        class_weights, dataset_config: dict,
        seed: int,
        output_dir: Path,
):
    arch_name = model_config['arch']
    backbone_name = model_config['backbone']
    num_bands = len(selected_bands)
    experiment_name = f"{dataset_name}/{band_set_name}/{ablation}/{arch_name}_{backbone_name}"

    print(f"\nRunning: {experiment_name}")
    print("=" * 60)
    print(f"  Dataset        : {dataset_name}")
    print(f"  Band set       : {band_set_name}")
    print(f"  Architecture   : {arch_name} / {backbone_name}")
    print(f"  Selected bands : {selected_bands}  ({num_bands} channels)")
    print("=" * 60)

    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = exp_dir / 'best_model.pth'
    predictions_path = exp_dir / 'predictions'
    predictions_path.mkdir(exist_ok=True)
    log_path = output_dir / 'skipped_experiments.log'
    log_path.touch(exist_ok=True)
    # Datasets & loaders
    try:
        train_dataset = HSIBandDataset(data_dir=train_data_dir, mask_dir=train_mask_dir,
                                       num_classes=num_classes, raw_in_channels=raw_in_channels,
                                       selected_bands=selected_bands,
                                       )
        val_dataset = HSIBandDataset(data_dir=val_data_dir, mask_dir=val_mask_dir,
                                     num_classes=num_classes, raw_in_channels=raw_in_channels,
                                     selected_bands=selected_bands,
                                     )
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Train or validation dataset is empty.")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True,)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True,)
        print(f"Dataset — train: {len(train_dataset)}, val: {len(val_dataset)}")

    except Exception as e:
        print(f"Skipping — data loading error: {e}")
        with open(log_path, 'a') as f:
            f.write(f"{experiment_name} - Data Loading Error: {e}\n")
        return

    # Model
    try:
        model_class = getattr(smp, arch_name)
        model = model_class(encoder_name=backbone_name, in_channels=num_bands, classes=num_classes,
                            ).to(DEVICE)
    except Exception as e:
        print(f"Skipping — model setup error: {e}")
        with open(log_path, 'a') as f:
            f.write(f"{experiment_name} - Model Setup Error: {e}\n")
        return

    # Loss & optimiser
    loss_fn = CombinedLoss(class_weights=class_weights, num_classes=num_classes,
                           ignore_index=IGNORE_CLASS, DEVICE=DEVICE,
                           ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4, )
    scaler = GradScaler()

    # Training loop
    best_iou = -1.0
    epochs_without_improvement = 0
    best_metrics = None
    best_classwise_metrics = None
    accumulation_steps = min(dataset_config.get('accumulation', 1), 4)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0

        for batch_idx, (images, masks, _, _, _) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch} Train")
        ):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            with torch.amp.autocast(device_type='cuda', enabled=True):
                logits = model(images)
                loss = loss_fn(logits, masks) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

        # Flush remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        epoch_metrics_sum = None
        epoch_classwise_sum = None
        all_masks_flat = []
        all_preds_flat = []

        with torch.no_grad():
            for images, masks, _, _, _ in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                logits = model(images)
                val_loss += loss_fn(logits, masks).item()

                batch_metrics = calculate_metrics(masks, logits, class_weights=class_weights,
                                                  num_classes=num_classes, ignore_class=IGNORE_CLASS, )
                batch_classwise = calculate_classwise_metrics(masks, logits, num_classes=num_classes,
                                                              ignore_class=IGNORE_CLASS, )

                if epoch_metrics_sum is None:
                    epoch_metrics_sum = dict(batch_metrics)
                else:
                    for k in batch_metrics:
                        epoch_metrics_sum[k] += batch_metrics[k]

                if epoch_classwise_sum is None:
                    epoch_classwise_sum = {cls: dict(m) for cls, m in batch_classwise.items()}
                else:
                    for cls, m in batch_classwise.items():
                        for k, v in m.items():
                            epoch_classwise_sum[cls][k] += v

                preds = logits.argmax(dim=1)
                all_masks_flat.append(masks.cpu().numpy().flatten())
                all_preds_flat.append(preds.cpu().numpy().flatten())

        if len(val_loader) == 0:
            print("Warning: empty validation loader — skipping epoch.")
            continue

        n_val = len(val_loader)
        avg_val_loss = val_loss / n_val
        current_metrics = {
            k: (v / n_val).item() if torch.is_tensor(v) else v / n_val
            for k, v in epoch_metrics_sum.items()
        }
        current_classwise = {
            cls: {k: v / n_val for k, v in m.items()}
            for cls, m in epoch_classwise_sum.items()
        }

        # Cohen's Kappa
        all_masks_np = np.concatenate(all_masks_flat)
        all_preds_np = np.concatenate(all_preds_flat)
        valid_idx = all_masks_np != IGNORE_CLASS
        kappa_value = 0.0
        if valid_idx.sum() > 0 and len(np.unique(all_preds_np[valid_idx])) > 1:
            kappa_value = cohen_kappa_score(
                all_masks_np[valid_idx], all_preds_np[valid_idx],
                labels=list(range(num_classes)),
            )
        current_metrics['kappa'] = kappa_value
        current_iou = current_metrics['mean_iou']

        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"mIoU: {current_iou:.4f} | Kappa: {kappa_value:.4f}"
              )

        # Early stopping / checkpoint
        if current_iou > best_iou:
            best_iou = current_iou
            epochs_without_improvement = 0
            best_metrics = current_metrics.copy()
            best_classwise_metrics = current_classwise.copy()
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ↑ New best mIoU: {best_iou:.4f} — checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_metrics is None:
        print("No successful epoch — skipping final steps.")
        return

    # Final predictions with best checkpoint
    final_model = model_class(
        encoder_name=backbone_name,
        in_channels=num_bands,
        classes=num_classes,
    ).to(DEVICE)

    try:
        final_model.load_state_dict(
            torch.load(checkpoint_path, map_location=DEVICE), strict=True,
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        with open(log_path, 'a') as f:
            f.write(f"{experiment_name} - Checkpoint Load Error: {e}\n")
        return

    final_model.eval()
    per_image_results = []

    with torch.no_grad():
        for images, masks, stems, original_shapes, _ in tqdm(
                val_loader, desc="Saving Predictions"
        ):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            preds = final_model(images).argmax(dim=1).cpu().numpy()

            for i in range(len(images)):
                H_orig = original_shapes[0][i].item()
                W_orig = original_shapes[1][i].item()
                stem = stems[i]

                pred_mask = preds[i][:H_orig, :W_orig]
                assert pred_mask.shape == (H_orig, W_orig), \
                    f"Shape mismatch after crop: {pred_mask.shape} vs ({H_orig},{W_orig})"

                image_iou = calculate_per_image_iou(
                    masks[i][:H_orig, :W_orig],
                    torch.from_numpy(pred_mask).to(masks.device),
                    num_classes=num_classes, ignore_class=IGNORE_CLASS,
                )
                per_image_results.append({'Image': stem, 'IoU': image_iou})

                Image.fromarray(pred_mask.astype(np.uint8), mode='L').save(
                    predictions_path / f"{stem}_pred.png"
                )

    if per_image_results:
        df = pd.DataFrame(per_image_results).sort_values('IoU', ascending=False)
        df.to_csv(exp_dir / 'per_image_iou.csv', index=False)

    # Save experiment summary
    rows = [
        {'Metric': 'Seed', 'Value': seed},
        {'Metric': 'Experiment', 'Value': experiment_name},
        {'Metric': 'Dataset', 'Value': dataset_name},
        {'Metric': 'Band_Set', 'Value': band_set_name},
        {'Metric': 'Ablation', 'Value': ablation},
        {'Metric': 'Architecture', 'Value': arch_name},
        {'Metric': 'Backbone', 'Value': backbone_name},
        {'Metric': 'Selected_Bands', 'Value': str(selected_bands)},
        {'Metric': 'Num_Bands', 'Value': num_bands},
        {'Metric': '', 'Value': ''},
        {'Metric': '## Overall Metrics', 'Value': ''},
    ]
    for key, label in [
        ('mean_iou', 'mIoU'),
        ('weighted_iou', 'Weighted_IoU'),
        ('mean_f1', 'mF1'),
        ('weighted_f1', 'Weighted_F1'),
        ('mean_accuracy', 'mAccuracy'),
        ('weighted_accuracy', 'Weighted_Accuracy'),
        ('mean_recall', 'mRecall'),
        ('weighted_recall', 'Weighted_Recall'),
        ('mean_precision', 'mPrecision'),
        ('weighted_precision', 'Weighted_Precision'),
        ('mean_specificity', 'mSpecificity'),
        ('weighted_specificity', 'Weighted_Specificity'),
        ('kappa', 'Kappa'),
    ]:
        rows.append({'Metric': label, 'Value': f"{best_metrics[key]:.6f}"})

    if best_classwise_metrics:
        rows += [{'Metric': '', 'Value': ''}, {'Metric': '## Class-wise Metrics', 'Value': ''}]
        for class_name, metrics in sorted(best_classwise_metrics.items()):
            cid = class_name.split('_')[1]
            rows += [
                {'Metric': '', 'Value': ''},
                {'Metric': f'--- Class {cid} ---', 'Value': ''},
                {'Metric': f'Class_{cid}_IoU', 'Value': f"{metrics['iou']:.6f}"},
                {'Metric': f'Class_{cid}_F1', 'Value': f"{metrics['f1']:.6f}"},
                {'Metric': f'Class_{cid}_Precision', 'Value': f"{metrics['precision']:.6f}"},
                {'Metric': f'Class_{cid}_Recall', 'Value': f"{metrics['recall']:.6f}"},
                {'Metric': f'Class_{cid}_Accuracy', 'Value': f"{metrics['accuracy']:.6f}"},
                {'Metric': f'Class_{cid}_Specificity', 'Value': f"{metrics['specificity']:.6f}"},
                {'Metric': f'Class_{cid}_Support', 'Value': int(metrics['support'])},
            ]

    pd.DataFrame(rows).to_csv(exp_dir / 'experiment_summary.csv', index=False)
    print(f"Summary saved to {exp_dir / 'experiment_summary.csv'}")

    all_results.append({
        'Seed': seed,
        'Dataset': dataset_name,
        'Band_Set': band_set_name,
        'Ablation': ablation,
        'Architecture': f"{arch_name}_{backbone_name}",
        'Selected_Bands': str(selected_bands),
        'Num_Bands': num_bands,
        'mIoU': best_metrics['mean_iou'],
        'Weighted_IoU': best_metrics['weighted_iou'],
        'mF1': best_metrics['mean_f1'],
        'Weighted_F1': best_metrics['weighted_f1'],
        'mAcc': best_metrics['mean_accuracy'],
        'Weighted_Acc': best_metrics['weighted_accuracy'],
        'mRecall': best_metrics['mean_recall'],
        'Weighted_Recall': best_metrics['weighted_recall'],
        'mPrecision': best_metrics['mean_precision'],
        'Weighted_Precision': best_metrics['weighted_precision'],
        'mSpecificity': best_metrics['mean_specificity'],
        'Weighted_Specificity': best_metrics['weighted_specificity'],
        'Kappa': best_metrics['kappa'],
    })
    print(f"Final — mIoU={best_metrics['mean_iou']:.4f}, Kappa={best_metrics['kappa']:.4f}")

    cleanup(model, train_dataset, val_dataset, train_loader, val_loader,
            optimizer, loss_fn, final_model)


# ENTRY POINT
if __name__ == '__main__':

    CONFIG_FILE_PATH = 'bandSelect_Config.json'
    
    # Load the JSON map up front
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r') as f:
            RAW_JSON_DATA = json.load(f)
        print(f"Successfully loaded configuration map from {CONFIG_FILE_PATH}")
    else:
        raise FileNotFoundError(f"Critical Error: {CONFIG_FILE_PATH} could not be found.")

    for seed in SEEDS:
        seed_everything(seed)
        OUTPUT_DIR = BASE_OUTPUT_DIR / f"{seed}"

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        final_results = []
        for dataset_name in DATASET_LIST:
            cfg = select_dataset(dataset_name)
            num_classes = cfg['classes']
            # batch_size = cfg['batch_size']
            batch_size = 16
            class_weights = cfg.get('class_weights', None)
            raw_in_channels = cfg['in_channels']

            train_data_dir, train_mask_dir, val_data_dir, val_mask_dir = cfg['OriginalDataset']

            if not os.path.exists(train_data_dir) or not os.path.exists(val_data_dir):
                print(f"ERROR: data paths not found for {dataset_name} — skipping.")
                continue

            # Iterate over every named band set
            for band_set_name, bands in SELECTED_BANDS.items():
                # Original = use all channels
                if band_set_name == 'Original':
                    selected_bands = list(range(raw_in_channels))

                    for model_cfg in MODEL_CONFIGS:
                        run_experiment(
                            dataset_name=dataset_name,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            train_data_dir=train_data_dir,
                            val_data_dir=val_data_dir,
                            train_mask_dir=train_mask_dir,
                            val_mask_dir=val_mask_dir,
                            raw_in_channels=raw_in_channels,
                            selected_bands=selected_bands,
                            band_set_name=band_set_name,
                            ablation=raw_in_channels,
                            model_config=model_cfg,
                            all_results=final_results,
                            class_weights=class_weights,
                            dataset_config=cfg,
                            seed=seed,
                            output_dir=OUTPUT_DIR,
                        )
                    continue

                for ablation in ABLATION:
                    if ablation > len(bands):
                        print(
                            f"Skipping {band_set_name} with ablation={ablation} "
                            f"(only {len(bands)} bands available)"
                        )
                        continue

                    selected_bands = bands[:ablation]

                    max_band = max(selected_bands)
                    if max_band >= raw_in_channels:
                        raise ValueError(
                            f"Band index {max_band} in '{band_set_name}' is out of range for "
                            f"{dataset_name} which has {raw_in_channels} channels."
                        )

                    for model_cfg in MODEL_CONFIGS:
                        try:
                            run_experiment(
                                dataset_name=dataset_name,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                train_data_dir=train_data_dir,
                                val_data_dir=val_data_dir,
                                train_mask_dir=train_mask_dir,
                                val_mask_dir=val_mask_dir,
                                raw_in_channels=raw_in_channels,
                                selected_bands=selected_bands,
                                band_set_name=band_set_name,
                                ablation=ablation,
                                model_config=model_cfg,
                                all_results=final_results,
                                class_weights=class_weights,
                                dataset_config=cfg,
                                seed=seed,
                                output_dir=OUTPUT_DIR,
                            )
                        except Exception as e:
                            import traceback

                            traceback.print_exc()
                            tag = (
                                f"{dataset_name}/{band_set_name}/{ablation}/"
                                f"{model_cfg['arch']}_{model_cfg['backbone']}"
                            )
                            print(f"Experiment crashed: {tag} — {e}")
                            with open(OUTPUT_DIR / 'skipped_experiments.log', 'a') as f:
                                f.write(f"{tag} - Runtime Crash: {e}\n")

        if final_results:
            results_df = pd.DataFrame(final_results)
            results_df['mIoU'] = pd.to_numeric(results_df['mIoU'], errors='coerce')
            results_df = results_df.sort_values(
                by=['Dataset', 'Band_Set', 'mIoU'], ascending=[True, True, False]
            )
            out_csv = OUTPUT_DIR / 'results.csv'
            results_df.to_csv(out_csv, index=False)
            print(f"\nAll results saved to {out_csv}")
        else:
            print("\nNo successful experiments.")
