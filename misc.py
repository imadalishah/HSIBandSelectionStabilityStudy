import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import random
import numpy as np
import gc
from model_1_original import QEFilterBank, plot_qe_bank
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional
import json

def select_dataset(dataset_name, config_path="dataset_config.json"):
    """Loads configuration for the specified dataset from a JSON file."""
    with open(config_path, "r") as f:
        dataset_configs = json.load(f)
    
    # Extract the requested dataset config
    config = dataset_configs[dataset_name]
    
    # Convert class_weights from a JSON list to a PyTorch tensor
    if "class_weights" in config:
        config["class_weights"] = torch.tensor(config["class_weights"])
        
    return config


def calculate_classwise_metrics(targets, outputs, num_classes=2, ignore_class=255):
    """Calculate per-class metrics for detailed analysis."""

    outputs = torch.argmax(outputs, dim=1)
    targets = targets.to(outputs.device).long()

    class_metrics = {}
    valid_classes = [i for i in range(num_classes) if i != ignore_class]

    for class_id in valid_classes:
        tp = torch.sum((targets == class_id) & (outputs == class_id)).float()
        tn = torch.sum((targets != class_id) & (targets != ignore_class) & (outputs != class_id)).float()
        fp = torch.sum((targets != class_id) & (targets != ignore_class) & (outputs == class_id)).float()
        fn = torch.sum((targets == class_id) & (outputs != class_id)).float()

        iou = (tp / (tp + fp + fn + 1e-12)).item()
        recall = (tp / (tp + fn + 1e-12)).item()
        precision = (tp / (tp + fp + 1e-12)).item()
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        accuracy = ((tp + tn) / (tp + fp + fn + tn + 1e-12)).item()
        specificity = (tn / (tn + fp + 1e-12)).item()

        class_metrics[f'class_{class_id}'] = {
            'iou': iou,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'support': (targets == class_id).sum().item()
        }

    return class_metrics


def calculate_per_image_iou(target, output, num_classes=2, ignore_class=255):
    """Calculate IoU for a single image."""

    output = torch.argmax(output, dim=0) if output.dim() == 3 else output
    target = target.to(output.device).long()

    valid_mask = (target != ignore_class)

    if valid_mask.sum() == 0:
        return 0.0

    target_valid = target[valid_mask]
    output_valid = output[valid_mask]

    intersection = torch.sum((target_valid == output_valid).float()).item()
    union = valid_mask.sum().item()

    iou = intersection / (union + 1e-12)
    return iou

def calculate_metrics(targets, outputs, class_weights=None, num_classes=2, ignore_class=255):
    """Calculate metrics for segmentation evaluation."""

    if class_weights is None:
        class_weights = torch.ones(num_classes).to(outputs.device)
    else:
        class_weights = class_weights.to(outputs.device)


    outputs = torch.argmax(outputs, dim=1)
    # targets = targets.to(outputs.dtype)
    targets = targets.to(outputs.device).long()

    weighted_iou = 0
    mean_iou = 0
    weighted_recall = 0
    mean_recall = 0
    weighted_precision = 0
    mean_precision = 0
    weighted_accuracy = 0
    mean_accuracy = 0
    weighted_specificity = 0
    mean_specificity = 0
    total_weight = 0

    valid_classes = [i for i in range(num_classes) if i != ignore_class]
    present_classes = [class_id for class_id in valid_classes if (targets == class_id).sum() > 0]

    if len(present_classes) == 0:
        return {
            'mean_iou': torch.tensor(0.0),
            'weighted_iou': torch.tensor(0.0),
            'mean_recall': torch.tensor(0.0),
            'weighted_recall': torch.tensor(0.0),
            'mean_precision': torch.tensor(0.0),
            'weighted_precision': torch.tensor(0.0),
            'mean_accuracy': torch.tensor(0.0),
            'weighted_accuracy': torch.tensor(0.0),
            'mean_specificity': torch.tensor(0.0),
            'weighted_specificity': torch.tensor(0.0),
            'mean_f1': torch.tensor(0.0),
            'weighted_f1': torch.tensor(0.0)
        }

    for class_id in present_classes:
        tp = torch.sum((targets == class_id) & (outputs == class_id)).float()
        tn = torch.sum((targets != class_id) & (targets != ignore_class) & (outputs != class_id)).float()
        fp = torch.sum((targets != class_id) & (targets != ignore_class) & (outputs == class_id)).float()
        fn = torch.sum((targets == class_id) & (outputs != class_id)).float()

        iou = tp / (tp + fp + fn + 1e-12)
        mean_iou += iou
        weighted_iou += class_weights[class_id] * iou

        recall = tp / (tp + fn + 1e-12)
        mean_recall += recall
        weighted_recall += class_weights[class_id] * recall

        precision = tp / (tp + fp + 1e-12)
        mean_precision += precision
        weighted_precision += class_weights[class_id] * precision

        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-12)
        mean_accuracy += accuracy
        weighted_accuracy += class_weights[class_id] * accuracy

        specificity = tn / (tn + fp + 1e-12)
        mean_specificity += specificity
        weighted_specificity += class_weights[class_id] * specificity

        total_weight += class_weights[class_id]

    num_present_classes = len(present_classes)
    mean_iou /= num_present_classes + 1e-12
    mean_recall /= num_present_classes + 1e-12
    mean_precision /= num_present_classes + 1e-12
    mean_accuracy /= num_present_classes + 1e-12
    mean_specificity /= num_present_classes + 1e-12
    mean_f1 = 2 * ((mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-12))

    weighted_iou /= total_weight + 1e-12
    weighted_recall /= total_weight + 1e-12
    weighted_precision /= total_weight + 1e-12
    weighted_accuracy /= total_weight + 1e-12
    weighted_specificity /= total_weight + 1e-12
    weighted_f1 = 2 * ((weighted_precision * weighted_recall) / (weighted_precision + weighted_recall + 1e-12))

    return {
        'mean_iou': mean_iou,
        'weighted_iou': weighted_iou,
        'mean_recall': mean_recall,
        'weighted_recall': weighted_recall,
        'mean_precision': mean_precision,
        'weighted_precision': weighted_precision,
        'mean_f1': mean_f1,
        'weighted_f1': weighted_f1,
        'mean_specificity': mean_specificity,
        'weighted_specificity': weighted_specificity,
        'mean_accuracy': mean_accuracy,
        'weighted_accuracy': weighted_accuracy,
    }

# Combined Loss (Dice Loss + Cross Entropy Loss)
class CombinedLoss(nn.Module):
    """Combines Dice Loss and Cross Entropy Loss"""

    def __init__(self, class_weights=None, num_classes=10, dice_weight=0.5, ce_weight=0.5, ignore_index=255, DEVICE='cpu'):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)

        if class_weights is not None:
            class_weights = class_weights.to(DEVICE)

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):

        loss_dice = self.dice_loss(pred, target)
        loss_ce = self.ce_loss(pred, target)
        return self.dice_weight * loss_dice + self.ce_weight * loss_ce

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup(model=None, train_dataset=None, val_dataset=None, train_loader=None, val_loader=None, optimizer=None, loss_fn=None, final_model=None):
    # 1. Handle final_model cleanup
    if final_model is not None:
        del final_model

    # 2. Handle optimizer cleanup (zero_grad and then del)
    if optimizer is not None:
        # Zero gradients for memory efficiency
        optimizer.zero_grad(set_to_none=True)
        del optimizer

    # 3. Conditionally delete all other arguments
    if model is not None:
        del model

    if train_dataset is not None:
        del train_dataset

    if val_dataset is not None:
        del val_dataset

    if train_loader is not None:
        del train_loader

    if val_loader is not None:
        del val_loader

    if loss_fn is not None:
        del loss_fn

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()





















