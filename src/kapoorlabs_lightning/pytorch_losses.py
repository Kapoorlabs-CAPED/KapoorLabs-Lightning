import torch
import torch.nn as nn
import torch.nn.functional as F


class OneatClassificationLoss(nn.Module):
    """Simple classification loss for ONEAT event classification."""
    def __init__(self, categories, class_weights_dict=None, device="cuda"):
        super().__init__()
        self.categories = categories
        self.device = device

        if class_weights_dict is not None:
            self.class_weights = torch.tensor(
                [class_weights_dict[i] for i in range(len(class_weights_dict))],
                dtype=torch.float32
            )
        else:
            self.class_weights = None

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Model output (batch, categories + box_vector, 1, 1, 1) or (batch, categories + box_vector)
            y_true: Class labels (batch,) as integers
        """
        # Squeeze spatial dimensions if present
        if y_pred.dim() > 2:
            y_pred = y_pred.squeeze(-1).squeeze(-1).squeeze(-1)

        # Extract only the class predictions
        pred_classes = y_pred[:, :self.categories]

        # Move weights to correct device
        weights = self.class_weights.to(y_pred.device) if self.class_weights is not None else None

        loss = F.cross_entropy(pred_classes, y_true.long(), weight=weights)
        return loss


def extract_ground_event_volume_truth(y_true, categories, box_vector):
    """
    Extracts class, position (xyzt), dimensions (hwd), and confidence from ground truth tensor.
    Format: [x, y, z, t, h, w, d, c] + [one-hot categories]

    Args:
        y_true (torch.Tensor): Ground truth tensor (batch, box_vector + categories)
        categories (int): Number of categories/classes.
        box_vector (int): Length of the box vector (8: x,y,z,t,h,w,d,c).
    Returns:
        tuple: (true_box_class, true_box_xyzt, true_box_hwd, true_box_conf)
    """
    # Box vector comes first, then categories
    true_box_xyzt = y_true[..., :4]  # x, y, z, t
    true_box_hwd = y_true[..., 4:7]  # h, w, d
    true_box_conf = y_true[..., 7]   # c (confidence)
    true_box_class = y_true[..., box_vector:]

    return true_box_class, true_box_xyzt, true_box_hwd, true_box_conf


def extract_ground_event_volume_pred(y_pred, categories, box_vector):
    """
    Extracts class, position (xyzt), dimensions (hwd), and confidence from the predicted tensor.
    Format: [x, y, z, t, h, w, d, c] + [one-hot categories]

    Args:
        y_pred (torch.Tensor): Predicted tensor (batch, box_vector + categories)
        categories (int): Number of categories/classes.
        box_vector (int): Length of the box vector (8: x,y,z,t,h,w,d,c).
    Returns:
        tuple: (pred_box_class, pred_box_xyzt, pred_box_hwd, pred_box_conf)
    """
    # Box vector comes first, then categories
    pred_box_xyzt = y_pred[..., :4]  # x, y, z, t
    pred_box_hwd = y_pred[..., 4:7]  # h, w, d
    pred_box_conf = y_pred[..., 7]   # c (confidence)
    pred_box_class = y_pred[..., box_vector:]

    return pred_box_class, pred_box_xyzt, pred_box_hwd, pred_box_conf


# Loss functions
def compute_conf_loss_volume(true_box_conf, pred_box_conf):

    loss_conf = F.mse_loss(true_box_conf, pred_box_conf, reduction="sum")

    return loss_conf


def calc_loss_xyzt_hwd(true_box_xyzt, pred_box_xyzt, true_box_hwd, pred_box_hwd):
    """
    Calculates the loss for position (xyzt) and dimensions (hwd).
    """
    loss_xyzt = torch.sum((true_box_xyzt - pred_box_xyzt) ** 2, dim=-1).sum()
    loss_hwd = torch.sum(
        (torch.sqrt(true_box_hwd + 1e-6) - torch.sqrt(pred_box_hwd + 1e-6)) ** 2, dim=-1
    ).sum()
    loss_total = loss_xyzt + loss_hwd

    return loss_total


def calc_loss_class(true_box_class, pred_box_class, class_weights_dict = None):
    if class_weights_dict is not None:
        class_weights = torch.tensor(
            [class_weights_dict[i] for i in range(len(class_weights_dict))],
            dtype=pred_box_class.dtype,
            device=pred_box_class.device
        )
    else:
        class_weights = None

    true_class_indices = torch.argmax(true_box_class, dim=-1)
    loss_class = F.cross_entropy(
        pred_box_class, true_class_indices, reduction="mean", weight=class_weights
    )
    return loss_class


class VolumeYoloLoss(nn.Module):
    def __init__(self, categories, box_vector, device, class_weights_dict=None):
        super().__init__()
        self.categories = categories
        self.box_vector = box_vector
        self.device = device
        self.class_weights_dict = class_weights_dict

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Model predictions (batch, categories + box_vector, 1, 1, 1) or (batch, categories + box_vector)
            y_true: Ground truth YOLO labels (batch, categories + box_vector)
        """
        # Squeeze spatial dimensions from predictions if present
        if y_pred.dim() > 2:
            y_pred = y_pred.squeeze(-1).squeeze(-1).squeeze(-1)

        y_true = y_true.to(y_pred.device)
        y_pred = y_pred.to(y_pred.device)

        (
            true_box_class,
            true_box_xyzt,
            true_box_hwd,
            true_box_conf,
        ) = extract_ground_event_volume_truth(y_true, self.categories, self.box_vector)
        (
            pred_box_class,
            pred_box_xyzt,
            pred_box_hwd,
            pred_box_conf,
        ) = extract_ground_event_volume_pred(y_pred, self.categories, self.box_vector)

        loss_xyzt_hwd = calc_loss_xyzt_hwd(
            true_box_xyzt, pred_box_xyzt, true_box_hwd, pred_box_hwd
        )
        loss_class = calc_loss_class(true_box_class, pred_box_class, class_weights_dict=self.class_weights_dict)
        loss_conf = compute_conf_loss_volume(true_box_conf, pred_box_conf)

        combined_loss = loss_xyzt_hwd + loss_conf + loss_class
        return combined_loss


__all__ = ["VolumeYoloLoss", "OneatClassificationLoss"]
