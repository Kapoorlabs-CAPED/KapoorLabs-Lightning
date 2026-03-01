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
    GT Format: [x, y, z, t, h, w, d, c] + [one-hot categories]

    Args:
        y_true (torch.Tensor): Ground truth tensor (batch, box_vector + categories)
        categories (int): Number of categories/classes.
        box_vector (int): Length of the box vector (8: x,y,z,t,h,w,d,c).
    Returns:
        tuple: (true_box_class, true_box_xyzt, true_box_hwd, true_box_conf)
    """
    # GT format: box_vector first, then categories
    true_box_xyzt = y_true[..., :4]  # x, y, z, t
    true_box_hwd = y_true[..., 4:7]  # h, w, d
    true_box_conf = y_true[..., 7]   # c (confidence)
    true_box_class = y_true[..., box_vector:]

    return true_box_class, true_box_xyzt, true_box_hwd, true_box_conf


def extract_ground_event_volume_pred(y_pred, categories, box_vector):
    """
    Extracts class, position (xyzt), dimensions (hwd), and confidence from the predicted tensor.
    Model output format: [categories (softmax)] + [x, y, z, t, h, w, d, c (sigmoid)]

    Args:
        y_pred (torch.Tensor): Predicted tensor (batch, categories + box_vector)
        categories (int): Number of categories/classes.
        box_vector (int): Length of the box vector (8: x,y,z,t,h,w,d,c).
    Returns:
        tuple: (pred_box_class, pred_box_xyzt, pred_box_hwd, pred_box_conf)
    """
    # Model output format: categories first, then box_vector
    pred_box_class = y_pred[..., :categories]
    pred_box_xyzt = y_pred[..., categories:categories + 4]  # x, y, z, t
    pred_box_hwd = y_pred[..., categories + 4:categories + 7]  # h, w, d
    pred_box_conf = y_pred[..., categories + 7]   # c (confidence)

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


def calc_loss_class(true_box_class, pred_box_class, class_weights_dict=None):
    """
    Calculate N-class cross entropy loss.
    Since model already applies softmax, we use NLLLoss with log probabilities.
    """
    if class_weights_dict is not None:
        class_weights = torch.tensor(
            [class_weights_dict[i] for i in range(len(class_weights_dict))],
            dtype=pred_box_class.dtype,
            device=pred_box_class.device
        )
    else:
        class_weights = None

    # Get true class indices from one-hot
    true_class_indices = torch.argmax(true_box_class, dim=-1)

    # Convert softmax probabilities to log probabilities for NLLLoss
    log_probs = torch.log(pred_box_class + 1e-8)

    # NLLLoss expects (N, C) log probabilities and (N,) class indices
    loss_class = F.nll_loss(log_probs, true_class_indices, weight=class_weights, reduction="mean")

    return loss_class


class VolumeYoloLoss(nn.Module):
    def __init__(self, categories, box_vector, device, class_weights_dict=None, return_components=False):
        super().__init__()
        self.categories = categories
        self.box_vector = box_vector
        self.device = device
        self.class_weights_dict = class_weights_dict
        self.return_components = return_components

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Model predictions (batch, box_vector + categories, 1, 1, 1) or (batch, box_vector + categories)
            y_true: Ground truth YOLO labels (batch, box_vector + categories)
        Returns:
            If return_components=False: combined_loss
            If return_components=True: (combined_loss, loss_xyzt_hwd, loss_conf, loss_class)
        """
        # Squeeze spatial dimensions from predictions if present
        if y_pred.dim() > 2:
            y_pred = y_pred.squeeze(-1).squeeze(-1).squeeze(-1)

        y_true = y_true.to(y_pred.device)

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

        # Note: Model already applies sigmoid to box values and softmax to categories
        # So we don't apply activations here

        loss_xyzt_hwd = calc_loss_xyzt_hwd(
            true_box_xyzt, pred_box_xyzt, true_box_hwd, pred_box_hwd
        )
        loss_class = calc_loss_class(true_box_class, pred_box_class, class_weights_dict=self.class_weights_dict)
        loss_conf = compute_conf_loss_volume(true_box_conf, pred_box_conf)

        combined_loss = loss_xyzt_hwd + loss_conf + loss_class

        if self.return_components:
            return combined_loss, loss_xyzt_hwd, loss_conf, loss_class
        return combined_loss


__all__ = ["VolumeYoloLoss", "OneatClassificationLoss"]
