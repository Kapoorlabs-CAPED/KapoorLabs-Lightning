import torch
import torch.nn as nn
import torch.nn.functional as F

class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P

    def forward(self, gts, preds):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2




# Simplified extract functions
def extract_ground_event_volume_truth(y_true, categories, box_vector):
    """
    Extracts class, position (xyz), dimensions (whd), and confidence from ground truth tensor.
    Args:
        y_true (torch.Tensor): Ground truth tensor.
        categories (int): Number of categories/classes.
        box_vector (int): Length of the box vector.
    Returns:
        tuple: (true_box_class, true_box_xyz, true_box_whd, true_box_conf)
    """
    true_box_class = y_true[..., :categories]
    true_nboxes = y_true[..., categories:].view(-1, 1, box_vector)
    true_box_xyz = true_nboxes[..., :3]
    true_box_whd = true_nboxes[..., 3:6]
    true_box_conf = true_nboxes[..., -1]

    return true_box_class, true_box_xyz, true_box_whd, true_box_conf


def extract_ground_event_volume_pred(y_pred, categories, box_vector):
    """
    Extracts class, position (xyz), dimensions (whd), and confidence from the predicted tensor.
    Args:
        y_pred (torch.Tensor): Predicted tensor.
        categories (int): Number of categories/classes.
        box_vector (int): Length of the box vector.
    Returns:
        tuple: (pred_box_class, pred_box_xyz, pred_box_whd, pred_box_conf)
    """
    pred_box_class = y_pred[..., :categories]
    pred_nboxes = y_pred[..., categories:].view(-1, 1, box_vector)
    pred_box_xyz = pred_nboxes[..., :3]
    pred_box_whd = pred_nboxes[..., 3:6]
    pred_box_conf = pred_nboxes[..., -1]

    return pred_box_class, pred_box_xyz, pred_box_whd, pred_box_conf


# Loss functions
def compute_conf_loss_volume(pred_box_whd, true_box_whd, pred_box_xyz, true_box_xyz, true_box_conf, pred_box_conf):
    """
    Computes the confidence loss for 3D volume predictions using IoU.
    """
    intersect_whd = torch.max(
        torch.zeros_like(pred_box_whd),
        (pred_box_whd + true_box_whd) / 2 - torch.abs(pred_box_xyz - true_box_xyz)
    )
    intersect_volume = intersect_whd[..., 0] * intersect_whd[..., 1] * intersect_whd[..., 2]
    true_volume = true_box_whd[..., 0] * true_box_whd[..., 1] * true_box_whd[..., 2]
    pred_volume = pred_box_whd[..., 0] * pred_box_whd[..., 1] * pred_box_whd[..., 2]
    union_volume = pred_volume + true_volume - intersect_volume
    iou = intersect_volume / (union_volume + 1e-6)

    best_ious = iou.max(dim=-1).values
    loss_conf = F.mse_loss(true_box_conf * best_ious, pred_box_conf, reduction='sum')

    return loss_conf


def calc_loss_xyzwhd(true_box_xyz, pred_box_xyz, true_box_whd, pred_box_whd):
    """
    Calculates the loss for position (xyz) and dimensions (whd).
    """
    loss_xyz = torch.sum((true_box_xyz - pred_box_xyz) ** 2, dim=-1).sum()
    loss_whd = torch.sum((torch.sqrt(true_box_whd + 1e-6) - torch.sqrt(pred_box_whd + 1e-6)) ** 2, dim=-1).sum()
    loss_xyzwhd = loss_xyz + loss_whd

    return loss_xyzwhd


def calc_loss_class(true_box_class, pred_box_class):
    """
    Calculates the classification loss using categorical cross-entropy.
    """
    loss_class = F.cross_entropy(pred_box_class, torch.argmax(true_box_class, dim=-1), reduction='mean')
    return loss_class


# Combined loss function
class VolumeYoloLoss(nn.Module):
    def __init__(self, categories, box_vector, device):
        super().__init__()
        self.categories = categories
        self.box_vector = box_vector
        self.device = device

    def forward(self, y_true, y_pred):

        
        y_true = y_true.reshape(y_pred.shape)
        print(y_true.shape, y_pred.shape)
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        
        true_box_class, true_box_xyz, true_box_whd, true_box_conf = extract_ground_event_volume_truth(
            y_true, self.categories, self.box_vector)
        pred_box_class, pred_box_xyz, pred_box_whd, pred_box_conf = extract_ground_event_volume_pred(
            y_pred, self.categories, self.box_vector)

        loss_xyzwhd = calc_loss_xyzwhd(true_box_xyz, pred_box_xyz, true_box_whd, pred_box_whd)
        loss_class = calc_loss_class(true_box_class, pred_box_class)
        loss_conf = compute_conf_loss_volume(pred_box_whd, true_box_whd, pred_box_xyz, true_box_xyz, true_box_conf, pred_box_conf)

        combined_loss = loss_xyzwhd + loss_conf + loss_class
        return combined_loss
