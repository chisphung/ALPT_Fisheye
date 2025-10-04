# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VertexNetLoss(nn.Module):
    def __init__(self):
        super(VertexNetLoss, self).__init__()
        self.lambda_cls = 4.0
        self.lambda_box = 0.8
        self.lambda_vert = 0.1
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, predictions, targets):
        """
        Args:
            predictions: A list of 4 tensors from the VertexNet heads.
            targets: A list containing target tensors for cls, box, and vertices.
                     Shape of targets: [cls_targets, box_targets, vert_targets, pos_anchors_mask]
        """
        # This is a simplified loss calculation. A full implementation would
        # involve matching predictions to ground truth across all anchor boxes.
        cls_preds, box_preds, vert_preds = predictions # Simplified unpacking
        cls_targets, box_targets, vert_targets, pos_mask = targets

        # Classification Loss
        loss_cls = self.cross_entropy(cls_preds, cls_targets)
        # Apply hard negative mining as described in the paper 
        
        # Apply mask for positive anchors to regression losses
        pos_mask = pos_mask.unsqueeze(-1).expand_as(box_preds)
        
        # Box Loss
        loss_box = self.smooth_l1(box_preds[pos_mask], box_targets[pos_mask]).sum()

        # Vertex Loss
        loss_vert = self.smooth_l1(vert_preds[pos_mask], vert_targets[pos_mask]).sum()

        num_positives = pos_mask.sum()
        if num_positives == 0:
            return torch.tensor(0.0)

        # Final Weighted Loss
        total_loss = (self.lambda_cls * loss_cls.sum() + \
                      self.lambda_box * loss_box + \
                      self.lambda_vert * loss_vert) / num_positives
                      
        return total_loss