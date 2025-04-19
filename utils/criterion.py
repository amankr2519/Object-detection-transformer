import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

class HungarianMatcher(nn.Module):
    def __init__(self, class_cost=1, bbox_cost=5, giou_cost=2):
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]

        indices = []

        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(out_bbox[b], tgt_bbox)

            cost_matrix = self.class_cost * cost_class + \
                          self.bbox_cost * cost_bbox + \
                          self.giou_cost * cost_giou

            indices_b = linear_sum_assignment(cost_matrix.cpu())
            indices.append((torch.as_tensor(indices_b[0], dtype=torch.int64),
                            torch.as_tensor(indices_b[1], dtype=torch.int64)))

        return indices

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef=0.1, losses=["labels", "boxes"]):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.weight_dict = {
            "loss_ce": 1,
            "loss_bbox": 5,
            "loss_giou": 2
        }

    def loss_labels(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs["pred_logits"]

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, ignore_index=self.num_classes)
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))

        return {
            "loss_bbox": loss_bbox.sum() / src_boxes.shape[0],
            "loss_giou": loss_giou.sum() / src_boxes.shape[0],
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        losses = {}
        for loss in self.losses:
            l_func = getattr(self, f"loss_{loss}")
            losses.update(l_func(outputs, targets, indices))

        return losses
