"""by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from src.core import register


__all__ = [
    "DETRPostProcessor", "RMDETRPostProcessor"
]


@register
class DETRPostProcessor(nn.Module):
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category",
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes):

        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ..data.coco import mscoco_label2category

            labels = (
                torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in labels.flatten()]
                )
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self

    @property
    def iou_types(
        self,
    ):
        return ("bbox",)

@register
class RMDETRPostProcessor(nn.Module):
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category",
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes):

        logits, boxes, masks = outputs["pred_logits"], outputs["pred_boxes"], outputs["pred_masks"]
        
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        b, q, h, w = masks.shape
        mask_pred = masks.flatten(2)  # [B, num_queries, H*W]
        # print(mask_pred.shape)
        if self.use_focal_loss:
            scores = F.sigmoid(logits) # 8, 300, 80
            # mask_pred = F.sigmoid(mask_pred)
            # scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            # labels = index % self.num_classes
            # index = index // self.num_classes
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)  # Query 차원에서 직접 정렬
            labels = index % self.num_classes  # 클래스 인덱스
            index = index // self.num_classes  # Query 인덱스
            
            
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )
            mask_pred = mask_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, mask_pred.shape[-1])
            )
            mask_pred = mask_pred.reshape(b,self.num_top_queries,h,w)
            # print(mask_pred.shape)
            # masks = (masks > 0).float()
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            masks = masks
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )
                
                
                # batch_size, num_queries, H, W = masks.shape
                # _, num_top_queries = index.shape

                # batch_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, num_top_queries).to(masks.device)
                # row_indices = index

                # batch_indices = batch_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
                # row_indices = row_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

                # masks = masks[batch_indices, torch.arange(num_top_queries).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(batch_size, 1, H, W).to(masks.device), row_indices, :]

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ..data.coco import mscoco_label2category

            labels = (
                torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in labels.flatten()]
                )
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        mask_pred = F.interpolate(mask_pred, size=(640, 640), mode="bilinear", align_corners=False)
        # mask_pred = (mask_pred.sigmoid() > 0.5).cpu()
        for lab, box, sco, mask, tgt_size in zip(labels, boxes, scores, mask_pred, orig_target_sizes):  # Iterate through masks as well
            mask = F.interpolate(mask.unsqueeze(1), size=(tgt_size[1], tgt_size[0]), mode="bilinear", align_corners=False).squeeze(1)
            mask = mask.sigmoid()
            result = dict(labels=lab, boxes=box, scores=sco, masks=mask)  # Add mask to the result
            results.append(result)

        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self

    @property
    def iou_types(
        self,
    ):
        return ("bbox", "segm")
        # return ("bbox",)