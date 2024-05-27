import torch
from torch import Tensor
from typing import Tuple


class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.area_intersect = torch.zeros(num_classes).to(device)
        self.area_pred_label = torch.zeros(num_classes).to(device)
        self.area_label = torch.zeros(num_classes).to(device)
        #self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        # self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)
        pred = pred[keep]
        target = target[keep]
        intersec = pred[pred==target]
        self.area_intersect += torch.histc(intersec, bins=(self.num_classes), min=0, max=self.num_classes-1)
        self.area_pred_label += torch.histc(pred, bins=(self.num_classes), min=0, max=self.num_classes-1)
        self.area_label += torch.histc(target, bins=(self.num_classes), min=0, max=self.num_classes-1)
        
    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.area_intersect / (self.area_label + self.area_pred_label - self.area_intersect)
        # ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        ious[ious.isnan()]=0.
        miou = ious.mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.area_intersect / (self.area_label + self.area_pred_label)
        # f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        f1[f1.isnan()]=0.
        mf1 = f1.mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.area_intersect / (self.area_label)
        # acc = self.hist.diag() / self.hist.sum(1)
        acc[acc.isnan()]=0.
        macc = acc.mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)
