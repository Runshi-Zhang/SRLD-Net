from typing import Optional

import torch

import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor
from mmseg.registry import MODELS

@MODELS.register_module()

class CombinedLoss(nn.Module):
    """MSE loss for combined target.

    CombinedTarget: The combination of classification target
    (response map) and regression target (offset map).
    Paper ref: Huang et al. The Devil is in the Details: Delving into
    Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 loss_weight: float = 1.,
                 loss_name='loss_udp'):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_channels: C
            - heatmaps height: H
            - heatmaps weight: W
            - num_keypoints: K
            Here, C = 3 * K

        Args:
            output (Tensor): The output feature maps with shape [B, C, H, W].
            target (Tensor): The target feature maps with shape [B, C, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        batch_size = output.size(0)
        num_channels = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        loss = 0.
        num_joints = num_channels // 4
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 4].squeeze()
            heatmap_gt = heatmaps_gt[idx * 4].squeeze()
            offset_x_pred = heatmaps_pred[idx * 4 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 4 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 4 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 4 + 2].squeeze()
            offset_z_pred = heatmaps_pred[idx * 4 + 3].squeeze()
            offset_z_gt = heatmaps_gt[idx * 4 + 3].squeeze()
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None]
                heatmap_pred = heatmap_pred * target_weight
                heatmap_gt = heatmap_gt * target_weight
            # classification loss
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            # regression loss
            loss += 0.5 * self.criterion(heatmap_gt * offset_x_pred,
                                         heatmap_gt * offset_x_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_y_pred,
                                         heatmap_gt * offset_y_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_z_pred,
                                         heatmap_gt * offset_z_gt)
        return loss / num_joints * self.loss_weight

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
