from typing import Optional

import torch

import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor
from mmseg.registry import MODELS

@MODELS.register_module()
class FocalHeatLoss(nn.Module):
    """MSE loss for heatmaps.

       Args:
           use_target_weight (bool): Option to use weighted MSE loss.
               Different joint types may have different target weights.
               Defaults to ``False``
           skip_empty_channel (bool): If ``True``, heatmap channels with no
               non-zero value (which means no visible ground-truth keypoint
               in the image) will not be used to calculate the loss. Defaults to
               ``False``
           loss_weight (float): Weight of the loss. Defaults to 1.0
       """

    def __init__(self,
                 alpha: int = 2,
                 beta: int = 4,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,
                 loss_weight: float = 1.,
                 loss_name='loss_heatfocal'):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.alpha = alpha
        self.beta = beta

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        """Calculate the modified focal loss for heatmap prediction.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()


        neg_weights = torch.pow(1 - target, self.beta)

        pos_loss = torch.log(output) * torch.pow(1 - output,
                                                 self.alpha) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(
            output, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss.sum()
        else:
            loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss * self.loss_weight

    def _get_mask(self, target: Tensor, target_weights: Optional[Tensor],
                  mask: Optional[Tensor]) -> Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert (mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1
                for d_m, d_t in zip(mask.shape, target.shape))), (
                f'mask and target have mismatched shapes {mask.shape} v.s.'
                f'{target.shape}')

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                'target_weights and target have mismatched shapes '
                f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape +
                                        (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any()
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask

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
