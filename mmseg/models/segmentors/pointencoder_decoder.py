# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor


@MODELS.register_module()
class PointEncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSampel`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0,0,0])
            ] * inputs.shape[0]

        seg = self.inference(inputs, batch_img_metas)
        for data_sample in data_samples:
            self.intersect_and_unionall(seg, data_sample['location'], data_sample['spacing'],
                                        data_sample['seg_map_path'][-12:-4])


        @staticmethod
        def intersect_and_unionall(self, pred_label,
                                   location, spacing, name):
            """Calculate Intersection and Union.

            Args:
                pred_label (torch.tensor): Prediction segmentation map
                    or predict result filename. The shape is (H, W).
                label (torch.tensor): Ground truth segmentation map
                    or label filename. The shape is (H, W).
                num_classes (int): Number of categories.
                ignore_index (int): Index that will be ignored in evaluation.

            Returns:
                torch.Tensor: The intersection of prediction and ground truth
                    histogram on all classes.
                torch.Tensor: The union of prediction and ground truth histogram on
                    all classes.
                torch.Tensor: The prediction histogram on all classes.
                torch.Tensor: The ground truth histogram on all classes.
            """
            '''
            C,H,W,L = pred_label.shape
            topk = 9
            score, maxindex = pred_label.view(C,1,-1).topk(topk, dim=-1)
            score = score.cpu().numpy()
            maxindex = maxindex.cpu().numpy()



            point = np.zeros([topk, 3], np.int16)
            blurkeypoint = np.zeros([18, 3])

            for index in range(18):
                for i in range(topk):
                    point[i, 0] = L* W
                    point[i, 0] = maxindex[index, 0, i] // point[i, 0]
                    point[i, 1] = maxindex[index, 0, i] - point[i, 0] * W * L
                    point[i, 1] = point[i, 1] // L
                    point[i, 2] = maxindex[index, 0, i] - point[i, 0] * W * L - point[i, 1] * L
                    point[i, 2] = point[i, 2] % L
                a = 0
                b = 0
                c = 0
                tempscore = score[index, ...]
                tempscore = tempscore / np.sum(tempscore)
                for i in range(topk):
                    a = a + point[i, 0] * tempscore[0, i]
                    b = b + point[i, 1] * tempscore[0, i]
                    c = c + point[i, 2] * tempscore[0, i]
                blurkeypoint[index, 0] = a
                blurkeypoint[index, 1] = b
                blurkeypoint[index, 2] = c

            '''

            result = np.zeros([18, 4])
            keypoint = np.zeros([18, 3])
            for i in range(18):
                onemax = np.max(pred_label[i, ...])
                prepoint = np.where(pred_label[i, ...] == onemax)
                '''
                temp = pred_label[i, prepoint[0][0],prepoint[1][0],prepoint[2][0]]
                pred_label[i, prepoint[0][0],prepoint[1][0],prepoint[2][0]]=0
                twopoint=np.where(pred_label[i,...]==np.max(pred_label[i,...]))
                pred_label[i, prepoint[0][0], prepoint[1][0], prepoint[2][0]] = temp
                a = np.sqrt(np.square(twopoint[0][0] - prepoint[0][0]) + np.square(twopoint[1][0] - prepoint[1][0]) + np.square(twopoint[2][0] - prepoint[2][0]))
                x1=0.25 * (twopoint[0][0] - prepoint[0][0]) / a
                y1=0.25 * (twopoint[1][0] - prepoint[1][0]) / a
                z1=0.25 * (twopoint[2][0] - prepoint[2][0]) / a    
                '''
                keypoint[i, 0] = prepoint[0][0]
                keypoint[i, 1] = prepoint[1][0]
                keypoint[i, 2] = prepoint[2][0]
                result[i, 0] = np.sqrt(np.square(prepoint[0][0] * spacing[0] - location[i, 0]) + np.square(
                    prepoint[1][0] * spacing[1] - location[i, 1]) + np.square(
                    prepoint[2][0] * spacing[2] - location[i, 2]))
                result[i, 1] = (np.abs(prepoint[0][0] * spacing[0] - location[i, 0]) + np.abs(
                    prepoint[1][0] * spacing[1] - location[i, 1]) + np.abs(
                    prepoint[2][0] * spacing[2] - location[i, 2])) / 3.0
                result[i, 2] = (np.square(prepoint[0][0] * spacing[0] - location[i, 0]) + np.square(
                    prepoint[1][0] * spacing[1] - location[i, 1])) + np.square(
                    prepoint[2][0] * spacing[2] - location[i, 2]) / 3.0
                # result[i,3] = np.sqrt(np.square((prepoint[0][0]+x1) * spacing[0] - location[i,0]) + np.square(((prepoint[1][0]+y1) * spacing[1] - location[i,1])) + np.square(((prepoint[2][0]+z1) * spacing[2] - location[i,2])))
            blurkeypoint = self.refine_keypoints_dark3d(keypoint, pred_label, spacing)
            # blurkeypoint = self.refine_keypoints_decoder(keypoint, pred_label, spacing)
            # self.savepoint(keypoint, name, spacing)
            # reffine = self.refine_keypoints(keypoint,pred_label)
            for i in range(18):
                result[i, 3] = np.sqrt(np.square(blurkeypoint[i, 0] * spacing[0] - location[i, 0]) + np.square(
                    blurkeypoint[i, 1] * spacing[1] - location[i, 1]) + np.square(
                    blurkeypoint[i, 2] * spacing[2] - location[i, 2]))
                # result[i,3] = np.sqrt(np.square(reffine[i,0]* spacing[0] - location[i,0]) + np.square(reffine[i,1] * spacing[1] - location[i,1]) + np.square(reffine[i,2] * spacing[2] - location[i,2]))
            result = np.abs(result)
            # self.savepoint(blurkeypoint, name, spacing)
            self.savepoint(result, name, spacing)


    @staticmethod
    def refine_keypoints(keypoints: np.ndarray,
                         heatmaps: np.ndarray) -> np.ndarray:
        """Refine keypoint predictions by moving from the maximum towards the
        second maximum by 0.25 pixel. The operation is in-place.

        Note:

            - instance number: N
            - keypoint number: K
            - keypoint dimension: D
            - heatmap size: [W, H]

        Args:
            keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
            heatmaps (np.ndarray): The heatmaps in shape (K, H, W)

        Returns:
            np.ndarray: Refine keypoint coordinates in shape (N, K, D)
        """
        K = keypoints.shape[0]
        L, H, W = heatmaps.shape[1:]

        for k in range(K):
            z, y, x = keypoints[k, :3].astype(int)

            if 1 < x < W - 1 and 0 < y < H and 0 < z < L:
                dx = heatmaps[k, z, y, x + 1] - heatmaps[k, z, y, x - 1]
            else:
                dx = 0.

            if 1 < y < H - 1 and 0 < x < W and 0 < z < L:
                dy = heatmaps[k, z, y + 1, x] - heatmaps[k, z, y - 1, x]
            else:
                dy = 0.
            if 1 < y < H and 0 < x < W and 0 < z < L - 1:
                dz = heatmaps[k, z + 1, y, x] - heatmaps[k, z - 1, y, x]
            else:
                dz = 0.
            keypoints[k] += np.sign([dz, dy, dx], dtype=np.float32) * 0.25

        return keypoints

    @staticmethod
    def refine_keypoints_decoder(keypoint: np.ndarray, heatmaps: np.ndarray, spacing
                                 ) -> np.ndarray:
        _K, L, H, W, = heatmaps.shape
        K = _K // 4

        d = 5
        tao = 1
        l = 7

        size = np.zeros([18, ])
        for index in range(18):
            rawsize = np.zeros([2, 3], dtype=np.int16)
            rawsize[0, 0] = max(0, keypoint[index, 0] - l)
            rawsize[0, 1] = max(0, keypoint[index, 1] - l)
            rawsize[0, 2] = max(0, keypoint[index, 2] - l)
            rawsize[1, 0] = min(L, keypoint[index, 0] + l + 1)
            rawsize[1, 1] = min(H, keypoint[index, 1] + l + 1)
            rawsize[1, 2] = min(W, keypoint[index, 2] + l + 1)
            softh = np.zeros(
                [rawsize[1, 0] - rawsize[0, 0], rawsize[1, 1] - rawsize[0, 1], rawsize[1, 2] - rawsize[0, 2]])
            for x in range(rawsize[1, 0] - rawsize[0, 0]):
                for y in range(rawsize[1, 1] - rawsize[0, 1]):
                    for z in range(rawsize[1, 2] - rawsize[0, 2]):
                        x1 = (x + rawsize[0, 0])
                        y1 = (y + rawsize[0, 1])
                        z1 = (z + rawsize[0, 2])
                        softh[x, y, z] = np.exp(heatmaps[index, x1, y1, z1] * 10)
            softh = softh / np.sum(softh)
            a = 0
            b = 0
            c = 0
            for x in range(rawsize[1, 0] - rawsize[0, 0]):
                for y in range(rawsize[1, 1] - rawsize[0, 1]):
                    for z in range(rawsize[1, 2] - rawsize[0, 2]):
                        a = tao * softh[x, y, z] * x + a
                        b = tao * softh[x, y, z] * y + b
                        c = tao * softh[x, y, z] * z + c
            keypoint[index, 0] = keypoint[index, 0] + a - l
            keypoint[index, 1] = keypoint[index, 1] + b - l
            keypoint[index, 2] = keypoint[index, 2] + c - l
        return keypoint

    @staticmethod
    def refine_keypoints_dark3d(keypoints: np.ndarray, heatmaps: np.ndarray, spacing
                                ) -> np.ndarray:
        """Refine keypoint predictions using distribution aware coordinate
        decoding. See `Dark Pose`_ for details. The operation is in-place.

        Note:

            - instance number: N
            - keypoint number: K
            - keypoint dimension: D
            - heatmap size: [W, H]

        Args:
            keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
            heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
            blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
                modulation

        Returns:
            np.ndarray: Refine keypoint coordinates in shape (N, K, D)

        .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
        """
        K = keypoints.shape[0]
        L, H, W = heatmaps.shape[1:]
        import scipy.ndimage.filters as filter
        kernel = 63
        # modulate heatmaps
        border = (kernel - 1) // 2
        # kernel = 2 * int（ truncate * sigma + 0.5）+ 1
        truncate = 3
        sigma = 3

        # spacing[0] =spacing[1]=spacing[2]=1.0
        border1 = int(truncate * sigma / spacing[0] + 0.5)
        border2 = int(truncate * sigma / spacing[1] + 0.5)
        border3 = int(truncate * sigma / spacing[2] + 0.5)

        # border1=border
        # border2=border
        # border3=border

        import numpy as np
        for k in range(K):
            origin_max = np.max(heatmaps[k])

            dr = np.zeros((L + 2 * border1, H + 2 * border2, W + 2 * border3), dtype=np.float32)
            dr[border1:-border1, border2:-border2, border3:-border3] = heatmaps[k].copy()
            '''
            dr[0:border1,border2:-border2, border3:-border3] = heatmaps[k][0:border1,:,:][::-1,:,:]
            dr[-border1:, border2:-border2, border3:-border3] = heatmaps[k][-border1:, :, :][::-1, :, :]


            dr[border1:-border1,0:border2, border3:-border3] = heatmaps[k][:,0:border2,:][:,::-1,:]
            dr[border1:-border1:, -border2:, border3:-border3] = heatmaps[k][:, -border2:, :][:, ::-1, :]

            dr[border1:-border1,border2:-border2, 0:border3] = heatmaps[k][:,:,0:border3][:,:,::-1]
            dr[border1:-border1:, border2:-border2, -border3:] = heatmaps[k][:, :, -border3:][:, :, ::-1]
            '''
            dr = filter.gaussian_filter(dr, truncate=truncate, sigma=sigma, spacings=spacing)
            '''

            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            x = np.arange(start=-border1, stop=border1, step=1) + int(keypoints[k, 0])
            y = np.arange(start=-border2, stop=border2, step=1) + int(keypoints[k, 1])
            X, Y = np.meshgrid(x, y)
            Z = np.zeros([X.shape[0], X.shape[1]])
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = dr[X[i, j], Y[i, j], int(keypoints[1, 2]) + border3]

            ax.plot_surface(X, Y, Z, alpha=0.9, cstride=1, rstride=1, cmap='rainbow')
            #plt.show()
            # dr = filter.gaussian_filter(dr, (kernel, kernel), 0)
            dr = filter.gaussian_filter(dr, truncate=truncate, sigma=sigma,spacings=spacing)

            fig = plt.figure()
            ax = plt.axes(projection="3d")
            x = np.arange(start=-border1, stop=border1, step=1) + int(keypoints[k, 0])
            y = np.arange(start=-border2, stop=border2, step=1) + int(keypoints[k, 1])
            X, Y = np.meshgrid(x, y)
            Z = np.zeros([X.shape[0], X.shape[1]])
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = dr[X[i, j], Y[i, j], int(keypoints[1, 2]) + border3]

            ax.plot_surface(X, Y, Z, alpha=0.9, cstride=1, rstride=1, cmap='rainbow')
            '''
            # plt.show()

            heatmaps[k] = dr[border1:-border1, border2:-border2, border3:-border3].copy()
            heatmaps[k] *= origin_max / np.max(heatmaps[k])

        np.maximum(heatmaps, 1e-10, heatmaps)
        np.log(heatmaps, heatmaps)

        for k in range(K):
            z, y, x = keypoints[k, :3].astype(int)
            if 1 < x < W - 2 and 1 < y < H - 2 and 1 < z < L - 2:
                '''
                dx = spacing[2] * (heatmaps[k, z, y, x + 1] - heatmaps[k, z, y, x - 1]) / ( 2)
                dy = spacing[1] * (heatmaps[k, z, y + 1, x] - heatmaps[k, z, y - 1, x]) / ( 2)
                dz = spacing[0] * (heatmaps[k, z + 1, y, x] - heatmaps[k, z - 1, y, x]) / (2)

                dxx = spacing[2] * spacing[2] * (
                              heatmaps[k, z, y, x + 2] - 2 * heatmaps[k, z, y, x] +
                              heatmaps[k, z, y, x - 2]) / (4)
                dxy = spacing[2] * spacing[1] *(
                              heatmaps[k, z, y + 1, x + 1] - heatmaps[k, z, y - 1, x + 1] -
                              heatmaps[k, z, y + 1, x - 1] + heatmaps[k, z, y - 1, x - 1]) / (
                                   4)
                dyy = spacing[1] * spacing[1] * (
                              heatmaps[k, z, y + 2, x] - 2 * heatmaps[k, z, y, x] +
                              heatmaps[k, z, y - 2, x]) / (4)
                dzz = spacing[0] * spacing[0] * (
                              heatmaps[k, z + 2, y, x] - 2 * heatmaps[k, z, y, x] +
                              heatmaps[k, z - 2, y, x]) / (4)
                dxz = spacing[2] * spacing[0] *(
                              heatmaps[k, z + 1, y, x + 1] - heatmaps[k, z - 1, y, x + 1] -
                              heatmaps[k, z + 1, y, x - 1] + heatmaps[k, z - 1, y, x - 1]) / (
                                   4)
                dyz = spacing[1] * spacing[0] * (
                              heatmaps[k, z + 1, y + 1, x] - heatmaps[k, z + 1, y - 1, x] -
                              heatmaps[k, z - 1, y + 1, x] + heatmaps[k, z - 1, y - 1, x]) / (
                                  4)
                '''
                dx = (heatmaps[k, z, y, x + 1] - heatmaps[k, z, y, x - 1]) / (spacing[2] * 2)
                dy = (heatmaps[k, z, y + 1, x] - heatmaps[k, z, y - 1, x]) / (spacing[1] * 2)
                dz = (heatmaps[k, z + 1, y, x] - heatmaps[k, z - 1, y, x]) / (spacing[0] * 2)

                dxx = (
                              heatmaps[k, z, y, x + 2] - 2 * heatmaps[k, z, y, x] +
                              heatmaps[k, z, y, x - 2]) / (spacing[2] * spacing[2] * 4)
                dxy = (
                              heatmaps[k, z, y + 1, x + 1] - heatmaps[k, z, y - 1, x + 1] -
                              heatmaps[k, z, y + 1, x - 1] + heatmaps[k, z, y - 1, x - 1]) / (
                                  spacing[2] * spacing[1] * 4)
                dyy = (
                              heatmaps[k, z, y + 2, x] - 2 * heatmaps[k, z, y, x] +
                              heatmaps[k, z, y - 2, x]) / (spacing[1] * spacing[1] * 4)
                dzz = (
                              heatmaps[k, z + 2, y, x] - 2 * heatmaps[k, z, y, x] +
                              heatmaps[k, z - 2, y, x]) / (spacing[0] * spacing[0] * 4)
                dxz = (
                              heatmaps[k, z + 1, y, x + 1] - heatmaps[k, z - 1, y, x + 1] -
                              heatmaps[k, z + 1, y, x - 1] + heatmaps[k, z - 1, y, x - 1]) / (
                                  spacing[2] * spacing[0] * 4)
                dyz = (
                              heatmaps[k, z + 1, y + 1, x] - heatmaps[k, z + 1, y - 1, x] -
                              heatmaps[k, z - 1, y + 1, x] + heatmaps[k, z - 1, y - 1, x]) / (
                                  spacing[1] * spacing[0] * 4)

                '''
                dx = 0.5 * (heatmaps[k, z, y, x + 1] - heatmaps[k, z, y, x - 1])
                dy = 0.5 * (heatmaps[k, z, y + 1, x] - heatmaps[k, z, y - 1, x])
                dz = 0.5 * (heatmaps[k, z + 1, y, x] - heatmaps[k, z - 1, y, x])

                dxx = 0.25 * (
                        heatmaps[k, z, y, x + 2] - 2 * heatmaps[k, z, y, x] +
                        heatmaps[k, z, y, x - 2])
                dxy = 0.25 * (
                        heatmaps[k, z, y + 1, x + 1] - heatmaps[k, z, y - 1, x + 1] -
                        heatmaps[k, z, y + 1, x - 1] + heatmaps[k, z, y - 1, x - 1])
                dyy = 0.25 * (
                        heatmaps[k, z, y + 2, x] - 2 * heatmaps[k, z, y, x] +
                        heatmaps[k, z, y - 2, x])
                dzz = 0.25 * (
                        heatmaps[k, z + 2, y, x] - 2 * heatmaps[k, z, y, x] +
                        heatmaps[k, z - 2, y, x])
                dxz = 0.25 * (
                        heatmaps[k, z + 1, y, x + 1] - heatmaps[k, z - 1, y, x + 1] -
                        heatmaps[k, z + 1, y, x - 1] + heatmaps[k, z - 1, y, x - 1])
                dyz = 0.25 * (
                        heatmaps[k, z + 1, y + 1, x] - heatmaps[k, z + 1, y - 1, x] -
                        heatmaps[k, z - 1, y + 1, x] + heatmaps[k, z - 1, y - 1, x])
                '''
                derivative = np.array([[dx], [dy], [dz]])
                hessian = np.array([[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]])
                if np.linalg.det(hessian) != 0:
                    hessianinv = np.linalg.inv(hessian)
                    offset = -hessianinv @ derivative
                    offset = np.squeeze(np.array(offset.T), axis=0)
                    offset = np.flipud(offset)
                    # offset = offset/spacing
                    keypoints[k, :3] += offset
        return keypoints

    @staticmethod
    def savepoint(point, name, spacing):
        with open('/home/zrs/result/' + name + '.txt', 'w') as f:
            for i in range(18):
                # f.write(str(point[i,0] * spacing[0]) + '\t' + str(point[i,1] * spacing[1]) + '\t' + str(point[i,2] * spacing[2]) + '\n')
                f.write(str(point[i, 3]) + '\n')


    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inferenceold(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride, l_stride = self.test_cfg.stride
        h_crop, w_crop, l_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img, l_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        l_grids = max(l_img - l_crop + l_stride - 1, 0) // l_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img, l_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img, l_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                for l_idx in range(l_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    z1 = l_idx * l_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    z2 = min(z1 + l_crop, l_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    z1 = max(z2 - l_crop, 0)
                    crop_img = inputs[:, :, y1:y2, x1:x2, z1:z2]
                    # change the image shape to patch shape
                    batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                    # the output of encode_decode is seg logits tensor map
                    # with shape [N, C, H, W]
                    crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)

                    preds += F.pad(crop_seg_logit,
                                   (int(z1), int(preds.shape[4] - z2),int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2, z1:z2] += 1

        #assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride, l_stride = self.test_cfg.stride
        h_crop, w_crop, l_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img, l_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        l_grids = max(l_img - l_crop + l_stride - 1, 0) // l_stride + 1
        preds = np.zeros([batch_size, out_channels, h_img, w_img, l_img], np.float32)
        count_mat = np.zeros([batch_size, 1, h_img, w_img, l_img], np.float32)
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                for l_idx in range(l_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    z1 = l_idx * l_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    z2 = min(z1 + l_crop, l_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    z1 = max(z2 - l_crop, 0)
                    crop_img = inputs[:, :, y1:y2, x1:x2, z1:z2]
                    # change the image shape to patch shape
                    batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                    # the output of encode_decode is seg logits tensor map
                    # with shape [N, C, H, W]
                    crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                    #crop_seg_logit = crop_seg_logit.cpu().numpy()
                    preds[:,:,int(y1):int(y2),int(z1):int(z2),int(x1):int(x2)] += crop_seg_logit.cpu().numpy()[:,:,:,:,:]

                    count_mat[:, :, y1:y2, x1:x2, z1:z2] += 1

        #assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat
        return seg_logits
    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
