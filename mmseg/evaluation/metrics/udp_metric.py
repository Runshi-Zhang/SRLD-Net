import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS


@METRICS.register_module()
class UDPMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = 18
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                self.results.append(
                    self.intersect_and_union(pred_label, label, data_sample['location'], data_sample['spacing'], data_sample['seg_map_path'][-12:-4]))


    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 72
        re = np.array(results,dtype=float)
        result = np.zeros([2,18,re.shape[1]])


        print_log('per class results:', logger)
        metrics = dict()
        for i in range(18):
            #result[i,0] = sum(results[4*i]) / len(results[4*i])
            #result[i, 1] = sum(results[4 * i + 1]) / len(results[4 * i + 1])
            #result[i, 2] = sum(results[4 * i + 2]) / len(results[4 * i + 2])
            #result[i, 3] = sum(results[4 * i + 3]) / len(results[4 * i + 3])
            result[0,i,:] = re[4*i,:]
            result[1, i, :] = re[4 * i + 3, :]
            #result[2, i, :] = re[4 * i + 1, :]
            #result[i, 1] = sum(results[4 * i + 1]) / len(results[4 * i + 1])
            #result[i, 2] = sum(results[4 * i + 2]) / len(results[4 * i + 2])
            #result[i, 3] = sum(results[4 * i + 3]) / len(results[4 * i + 3])
            metrics['mean' + str(i + 1)] = np.mean(result[0,i,:])
            metrics['std' + str(i + 1)] = np.std(result[0,i, :])
            #metrics['meandekr' + str(i + 1)] = np.mean(result[1,i,:])
            #metrics['stddekr' + str(i + 1)] = np.std(result[1,i, :])
            #metrics['xyzloss' + str(i + 1)] = result[i, 1]
            #print_log('\n' + str(result[i]), logger=logger)
        metrics['mean'] = np.mean(result[0,:,:])
        metrics['std'] = np.std(result[0,:, :])
        metrics['aftermean'] = np.mean(result[1,:, :])
        metrics['afterstd'] = np.std(result[1,:, :])
        return metrics

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
        #kernel = 2 * int（ truncate * sigma + 0.5）+ 1
        truncate = 3
        sigma = 3

        #spacing[0] =spacing[1]=spacing[2]=1.0
        border1=int(truncate*sigma/spacing[0] + 0.5)
        border2=int(truncate*sigma/spacing[1] + 0.5)
        border3=int(truncate*sigma/spacing[2] + 0.5)

        #border1=border
        #border2=border
        #border3=border

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
            #plt.show()


            heatmaps[k] = dr[border1:-border1, border2:-border2,border3:-border3].copy()
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
                        heatmaps[k, z, y + 1, x - 1] + heatmaps[k, z, y - 1, x - 1]) / (spacing[2] * spacing[1] * 4)
                dyy = (
                        heatmaps[k, z, y + 2, x] - 2 * heatmaps[k, z, y, x] +
                        heatmaps[k, z, y - 2, x]) / ( spacing[1] * spacing[1] * 4)
                dzz = (
                        heatmaps[k, z + 2, y, x] - 2 * heatmaps[k, z, y, x] +
                        heatmaps[k, z - 2, y, x]) / ( spacing[0] * spacing[0] * 4)
                dxz = (
                        heatmaps[k, z + 1, y, x + 1] - heatmaps[k, z - 1, y, x + 1] -
                        heatmaps[k, z + 1, y, x - 1] + heatmaps[k, z - 1, y, x - 1]) / ( spacing[2] * spacing[0] * 4)
                dyz = (
                        heatmaps[k, z + 1, y + 1, x] - heatmaps[k, z + 1, y - 1, x] -
                        heatmaps[k, z - 1, y + 1, x] + heatmaps[k, z - 1, y - 1, x]) / (spacing[1] * spacing[0] * 4)

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
                    #offset = offset/spacing
                    keypoints[k, :3] += offset
        return keypoints

    @staticmethod
    def savepoint(point,name,spacing):
        with open('/home/zrs/result/'+ name +'.txt', 'w') as f:
            for i in range(18):
                f.write(str(point[i,0] * spacing[0]) + '\t' + str(point[i,1] * spacing[1]) + '\t' + str(point[i,2] * spacing[2]) + '\n')
                #f.write(str(point[i,3])+ '\n')
    def intersect_and_union(self, pred_label: torch.tensor, label: torch.tensor,
                            location,spacing, name):
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

        pred_label = pred_label.cpu().numpy()

        _K, L, H, W, = pred_label.shape
        K = _K // 4

        import scipy.ndimage.filters as filter

        truncate = 3
        sigma = 6

        keypoint = np.zeros([18, 3])
        for index in range(18):
            heatmaps_flatten = pred_label[4 * index, ...]
            heatmaps_flatten = filter.gaussian_filter(heatmaps_flatten, truncate=truncate,
                                                      sigma=sigma, spacings=[1, 1, 1])
            heatmaps_flatten = heatmaps_flatten.reshape(1, -1)
            dd = np.argmax(heatmaps_flatten, axis=1)
            z, y, x = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(L, H, W))
            keypoint[index, 0] = x
            keypoint[index, 1] = y
            keypoint[index, 2] = z

        radius = 48 * 0.0546875
        x_off = pred_label[1::4].flatten() * radius
        y_off = pred_label[2::4].flatten() * radius
        z_off = pred_label[3::4].flatten() * radius

        ind = (keypoint[:, 0] * 48 * H + keypoint[:, 1] * 48 + keypoint[:, 2]).flatten()
        ind += L * W * H * np.arange(0, K)
        ind = ind.astype(int)

        keypoint += np.stack((z_off[ind], y_off[ind], x_off[ind]), axis=-1)
        for i in range(18):
            keypoint[i, 0] = keypoint[i, 0] / 31.0 * 128
            keypoint[i, 1] = keypoint[i, 1] / 31.0 * 128
            keypoint[i, 2] = keypoint[i, 2] / 47.0 * 192
        result = np.zeros([18,4])
        for i in range(18):
            result[i,0] = np.sqrt(np.square(keypoint[i, 0] * spacing[0] - location[i,0]) +np.square(keypoint[i, 1] * spacing[1] - location[i,1]) + np.square(keypoint[i, 2] * spacing[2] - location[i,2]))
            result[i, 1] = (np.abs(keypoint[i, 0] * spacing[0] - location[i,0]) + np.abs(keypoint[i, 1] * spacing[1] - location[i,1]) + np.abs(keypoint[i, 2] * spacing[2] - location[i,2])) / 3.0
            result[i, 2] = (np.square(keypoint[i, 0] * spacing[0] - location[i,0]) +np.square(keypoint[i, 1] * spacing[1] - location[i,1])) + np.square(keypoint[i, 2] * spacing[2] - location[i,2])/3.0
            #result[i,3] = np.sqrt(np.square((prepoint[0][0]+x1) * spacing[0] - location[i,0]) + np.square(((prepoint[1][0]+y1) * spacing[1] - location[i,1])) + np.square(((prepoint[2][0]+z1) * spacing[2] - location[i,2])))
        #blurkeypoint = self.refine_keypoints_dark3d(keypoint,pred_label,spacing)
        #self.savepoint(keypoint, name, spacing)
        #reffine = self.refine_keypoints(keypoint,pred_label)
        #for i in range(18):
            #result[i,3] = np.sqrt(np.square(blurkeypoint[i,0] * spacing[0] - location[i,0]) +np.square(blurkeypoint[i,1] * spacing[1] - location[i,1]) + np.square(blurkeypoint[i,2] * spacing[2] - location[i,2]))
            #result[i,3] = np.sqrt(np.square(reffine[i,0]* spacing[0] - location[i,0]) + np.square(reffine[i,1] * spacing[1] - location[i,1]) + np.square(reffine[i,2] * spacing[2] - location[i,2]))
        result = np.abs(result)
        #self.savepoint(blurkeypoint, name, spacing)
        #self.savepoint(result, name, spacing)
        return result[0,0],result[0,1],result[0,2],result[0,3],\
            result[1,0],result[1,1],result[1,2],result[1,3],\
            result[2,0],result[2,1],result[2,2],result[2,3],\
            result[3,0],result[3,1],result[3,2], result[3,3],\
            result[4, 0], result[4, 1], result[4, 2], result[4,3],\
            result[5, 0], result[5, 1], result[5, 2], result[5,3],\
            result[6, 0], result[6, 1], result[6, 2], result[6,3],\
            result[7, 0], result[7, 1], result[7, 2], result[7,3],\
            result[8, 0], result[8, 1], result[8, 2], result[8,3],\
            result[9, 0], result[9, 1], result[9, 2], result[9,3],\
            result[10, 0], result[10, 1], result[10, 2], result[10,3],\
            result[11, 0], result[11, 1], result[11, 2], result[11,3],\
            result[12, 0], result[12, 1], result[12, 2], result[12,3],\
            result[13, 0], result[13, 1], result[13, 2], result[13,3],\
            result[14, 0], result[14, 1], result[14, 2], result[14,3],\
            result[15, 0], result[15, 1], result[15, 2], result[15,3],\
            result[16, 0], result[16, 1], result[16, 2], result[16,3],\
            result[17, 0], result[17, 1], result[17, 2], result[17,3]

