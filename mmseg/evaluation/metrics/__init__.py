# Copyright (c) OpenMMLab. All rights reserved.
from .citys_metric import CityscapesMetric
from .iou_metric import IoUMetric
from .keypoint_metric import TenMetric
from .eight_metric import EightMetric
from .udp_metric import UDPMetric
__all__ = ['IoUMetric', 'CityscapesMetric','TenMetric','EightMetric','UDPMetric']
