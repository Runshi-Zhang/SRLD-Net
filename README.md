# Super-resolution Landmark Detection Networks for Medical Images
Here is the official implementation of the paper:

[Zhang R, Mo H, Hu W, et al. Super-resolution landmark detection networks for medical images[J]. Computers in Biology and Medicine, 2024, 182: 109095.](https://www.sciencedirect.com/science/article/pii/S0010482524011806))

The neck and head of our proposed SRLD-Net is 'SRLD-Net/mmseg/models/decode_heads/ourfuseuper_head.py'.
And the SR-UNet is 'SRLD-Net/mmseg/models/decode_heads/srpose_head.py'.

## Requirments
We trained our models depending on:
Pytorch 1.13.1
Python 3.8
mmcv>=2.0.0rc1,<2.1.0
mmengine>=0.4.0,<1.0.0

## Train and infer
The configs is located in /configs/3dnii/.
The training and infering methods are according to [openmmlab](https://mmsegmentation.readthedocs.io/en/latest/).

## Reference and Acknowledgments
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

[SRPose](https://github.com/haonanwang0522/SRPose)
