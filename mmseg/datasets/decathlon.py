# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List

from mmengine.fileio import load

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DecathlonDataset(BaseSegDataset):
    """Dataset for Dacathlon dataset.

    The dataset.json format is shown as follows

    .. code-block:: none

        {
            "name": "BRATS",
            "tensorImageSize": "4D",
            "modality":
            {
                "0": "FLAIR",
                "1": "T1w",
                "2": "t1gd",
                "3": "T2w"
            },
            "labels": {
                "0": "background",
                "1": "edema",
                "2": "non-enhancing tumor",
                "3": "enhancing tumour"
            },
            "numTraining": 484,
            "numTest": 266,
            "training":
            [
                {
                    "image": "./imagesTr/BRATS_306.nii.gz"
                    "label": "./labelsTr/BRATS_306.nii.gz"
                    ...
                }
            ]
            "test":
            [
                "./imagesTs/BRATS_557.nii.gz"
                ...
            ]
        }
    """
    METAINFO = dict(
        #classes=('ZY', 'AR'),
        #palette=[[255, 0, 0], [0, 0, 142]]
        
        #classes=('MA', 'ME', 'ZL', 'ZR', 'AL', 'AR', 'FL'),
        #palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 #[4, 200, 3], [120, 120, 80], [140, 140, 140]]

        #classes=('MA', 'ME', 'ZY', 'AR', 'FL'),
        #palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                  #[140, 140, 140]]
    #classes = ('BD','MA', 'ME', 'ZY', 'AR', 'FL', 'BB'),
    #palette = [[0,0,0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               #[140, 140, 140],[230,230,50]]

    classes = ('BD', 'MA', 'ME', 'ZY', 'AR', 'FL'),
    palette = [[0, 0, 0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [140, 140, 140]]
        )
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        raw_data_list = annotations[
            'training'] if not self.test_mode else annotations['test']
        data_list = []

        for raw_data_info in raw_data_list:
            # `2:` works for removing './' in file path, which will break
            # loading from cloud storage.
            if isinstance(raw_data_info, dict):
                data_info = dict(
                    img_path=osp.join(self.data_root, raw_data_info['image']
                                      [2:]))
                data_info['seg_map_path'] = osp.join(
                    self.data_root, raw_data_info['label'][2:])
            else:
                data_info = dict(
                    img_path=osp.join(self.data_root, raw_data_info)[2:])
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)


        annotations.pop('training')
        annotations.pop('test')

        metainfo = copy.deepcopy(annotations)
        metainfo['classes'] = [*metainfo['labels'].values()]
        print(metainfo['classes'])
        #self.METAINFO = metainfo
        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        #print(metainfo)
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        return data_list
