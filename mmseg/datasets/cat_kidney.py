# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.voc import PascalVOCDataset
from mmseg.registry import DATASETS


# SolarHoof dataset definition
@DATASETS.register_module()
class CatKidneyDataset(PascalVOCDataset):

    METAINFO = dict(
        classes=('background', 'left_kidney', 'right_kidney'),
        palette=[[120, 120, 120], [6, 230, 230], [56, 59, 120]]
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)