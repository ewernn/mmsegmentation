# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.voc import PascalVOCDataset
from mmseg.registry import DATASETS


# SolarHoof dataset definition
@DATASETS.register_module()
class LatAndSolarHoofDataset(PascalVOCDataset):

    METAINFO = dict(
        classes=('background', 'hoof_solar', 'new_scale', 'hoof_lat', 'block_lat'),
        palette=[[120, 120, 120], [6, 230, 230], [56, 59, 120], [255, 166, 0], [255, 0, 255]]
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
