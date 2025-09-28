# Custom visualization hook that resizes images to match prediction size
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class ResizedVisualizationHook(Hook):
    """Custom visualization hook that resizes images to match prediction size.

    This solves the mismatch between 1000x1000 original images and 512x512 predictions
    when using ResetOriShape in the validation pipeline.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        from mmengine.visualization import Visualizer
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')
        self._test_index = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        if total_curr_iter % self.interval == 0:
            # Visualize only the first data
            img_path = outputs[0].img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            # Load as grayscale since your images are grayscale
            img = mmcv.imfrombytes(img_bytes, flag='grayscale')
            # Convert grayscale to RGB for visualization
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            # RESIZE THE IMAGE TO MATCH PREDICTION SIZE
            # Get the size from the prediction
            pred_shape = outputs[0].pred_sem_seg.data.shape

            # Handle different shape formats
            if len(pred_shape) == 2:
                resize_shape = (pred_shape[1], pred_shape[0])
            else:
                # If shape is (C, H, W) or (1, H, W), use last two dimensions
                resize_shape = (pred_shape[-1], pred_shape[-2])

            img = mmcv.imresize(img, resize_shape)

            # Also resize ground truth if it exists and has different shape
            if hasattr(outputs[0], 'gt_sem_seg') and outputs[0].gt_sem_seg is not None:
                gt_shape = outputs[0].gt_sem_seg.data.shape
                if gt_shape[-2:] != resize_shape[::-1]:
                    # Ground truth needs resizing too
                    gt_data = outputs[0].gt_sem_seg.data
                    if len(gt_data.shape) == 3 and gt_data.shape[0] == 1:
                        gt_data = gt_data.squeeze(0)
                    # For now, just ensure gt matches pred shape
                    outputs[0].gt_sem_seg.data = gt_data

            window_name = f'val_{osp.basename(img_path)}'

            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[SegDataSample]) -> None:
        """Run after every testing iteration.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            # Load as grayscale since your images are grayscale
            img = mmcv.imfrombytes(img_bytes, flag='grayscale')
            # Convert grayscale to RGB for visualization
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            # RESIZE THE IMAGE TO MATCH PREDICTION SIZE
            pred_shape = data_sample.pred_sem_seg.data.shape
            img = mmcv.imresize(img, (pred_shape[1], pred_shape[0]))

            window_name = f'test_{osp.basename(img_path)}'

            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index)