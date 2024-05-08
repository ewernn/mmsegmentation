## package installations
pip install -q -r mmsegmentation/requirements.txt
export PYTHONPATH="$PYTHONPATH:/Users/ewern/Desktop/code/img_segmentation/mmsegmentation"

## pytorch installation
> for GPU (MPS)
conda install pytorch torchvision torchaudio -c pytorch-nightly
> For GPU (CUDA)
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
> For CPU-only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

> visualize experiment outputs in tensorboard if TensorboardVisBackend is added in vis_backends
tensorboard --logdir work_dirs/test_visual/20220810_115248/vis_data

> run training
python tools/train.py configs/unet/eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_solar-512x512.py --work-dir ../work_dir