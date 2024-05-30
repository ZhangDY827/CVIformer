# CVIformer
### PyTorch implementation of "CVIformer: Cross-View Interactive Transformer for Efficient Stereoscopic Image Super-Resolution"

## Highlights:
#### 1. We propose a novel stereo cross-view interactive Transformer (CVIformer), which integrates effcient residual Transformer blocks (ERTB) for longrange cross-view information extraction within linear complexity. 
 <p align="center"> <img src="https://github.com/ZhangDY827/CVIformer/blob/main/Figs/network.png" width="80%"></p>

 #### 2. The proposed MCAB and ERTB are indeed benefcial to explore more long-range stereo correspondence, resulting in much clearer images with high perceptual quality. 
 <p align="center"> <img src="https://github.com/ZhangDY827/CVIformer/blob/main/Figs/attention.png" width="80%"></p>

 #### 3. Evaluation on benchmark datasets demonstrates that our CVIformer achieves impressive results in terms of speed and accuracy, while requiring signifcantly fewer parameters compared to state-of-the-art (SOTA) methods.
 <p align="center"> <img src="https://github.com/ZhangDY827/CVIformer/blob/main/Figs/results.png" width="80%"></p>

## Codes and Models:
### Requirement:
* **PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.7, cuda=9.0.**
* **Matlab (For training/test data generation and performance evaluation)**

### Train and Test datasets:
* **Following the method iPASSR, we employ the Flickr1024 as the training datsset, which comprises of 800 high-quality images. We evaluate the model performamce on four datasets, Middlebury, KITTI2012, KITTI2015 and Flickr1024. Please refer to the [iPASSR](https://github.com/YingqianWang/iPASSR)**

### Running:
* **Run `train.py` to perform training. Checkpoint will be saved to  `--checkpoint`.**
* **Run `dats_Saveimag.py` to perform a demo inference. Results (`.png` files) will be saved to `--testset_dir`.**
* **Run `att_map.py` to generate the attention maps.**

