# Selective Frequency Network for Image Restoration (ICLR2023)

Yuning Cui, Yi Tao, [Zhenshan Bing](https://scholar.google.com.hk/citations?user=eIz0XvMAAAAJ&hl=zh-CN&oi=ao), [Wenqi Ren](https://scholar.google.com.hk/citations?user=VwfgfR8AAAAJ&hl=zh-CN&oi=ao), Xinwei Gao, [Xiaochun Cao](https://scholar.google.com.hk/citations?user=PDgp6OkAAAAJ&hl=zh-CN&oi=ao), [Kai Huang](https://scholar.google.com.hk/citations?hl=zh-CN&user=70_wl8kAAAAJ), [Alois Knoll](https://scholar.google.com.hk/citations?user=-CA8QgwAAAAJ&hl=zh-CN&oi=ao)

[![](https://img.shields.io/badge/ICLR-Paper-blue.svg)](https://openreview.net/forum?id=tyZ1ChGZIKO)

## Update 

### Our extended paper has been accepted by T-PAMI ([Image Restoration via Frequency Selection](https://ieeexplore.ieee.org/abstract/document/10310164))

- 2023.08.04  :tada: Release the model and code for image deraining
- 2023.04.19  :tada: Release our results on RSBlur
- 2023.04.14 	:tada: Release pre-trained models, code for three tasks (dehazing, motion deblurring, desnowing), and resulting images on most datasets.

>Image restoration aims to reconstruct the latent sharp image from its corrupted counterpart. Besides dealing with this long-standing task in the spatial domain, a few approaches seek solutions in the frequency domain in consideration of the large discrepancy between spectra of sharp/degraded image pairs. However, these works commonly utilize transformation tools, e.g., wavelet transform, to split features into several frequency parts, which is not flexible enough to select the most informative frequency component to recover. In this paper, we exploit a multi-branch and content-aware module to decompose features into separate frequency subbands dynamically and locally, and then accentuate the useful ones via channel-wise attention weights. In addition, to handle large-scale degradation blurs, we propose an extremely simple decoupling and modulation module to enlarge the receptive field via global and window-based average pooling. Integrating two developed modules into a U-Net backbone, the proposed Selective Frequency Network (SFNet) performs favorably against state-of-the-art algorithms on five image restoration tasks, including single-image defocus deblurring, image dehazing, image motion deblurring, image desnowing, and image deraining.

## Architecture
![](figs/SFNet.png)

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
## Training and Evaluation
Please refer to respective directories.
## Results

|Task|Dataset|PSNR|SSIM|Resulting Images|
|----|------|-----|----|------|
|**Motion Deblurring**|GoPro|33.27|0.963|[gdrive](https://drive.google.com/file/d/1mVerQce1ZZFkKOj0Cbwyj49ZPWdHcO2z/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1ZVHjcpVeZBZU13npEgWCCg?pwd=7kqi)|
||HIDE|31.10|0.941|[gdrive](https://drive.google.com/file/d/1T1ZBg2gfqRCmhjvYD6qrgOdv9ZY0abHk/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1GqGs_oGUbupQ1kSaMmSVGw?pwd=vu68)|
||RSBlur|34.35|0.872|[gdrive](https://drive.google.com/file/d/1-3IO7dOfhkH-rQVudxhRSMI59LfLXdH4/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1HqP4gkRxCPxu8vYmdqWvEg?pwd=bfhh)|
|**Image Dehazing**|SOTS-Indoor|41.24|0.996|[gdrive](https://drive.google.com/file/d/1d-IMbzp3N42dEP1IN-VphAT2Ok2dTH90/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1ewi9VLbGnmDbQDyLeXgpuQ?pwd=occl)|
||SOTS-Outdoor|40.05|0.996|[gdrive](https://drive.google.com/file/d/1m_FTpMYBZBqN76VtEkp2VEKA7WpIL0hO/view?usp=sharing),[Baidu](https://pan.baidu.com/s/10e-pIhxwB-Nt1uWCjmgJzw?pwd=07rl)|
||Dense-Haze|17.46|0.578|[gdrive](https://drive.google.com/file/d/1XfW0PzfxIEhI4GWMTvqOjxIsPE_syB0V/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1AAsA5cKGz6tnIJMpeuU98A?pwd=d3nd)|
|**Image Desnowing**|CSD|38.41|0.99|[gdrive](https://drive.google.com/file/d/1zbqrLwCuvNjfOmER_mvBFbd7WGO9ciYH/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1rM4ybZzXu62Ei7EzKYLWOg?pwd=mwql)|
||SRRS|32.40|0.98|[gdrive](https://drive.google.com/file/d/1XbjtHg5frKTDtoAabCHdPRGV37_WJ_2C/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1-Z6aL4OPB5bYAX4PK7Xkdg?pwd=5vwc)|
||Snow100K|33.79|0.95|[gdrive](https://drive.google.com/file/d/17MQpMn02-l2duiB4t6PEHDTahIeJgsKU/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1cx65WtJFk5Pgf1Os5x_RPQ?pwd=cftv)|
|**Image Deraining**|Average|33.56|0.929|[gdrive](https://drive.google.com/file/d/1QSSXEMs7Mc6U8e0rYpK7ik99Vs1RfojF/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1IYdzzQlFX6ubTBga7EeGtA?pwd=jaa5)|
|**Defocus Deblurring**|DPDD<sub>*single*</sub>|26.23|0.811|[gdrive](https://drive.google.com/file/d/15ep5U--RRRPOwzb0rkd61NHz_8_poXDz/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1jaDr4bY3FzoESsRgHjlJGw?pwd=95gn)|
||DPDD<sub>*dual*</sub>|26.34|0.817|[gdrive](https://drive.google.com/file/d/10EvKjAtbVdwoPCPALQNnEKfy3ele79SK/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1x6ngjYKtktg5jHSqF-4Y3Q?pwd=zc65)|


## Citation
If you find this project useful for your research, please consider citing:
~~~
@inproceedings{cui2023selective,
  title={Selective Frequency Network for Image Restoration},
  author={Cui, Yuning and Tao, Yi and Bing, Zhenshan and Ren, Wenqi and Gao, Xinwei and Cao, Xiaochun and Huang, Kai and Knoll, Alois},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
~~~
## Acknowledgement
This project is mainly based on [MIMO-UNet](https://github.com/chosj95/MIMO-UNet).
## Contact
Should you have any question, please contact Yuning Cui.
