# AirNet-PyTorch
Implementation of the paper ''Attention Inspiring Receptive-fields Network'' (under review), which contains the evaluation code and trained model by:

[Lu Yang](https://github.com/soeaver), Qing Song, Yingqi Wu and Mengjie Hu


## Install
* Install [PyTorch>=0.3.0](http://pytorch.org/)
* Install [torchvision>=0.2.0](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/soeaver/pytorch-priv
  ```

## Results

### ImageNet1k
Single-crop (224x224) validation error rate is reported. 

| Network                 | Flops (G) | Params (M) | Top-1 Error (%) | Top-5 Error (%) |
| :---------------------: | --------- |----------- | --------------- | --------------- |
| AirNet50-1x64d          | 4342.1    | 25.5       | 23.52           | 7.01            |
| AirNet101-1x64d         | 8039.0    | 44.5       | 22.18           | 6.23            |
| AirNeXt50-32x4d         | 4342.1    | 25.5       | 23.52           | 7.01            |
| AirNeXt101-32x4d (r=16) | 8039.0    | 44.5       | 22.18           | 6.23            |
| AirNeXt101-32x4d (r=2)  | 8039.0    | 44.5       | 22.18           | 6.23            |
