# AirNet-PyTorch
Implementation of the paper ''Attention Inspiring Receptive-fields Network'' (under review), which contains the evaluation code and trained model by:

[Lu Yang](https://github.com/soeaver), Qing Song, Yingqi Wu and Mengjie Hu


## Install
* Install [PyTorch>=0.3.0](http://pytorch.org/)
* Install [torchvision>=0.2.0](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/soeaver/AirNet-PyTorch
  ```

## Results

### ImageNet1k
Single-crop (224x224) validation error rate is reported. 

| Network                 | Flops (G) | Params (M) | Top-1 Error (%) | Top-5 Error (%) |
| :---------------------: | --------- |----------- | --------------- | --------------- |
| AirNet50-1x64d          | 4.72      | 27.4       | 21.83           | 5.89            |
| AirNet101-1x64d         | 9.23      | 27.6       | 20.68           | 5.45            |
|                         |           |            |                 |                 |
| AirNeXt50-32x4d         | 5.29      | 25.5       | 20.87           | 5.52            |
| AirNeXt101-32x4d (r=16) | 8.47      | 45.4       | 20.21           | 5.15            |
| AirNeXt101-32x4d (r=2)  | 10.37     | 54.1       | 19.88           | 4.98            |


## Other Resources (from [DPNs](https://github.com/cypw/DPNs))

ImageNet-1k Trainig/Validation List:
- Download link: [GoogleDrive](https://goo.gl/Ne42bM)

ImageNet-1k category name mapping table:
- Download link: [GoogleDrive](https://goo.gl/YTAED5)

