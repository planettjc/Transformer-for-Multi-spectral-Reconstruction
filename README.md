# Network Code for Multi-spectral Reconstruction in "Handheld Snapshot Multi-spectral Camera at Tens-of-Megapixel Resolution"

#### Author
Weihang Zhang

Modified from the [code](https://github.com/caiyuanhao1998/MST) in [Coarse-to-Fine Sparse Transformer for Hyperspectral Image Reconstruction (ECCV 2022)](https://link.springer.com/chapter/10.1007/978-3-031-19790-1_41)


## 1. Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

```shell
pip install -r requirements.txt
```

## 2. Prepare Dataset:
Randomly crop the encoded measurement and the calibrated masks and ground truths of each spectral channel into small patches (e.g. 256 in width and in height) and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell
|--THETA
    |--datasets
        |--real_data
            |--gt
                |--scene0000.mat [12*256*256]
                |--scene0001.mat
                ：  
            |--gt_test
                |--scene0000.mat [12*256*256]
                |--scene0001.mat
                ：  
            |--mask
                |--scene0000.mat [12*256*256]
                |--scene0001.mat
                ：  
            |--mask_test
                |--scene0000.mat [12*256*256]
                |--scene0001.mat
                ：  
            |--meas
                |--scene0000.mat [256*256]
                |--scene0001.mat
                ：  
            |--meas_test
                |--scene0000.mat [256*256]
                |--scene0001.mat
                ：  
    |--CST.py
    |--ssim_torch.py
    |--test_real.py
    |--train_real.py
    |--utils.py
```

## 3. Real Experiment:

### 3.1　Training

```shell
cd THETA

# CST_S
python train_real.py --outf ./exp/cst_s_real/ --method cst_s 

# CST_M
python train_real.py --outf ./exp/cst_m_real/ --method cst_m  

# CST_L
python train_real.py --outf ./exp/cst_l_real/ --method cst_l
```

The training log, trained model, and reconstrcuted multi-spectral data will be available in `THETA/exp/` as determined by the `outf` parameter. 


### 3.2　Testing	

The parameter `pretrained_model_path` should be set to be the path of the trained model, e.g., `./exp/cst_s_real/2022_08_16_21_39_19/model/model_epoch_459.pth`.

```shell
cd THETA

# CST_S
python test_real.py --outf ./exp/cst_s_real/ --method cst_s --pretrained_model_path ./exp/cst_s_real/pre_trained_model.pth

# CST_M
python test_real.py --outf ./exp/cst_m_real/ --method cst_m --pretrained_model_path ./exp/cst_m_real/pre_trained_model.pth

# CST_L
python test_real.py --outf ./exp/cst_l_real/ --method cst_l --pretrained_model_path ./exp/cst_l_real/pre_trained_model.pth

```

- The reconstrcuted multi-spectral data will be output into the `test` subfolder in the folder determined by the `outf` parameter.  


