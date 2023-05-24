from utils import *
from CST import CST
import torch
import scipy.io as scio
import time
import os
import numpy as np
from torch.autograd import Variable
import datetime
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description="Multi-spectral Reconstruction by CST")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='./datasets/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/cst_s_real/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='cst_s', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default='./exp/cst_s_real/pre_trained_model.pth', help='pretrained model directory')

# Training specifications
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')

opt = parser.parse_args()

# dataset
opt.test_path = f"{opt.data_root}/real_data/meas_test/"
opt.gt_test_path = f"{opt.data_root}/real_data/gt_test/"
opt.mask_test_path = f"{opt.data_root}/real_data/mask_test/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
    

# init mask
input_mask_test = torch.from_numpy(sio.loadmat(opt.mask_test_path+'scene0000.mat')['mask']).cuda().float()
[nC, H, W] = input_mask_test.shape

# dataset
test_data = LoadData(opt.test_path)

# saving path
result_path = opt.outf + 'test/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# model    
if opt.method == 'cst_s':
    model = CST(num_blocks=[1, 1, 2], dim=nC, sparse=True).cuda()
elif opt.method == 'cst_m':
    model = CST(num_blocks=[2, 2, 2], dim=nC, sparse=True).cuda()
elif opt.method == 'cst_l':
    model = CST(num_blocks=[2, 4, 6], dim=nC, sparse=True).cuda()  
if opt.pretrained_model_path is not None:
    print(f'load model from {opt.pretrained_model_path}')
    checkpoint = torch.load(opt.pretrained_model_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                            strict=True)
model = model.cuda()

def main():
    for k in range(len(test_data)):
        test_gt = torch.from_numpy(sio.loadmat(opt.gt_test_path + test_data[k])['gt']).cuda().float()        
        input_mask_test = sio.loadmat(opt.mask_test_path + test_data[k])['mask']
        meas = sio.loadmat(opt.test_path + test_data[k])['input_meas']
        input_meas = torch.from_numpy(np.tile(meas[np.newaxis, np.newaxis, :, :], (1, nC, 1, 1))).cuda().float()
        mask3d_batch = torch.from_numpy(np.tile(input_mask_test[np.newaxis, :, :, :], (1, 1, 1, 1))).cuda().float()
        model.eval()
        with torch.no_grad():
            model_out, _ = model(input_meas, mask3d_batch)

        psnr_val = torch_psnr(model_out[0, :, :, :], test_gt)
        ssim_val = torch_ssim(model_out[0, :, :, :], test_gt)
        pred = np.transpose(model_out[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)).astype(np.float32)
        truth = np.transpose(test_gt.cpu().numpy(), (1, 2, 0)).astype(np.float32)
        name = result_path + 'pred{:0>4d}.mat'.format(k)
        print(f'Save reconstructed HSIs as {name}.')
        scio.savemat(name, {'truth': truth, 'pred': pred})
        model.train()    

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


