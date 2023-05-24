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
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')

# Training specifications
parser.add_argument('--batch_size', type=int, default=5, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=500, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)

opt = parser.parse_args()

# dataset
opt.data_path = f"{opt.data_root}/real_data/meas/"
opt.mask_path = f"{opt.data_root}/real_data/mask/"
opt.test_path = f"{opt.data_root}/real_data/meas_test/"
opt.gt_path = f"{opt.data_root}/real_data/gt/"
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
input_mask_train = torch.from_numpy(sio.loadmat(opt.mask_path+'scene0000.mat')['mask']).cuda().float()
[nC, H, W] = input_mask_train.shape

# dataset
train_set = LoadData(opt.data_path)
test_data = LoadData(opt.test_path)

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

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

# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
mse = torch.nn.MSELoss().cuda()

def gen_meas(index):
    processed_data = np.zeros((opt.batch_size, nC, H, W), dtype=np.float32)
    for i in range(opt.batch_size):
        meas = sio.loadmat(opt.data_path + train_set[index[i]])['input_meas']
        processed_data[i, :, :, :] = np.tile(meas[np.newaxis, :, :], (nC, 1, 1))
    input_meas = torch.from_numpy(processed_data).cuda().float()
    return input_meas

def gen_mask(index):
    processed_data = np.zeros((opt.batch_size, nC, H, W), dtype=np.float32)
    for i in range(opt.batch_size):
        processed_data[i, :, :, :] = sio.loadmat(opt.mask_path + train_set[index[i]])['mask']
    input_mask = torch.from_numpy(processed_data).cuda().float()
    return input_mask

def train(epoch, logger):
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    for u in range(batch_num):        
        gt_batch = []
        index = np.random.choice(range(len(train_set)), opt.batch_size)
        input_meas = gen_meas(index)
        mask3d_batch = gen_mask(index)
        for i in range(opt.batch_size):
            gt_batch.append(torch.from_numpy(sio.loadmat(opt.gt_path + train_set[index[i]])['gt']).cuda().float())
        gt_batch = torch.stack(gt_batch, dim=0)

        gt = Variable(gt_batch).cuda().float()
        optimizer.zero_grad()

        model_out, diff_pred = model(input_meas, mask3d_batch) # 12 meas and 12 shifted masks
        loss = torch.sqrt(mse(model_out, gt))
        diff_gt = torch.mean(torch.abs(model_out.detach() - gt),dim=1, keepdim=True)  # [b,1,h,w]
        loss_sparsity = F.mse_loss(diff_gt, diff_pred)
        loss = loss + 2 * loss_sparsity

        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    end = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, epoch_loss / batch_num, (end - begin)))
    return 0

def test(epoch, logger):
    psnr_list, ssim_list = [], []
    begin = time.time()
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
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        pred = np.transpose(model_out[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)).astype(np.float32)
        truth = np.transpose(test_gt.cpu().numpy(), (1, 2, 0)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    end = time.time()
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)
        scheduler.step()
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 28:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                savecheckpoint(model, epoch, model_path, logger)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


