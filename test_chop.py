
import argparse
import torch
import numpy as np
import time, math
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils_test import TestSetLoader, rgb2y
import matplotlib.pyplot as plt
from numpy import clip
from torchvision.transforms import ToPILImage
import scipy.io as scio
import os
from functools import partial
import pickle
#from skimage import measure
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
#from model_hekun import *
from model_my import *
#from model1_3 import *
#MyNet的test，用utils_hekun

parser = argparse.ArgumentParser(description="Pytorch SRResNet Eval")
parser.add_argument("--model", type=str, default="../../ckpt/SRResNet/SRResNet.pth", help="model path")

#parser.add_argument("--model_sam", type=str, default="../../ckpt/SRResNet/SRResNet_SAM.pth", help="model path")
parser.add_argument("--model_sam", type=str, default="/home/hekun/Programs/MyNet/checkpoint_V14_hk/modelcover_epoch_48.pth", help="model path")
#parser.add_argument("--model_sam", type=str, default="/home/hekun/Programs/MyNet/checkpoint_V12_hk/modelcover_epoch_44.pth", help="model path")
#parser.add_argument("--model_sam", type=str, default="/home/hekun/Programs/MyNet/checkpoint_My_V11_x2/model_epoch_39.pth", help="model path")

parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--scale", type=int, default=2, help="upscale factor")#对应x2，x4
#parser.add_argument('--testset_dir', type=str, default='/data/data-home/hekun/SSR/test')
parser.add_argument('--testset_dir', type=str, default='/home/zhangdy/test')
parser.add_argument('--dataset', type=str, default='KITTI2015')
parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint_model1_x2_3_1/')
#parser.add_argument('--checkpoint_path', type=str, default='./log_x2/')
parser.add_argument('--model_name', type=str, default='iPASSR_2xSR_epoch140')



opt = parser.parse_args()
if opt.cuda:
    #print("=> use gpu id: '{}'".format(opt.gpus))
    #os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    torch.cuda.set_device(2)
def valid_sam(testing_data_loader,  model):
    psnr_epoch = 0
    ssim_epoch = 0
    count = 0
    for iteration, (HR_left, _, LR_left, LR_right) in enumerate(testing_data_loader):
        LR_left, LR_right, HR_left = LR_left / 255, LR_right / 255, HR_left / 255
        input_l, input_r, target = Variable(LR_left), Variable(LR_right), Variable(HR_left)
        print(input_l.shape, target.shape)
        _, _, H, W = input_l.shape
        #if H>300 or W>300:
        #    print('skip--------------------------------------------')
        #    continue
        if opt.cuda:
            input_l = input_l.cuda()
            input_r = input_r.cuda()
            target = target.cuda()
        SR_left_np,out_right = model(input_l, input_r,is_training=0)
        #SR_left_np,out_right = forward_chop(4, model, input_l, input_r)
        HR = SR_left_np
        SR_left_np = np.array(torch.squeeze(HR[:, :, :, :].data.cpu(), 0).permute(1, 2, 0))
        HR_left_np = np.array(torch.squeeze(target[:, :, :, :].data.cpu(), 0).permute(1, 2, 0))
        #print(SR_left_np.shape)
        PSNR = psnr(HR_left_np, SR_left_np)
        SSIM = ssim(HR_left_np, SR_left_np, multichannel=True)
        psnr_epoch = psnr_epoch + PSNR
        ssim_epoch =ssim_epoch + SSIM
        count = count + 1
    print("===> SRResNet_SAM Avg. PSNR: {:.8f} dB SSIM: {:.8f} dB".format(psnr_epoch/(count), ssim_epoch/(count)))

def forward_chop(scale, model, x_l, x_r, shave=10, min_size=1600000):
    n_GPUs = 2
    b, c, h, w = x_l.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list_l = [
        x_l[:, :, 0:h_size, 0:w_size],
        x_l[:, :, 0:h_size, (w - w_size):w],
        x_l[:, :, (h - h_size):h, 0:w_size],
        x_l[:, :, (h - h_size):h, (w - w_size):w]]

    lr_list_r = [
        x_r[:, :, 0:h_size, 0:w_size],
        x_r[:, :, 0:h_size, (w - w_size):w],
        x_r[:, :, (h - h_size):h, 0:w_size],
        x_r[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        # print('here')
        sr_list_l = []
        sr_list_r = []
        for i in range(0, 4, n_GPUs):
            lr_batch_l = torch.cat(lr_list_l[i:(i + n_GPUs)], dim=0)
            lr_batch_r = torch.cat(lr_list_r[i:(i + n_GPUs)], dim=0)
            sr_batch_l, sr_batch_r  = model(lr_batch_l, lr_batch_r,is_training=0)
            sr_list_l.extend(sr_batch_l.chunk(n_GPUs, dim=0))
            sr_list_r.extend(sr_batch_r.chunk(n_GPUs, dim=0))

    else:
        sr_list_l=[]
        sr_list_r=[]
        for (patch_l, patch_r) in zip(lr_list_l, lr_list_r):
            x_l, x_r = forward_chop(scale=scale, model=model, x_l=patch_l, x_r=patch_r, shave=shave, min_size=min_size)
            sr_list_l.append(x_l)
            sr_list_r.append(x_r)

        

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output_l = x_l.new(b, c, h, w)
    output_l[:, :, 0:h_half, 0:w_half] \
        = sr_list_l[0][:, :, 0:h_half, 0:w_half]
    output_l[:, :, 0:h_half, w_half:w] \
        = sr_list_l[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output_l[:, :, h_half:h, 0:w_half] \
        = sr_list_l[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output_l[:, :, h_half:h, w_half:w] \
        = sr_list_l[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        
    output_r = x_r.new(b, c, h, w)
    output_r[:, :, 0:h_half, 0:w_half] \
        = sr_list_r[0][:, :, 0:h_half, 0:w_half]
    output_r[:, :, 0:h_half, w_half:w] \
        = sr_list_r[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output_r[:, :, h_half:h, 0:w_half] \
        = sr_list_r[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output_r[:, :, h_half:h, w_half:w] \
        = sr_list_r[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output_l, output_r

def main():

    net = Net(opt.scale)
    #model = torch.load(opt.checkpoint_path + opt.model_name + '.pth.tar')
    #model = torch.load('/home/zhangdy/Stereo_hk/checkpoint_/iPASSR_2xSR_epoch140.pth.tar')
    model = torch.load(opt.model)
    #model = torch.load('./log/' + opt.model_name + '.pth.tar')
    net.load_state_dict(model['state_dict'])


    if opt.cuda:
        net.cuda()
    test_set1 = TestSetLoader(dataset_dir=opt.testset_dir + '/' + "Middlebury", scale_factor=opt.scale)
    test_loader1 = DataLoader(dataset=test_set1, num_workers=1, batch_size=1, shuffle=False)
    test_set2 = TestSetLoader(dataset_dir=opt.testset_dir + '/' + "KITTI2012", scale_factor=opt.scale)
    test_loader2 = DataLoader(dataset=test_set2, num_workers=1, batch_size=1, shuffle=False)
    test_set3 = TestSetLoader(dataset_dir=opt.testset_dir + '/' + "KITTI2015", scale_factor=opt.scale)
    test_loader3 = DataLoader(dataset=test_set3, num_workers=1, batch_size=1, shuffle=False)
    test_set4 = TestSetLoader(dataset_dir=opt.testset_dir + '/' + "Flickr1024", scale_factor=opt.scale)
    test_loader4 = DataLoader(dataset=test_set4, num_workers=1, batch_size=1, shuffle=False)
    import datetime
    oldtime = datetime.datetime.now()
    
    print("Middlebury")
    valid_sam(test_loader1,net)
    wotime = datetime.datetime.now()
    print('Time consuming: ', wotime - oldtime)

    print("KITTI2012")
    valid_sam(test_loader2, net)
    wotime1 = datetime.datetime.now()
    print('Time consuming: ', wotime1 - wotime)

    print("KITTI2015")
    valid_sam(test_loader3, net)
    wotime2 = datetime.datetime.now()
    print('Time consuming: ', wotime2 - wotime1)
    
    #print("Flickr1024")
    #valid_sam(test_loader4, net)
    #wotime3 = datetime.datetime.now()
    #print('Time consuming: ', wotime3 - wotime2)

#未用到 和psnr()有何不同？
def cal_psnr(img1, img2):
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1_np = img1.detach().numpy()
    img2_np = img2.detach().numpy()
    return psnr(img1_np, img2_np)

if __name__ == '__main__':
    main()
