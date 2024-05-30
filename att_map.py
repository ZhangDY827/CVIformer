from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import os
#from STSRNet_V11 import *
#from model1_3 import *
from model_my import *
import matplotlib.pyplot as plt

## generate the attention maps

#python att_map.py --model 'checkpoint_model_my_x4/model_my_4xSR_epoch200.pth.tar'

#os.environ['CUDA_VISIBLE_DEVICES']='2'
def parse_args():
    parser = argparse.ArgumentParser()
#    parser.add_argument('--testset_dir', type=str, default='./data/test/')
    parser.add_argument('--testset_dir', type=str, default='/home/hekun/Programs/Stereo_test_dataset/')
    parser.add_argument("--model", type=str, default="../../ckpt/SRResNet/SRResNet.pth", help="model path")
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:2')
    #parser.add_argument('--model_name', type=str, default='iPASSR_2xSR')
    parser.add_argument('--model_name', type=str, default='iPASSR_4xSR')
    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    #net = Net(cfg.scale_factor).to(cfg.device)
    net = Net(cfg.scale_factor)
    #net = StrNet(nfeats=64, factor=4)
    #model = torch.load(cfg.model,map_location={'cuda:2': 'cuda:0','cuda:1': 'cuda:0'})
    model = torch.load(cfg.model, map_location=torch.device('cpu'))
    net.load_state_dict(model['state_dict'])
    #net.load_state_dict(model['model']) #for strnet 
    #L_path = '/home/zhangdy/test/Middlebury/lr_x4/sword2/lr0.png'
    L_path = '/home/zhangdy/test/Middlebury/lr_x4/pipes/lr0.png'
    R_path = '/home/zhangdy/test/Middlebury/lr_x4/pipes/lr1.png'
    #R_path = '/home/zhangdy/test/Middlebury/lr_x4/sword2/lr1.png'
    LR_left = Image.open(L_path)
    LR_right = Image.open(R_path)
    LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
    LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
    #LR_left, LR_right = Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
    LR_left, LR_right = Variable(LR_left), Variable(LR_right)
    with torch.no_grad():
        out_left, out_right, (M_right_to_left, M_left_to_right), (V_left, V_right) = net(LR_left, LR_right, is_training = 1)
        print(LR_left.shape)
        print(M_right_to_left.shape)
        #att_left, att_right = torch.clamp(M_right_to_left, 0, 1), torch.clamp(M_left_to_right, 0, 1)
        att_left, att_right = M_right_to_left[:,30,:,:], M_left_to_right[:,30,:,:]
        #SR_left_img = transforms.ToPILImage()(torch.squeeze(att_left.data.cpu(), 0))
        plt.axis('off')
        plt.imshow(torch.squeeze(att_left.data.cpu(), 0).numpy())
        #SR_left_img.save('_Latt.png')
        plt.savefig('_Latt_4.png')
        plt.axis('off')
        #plt.imshow(torch.squeeze(att_right.data.cpu(), 0).numpy(), cmap='jet')
        plt.imshow(torch.squeeze(att_right.data.cpu(), 0).numpy())
        plt.savefig('_Ratt_4.png')
        #SR_right_img = transforms.ToPILImage()(torch.squeeze(att_right.data.cpu(), 0))
        #SR_right_img.save('_Ratt.png')
