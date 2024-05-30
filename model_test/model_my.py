import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from torchvision import transforms
from einops import rearrange
import numbers

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor#设置上采样因子 train.py中默认x4
        self.init_feature = nn.Conv2d(3, 64, 3, 1, 1, bias=True)# 1.对输入左图 Conv1x1，输出64通道
        self.deep_feature1 = RDG(G0=64, n_RDB=4) #输入64，输出64
        self.pam1 = PAM(64)
        self.fusion1 = nn.Sequential(
            TransformerBlock(128),
            CALayer(128),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True))


        self.RST1=TransformerBlock(64)
        self.scam1=MCAB(64)
        self.RST2=TransformerBlock(64)
        self.scam2=MCAB(64)
        self.RST3 = TransformerBlock(64)
        self.scam3 = MCAB(64)
        self.RST4 = TransformerBlock(64)
        self.scam4 = MCAB(64)

        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=True))

    def forward(self, x_left, x_right, is_training=0): #train.py默认is_training=1
        #设置上采样因子 train.py中默认x4
        x_left_upscale = F.interpolate(x_left, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        # 1.对输入左图 Conv1x1
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        #2.buffer_left输入到4个RDB组，输出buffer_left, catfea_left
        buffer_left, catfea_left = self.deep_feature1(buffer_left)
        buffer_right, catfea_right = self.deep_feature1(buffer_right)
        #print(buffer_left.shape)
        if is_training == 1:
            buffer_leftT, buffer_rightT, (M_right_to_left, M_left_to_right), (V_left, V_right)\
                = self.pam1(buffer_left, buffer_right, catfea_left, catfea_right, is_training)
        if is_training == 0:
            buffer_leftT, buffer_rightT \
                = self.pam1(buffer_left, buffer_right, catfea_left, catfea_right, is_training)

        buffer_leftF = self.fusion1(torch.cat([buffer_left, buffer_leftT], dim=1))
        buffer_rightF = self.fusion1(torch.cat([buffer_right, buffer_rightT], dim=1))

        buffer_left,buffer_right=self.RST1(buffer_leftF),self.RST1(buffer_rightF)
        buffer_left, buffer_right=self.scam1(buffer_left,buffer_right)
        buffer_left, buffer_right = self.RST2(buffer_left), self.RST2(buffer_right)
        buffer_left, buffer_right = self.scam2(buffer_left, buffer_right)
        buffer_left, buffer_right = self.RST3(buffer_left), self.RST3(buffer_right)
        buffer_left, buffer_right = self.scam3(buffer_left, buffer_right)
        buffer_left, buffer_right = self.RST4(buffer_left), self.RST4(buffer_right)
        buffer_left, buffer_right = self.scam4(buffer_left, buffer_right)


        out_left = self.upscale(buffer_left) + x_left_upscale
        out_right = self.upscale(buffer_right) + x_right_upscale

        if is_training == 1:
            return out_left, out_right, (M_right_to_left, M_left_to_right), (V_left, V_right)
        if is_training == 0:
            return out_left, out_right

#RDB调用
class one_conv(nn.Module):
    def __init__(self, G0, G):#GO：输入通道，G：输出通道
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)
        #print(torch.cat((x, output), dim=1).shape)
#RDG调用 输入X,输出Y
class RDB(nn.Module):
    def __init__(self, G0, C, G):  #GO：输入通道，G：输出通道，维度未变，C:层数？
        super(RDB, self).__init__()
        convs = [] #列表添加特征
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x
#Net调用    1
class RDG(nn.Module):
    def __init__(self, G0, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(TransformerBlock(G0))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1) #叠起来_cat,G0*n_RDB维
        out = self.conv(buffer_cat) #Conv1x1卷积降维，G0*n_RDB-->G0维
        return out, buffer_cat


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
# LayerNorm调用
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
# LayerNorm调用
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
# LayerNorm调用
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
# TransformerBlock调用
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
##########################################################################
## residual cross-Dconv Feed-Forward block    TransformerBlock调用
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
##########################################################################
## residual multi-Dconv block (RMDB)  TransformerBlock调用
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
##########################################################################
#Efficient Residual Transformer Block (ERTB)
#####
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type= 'WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

#Net调用    2
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//16, 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channel//16, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

#PAM调用
class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x
#Net调用
def M_Relax(M, num_pixels):
    _, u, v = M.shape
    M_list = []
    M_list.append(M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        pad_M = pad(M[:, :-1-i, :])
        M_list.append(pad_M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        pad_M = pad(M[:, i+1:, :])
        M_list.append(pad_M.unsqueeze(1))
    M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
    return M_relaxed
#Net调用    3
class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.bq = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.bs = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(4 * channels)
        self.bn = nn.BatchNorm2d(4 * channels)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, is_training):
        b, c0, h0, w0 = x_left.shape
        Q = self.bq(self.rb(self.bn(catfea_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr
        M_right_to_left = self.softmax(score)                                                   # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                                  ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c0, 1, 1)

        if is_training == 1:
            return out_left, out_right, \
                   (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)),\
                   (V_left_tanh, V_right_tanh)
        if is_training == 0:
            return out_left, out_right

class MCAB(nn.Module):
    '''
    Stereo Cross Attention Module (MCAB) design by ZDY
    '''

    def __init__(self, dim, bias=False):
        super().__init__()
        self.scale = dim ** -0.5

        self.norm_l = LayerNorm(dim, 'WithBias')
        self.norm_r = LayerNorm(dim, 'WithBias')
        
        self.l_m_proj1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.r_m_proj1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.l_m_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.r_m_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        
        self.l_proj1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        
    def forward(self, x_l, x_r):
        b, c, h, w = x_l.shape
        fousion_l = self.l_m_dwconv(self.l_m_proj1(x_l))
        V_l, l_mid = fousion_l.chunk(2, dim=1)
        
        fousion_r = self.r_m_dwconv(self.r_m_proj1(x_r))
        V_r, r_mid = fousion_r.chunk(2, dim=1)
        
        
        Q_l = self.l_proj1(self.norm_l(l_mid)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(r_mid)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = V_l.permute(0, 2, 3, 1)  # B, H, W, c
        V_r = V_r.permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

import time
if __name__ == "__main__":
    #4RTB+biPAM+4RTB（scan）
    net = Net(upscale_factor=4)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    
    #Left = torch.Tensor(1,64,32,32)
    #Right = torch.Tensor(1,64,32,32)

    #MCAB = MCAB(64)
    #out = MCAB(Left,Right)
    #print(out[0].shape) 
    #Left_img = torch.Tensor(1,3,32,31)
    #Right_img = torch.Tensor(1,3,32,31)    
    #model = Net(4)
    #out = model(Left_img,Right_img,is_training = 0)
    #print(out[0].shape,out[1].shape)  
    #t1 = torch.Tensor(1,3,162,138)
    #t1 = t1.cuda()
    #t2 = torch.Tensor(1,3,185,124)
    #t2 = t2.cuda()
    #t3 = torch.Tensor(1,3,176,120)
    #t3 = t3.cuda()
    #t4 = torch.Tensor(1,3,183,121)
    #t4 = t4.cuda()
    #t5 = torch.Tensor(1,3,178,125)
    #t5 = t5.cuda()
    #net = net.cuda()
    #import time 
    #start_time = time.time()
    #net(t1,t1)
    #net(t2,t2)
    #net(t3,t3)
    #net(t4,t4)
    #net(t5,t5)
    #end_time = time.time()
    #print('inference time: {:.5f}s'.format(end_time - start_time))
    from thop import profile
    t1 = torch.Tensor(1,3,100,100)
    t1 = t1.cuda()
    t2 = torch.Tensor(1,3,100,100)
    t2 = t2.cuda()
    t3 = torch.Tensor(1,3,100,100)
    t3 = t3.cuda()
    t4 = torch.Tensor(1,3,100,100)
    t4 = t4.cuda()
    t5 = torch.Tensor(1,3,100,100)
    t5 = t5.cuda()
    net = net.cuda()
    import time
    start_time = time.time()
    net(t1,t1,0)
    net(t2,t2,0)
    net(t3,t3,0)
    net(t4,t4,0)
    net(t5,t5,0)
    end_time = time.time()
    print('inference time: {:.5f}s'.format((end_time - start_time)/5))

    flops, params = profile(net.cpu(), (torch.ones(1, 3, 100, 100), torch.ones(1, 3, 100, 100),1))
    print('inference time: {:.5f}s'.format((end_time - start_time)/5))    
    print('FLOPs: %.1fGFlops' % (flops / 1e9))  # FLOPs: 185.7GFlops
    print('Number of params: %.2fM' % (total / 1e6))
    #print('inference time: {:.2f}s'.format(end_time - start_time))  # 部署后的推理时间 inference time: 1.95s
