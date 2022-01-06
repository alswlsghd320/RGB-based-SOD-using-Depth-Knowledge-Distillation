import torch
from torch import nn
import torch.nn.functional as F

from models.PoolNet.deeplab_resnet import resnet50_locate
from models.PoolNet.vgg import vgg16_locate
from models.KD_models.aggregation import aggregation_add
from utils.utils import _upsample_like

config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False], [True, True, True, False]], 'score': 128}  # no convert layer, no conv6

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 128}

class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=False)))
        self.convert0 = nn.ModuleList(up)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl

class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out, need_x2, need_fuse):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [2,4,8]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)

    def forward(self, x, x2=None, x3=None):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(torch.add(resl, x2), x3))
        return resl

class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k ,1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    convert_layers, deep_pool_layers, score_layers = [], [], []
    convert_layers = ConvertLayer(config['convert'])

    for i in range(len(config['deep_pool'][0])):
        deep_pool_layers += [DeepPoolLayer(config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])]

    score_layers = ScoreLayer(config['score'])

    return vgg, convert_layers, deep_pool_layers, score_layers


class PoolNet(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, deep_pool_layers, score_layers, kd=False):
        super(PoolNet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base
        self.deep_pool = nn.ModuleList(deep_pool_layers)
        self.score = score_layers
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers
        self.kd = kd

        if self.kd:
            self.rfb1 = nn.Conv2d(256, 32, 1, padding=0)
            self.rfb2 = nn.Conv2d(128, 32, 1, padding=0)
            self.rfb3 = nn.Conv2d(128, 32, 1, padding=0)
            self.att_1 = nn.Conv2d(32, 1, 3, padding=1)
            self.att_2 = nn.Conv2d(32, 1, 3, padding=1)
            self.att_3 = nn.Conv2d(32, 1, 3, padding=1)
            self.agg1 = aggregation_add(32)

    def forward(self, x):
        x_size = x.size()
        conv2merge, infos = self.base(x)
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)
        conv2merge = conv2merge[::-1]

        att_module = []
        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0]) #merge1 512x32x32

        length = len(conv2merge)
        for k in range(1, length - 1):
            merge = self.deep_pool[k](merge, conv2merge[k + 1], infos[k])
            if self.kd:
                att_module.append(merge)
        merge = self.deep_pool[-1](merge) # batch x 128 x 128 x 128

        if self.kd:
            att_module.append(merge)
            x1 = self.rfb1(att_module[-3])
            att_1 = self.att_1(x1)
            x1 = x1 * F.sigmoid(att_1)

            x2 = self.rfb2(att_module[-2])
            att_2 = self.att_2(x2)
            x2 = x2 * F.sigmoid(att_2)

            x3 = self.rfb3(att_module[-1])
            att_3 = self.att_3(x3)
            x3 = x3 * F.sigmoid(att_3)

            agg = self.agg1(x3, x2, x1)

            return _upsample_like(agg, x), _upsample_like(att_1, x), _upsample_like(att_2, x), _upsample_like(att_3, x)

        else:
            merge = self.score(merge, x_size)
            return merge

def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, vgg16_locate()))
    elif base_model_cfg == 'resnet':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
