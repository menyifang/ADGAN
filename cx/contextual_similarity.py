# -*- coding: utf-8 -*-
"""
@Project:   metrics_bundle
@File   :   contextual_similarity
@Author :   TonyMao@AILab
@Date   :   2019-10-29
@Desc   :   None
"""
import os
from pprint import pprint

# from __future__ import absolute_import
import torch
import torch.utils.data
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


class CXLoss(nn.Module):
    def __init__(self, sigma=0.1, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCxHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W / W_sum

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''
        # NCHW
        # print(featureI.shape)

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            # See the torch document for functional.conv2d
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        # print(CX.shape)
        CX = -torch.log(CX)
        CX = torch.mean(CX)
        return CX


# vgg definition that conveniently let's you grab the outputs from any layer
class VGGModule(nn.Module):
    def __init__(self, pool='max'):
        super(VGGModule, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


def contextual_similarity(inputs, batch_size=32, style_layers=('r32', 'r42')):
    # vgg pretrained.
    vgg = VGGModule('max').cuda()
    vgg.train(False)
    vgg.load_state_dict(torch.load('/mnt/cephfs_new_wj/lab_ad_idea/maoyiming/data/vgg_conv.pth'))
    loss_layer = CXLoss().cuda()
    cx = 0.0
    dataloader_ = torch.utils.data.DataLoader(inputs, batch_size=batch_size)
    with torch.no_grad():
        for ref, fake in tqdm(dataloader_, ascii=True, ncols=80):
            ref.requires_grad = False
            fake.requires_grad = False
            vgg_style = vgg(ref.cuda(), style_layers)
            vgg_fake = vgg(fake.cuda(), style_layers)
            for j, (ref_single, fake_single) in enumerate(zip(vgg_style, vgg_fake)):
                cx += float(loss_layer(ref_single, fake_single))
                del ref_single, fake_single
            del vgg_style, vgg_fake
    del dataloader_, vgg
    return cx


class DualDataset(Dataset):
    def __init__(self, img_paths, indexs, img_height, img_width):
        self.imgs = img_paths
        self.h = img_height
        self.w = img_width
        self.indexs = indexs
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item):
        image = Image.open(os.path.join(root, self.imgs[item]))  # PIL image, label
        image = self.transform(image)
        out = []
        for idx in self.indexs:
            out.append(image[:, :, idx * self.w:(idx + 1) * self.w])
        return out

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    print("Calculating CX...")
    metric_name = "CX"
    config_file = '/mnt/cephfs_new_wj/lab_ad_idea/maoyiming/code/Pose-Transfer-exp/metrics_bundle/configs/compare.yaml'
    config = yaml.load(open(config_file, 'r'))
    pprint(config)
    root = "/mnt/cephfs_new_wj/lab_ad_idea/maoyiming/data/comparison_6methods"
    imgs = os.listdir(root)
    img_width = config['data_format']['img_width']
    img_height = config['data_format']['img_height']
    data_keys = config['data_format']['keys']
    data_begin = config['data_format']['data_begin_index']
    metric_config = yaml.load(open("/mnt/cephfs_new_wj/lab_ad_idea/maoyiming/code/Pose-Transfer-exp/metrics_bundle/configs/metrics.yaml"))['metrics']

    print(metric_config[metric_name]['keys'])
    print(data_keys)

    if '.DS_Store' in imgs:
        imgs.remove('.DS_Store')
    print("CX. ref/gen: ----")
    ref = data_keys.index(metric_config[metric_name]['keys'][0])
    for i in range(data_begin, len(data_keys)):
        dataset = DualDataset(imgs, indexs=[ref, i], img_height=img_height, img_width=img_width)
        print(data_keys[i], contextual_similarity(dataset, batch_size=48))

    # print(contextual_similarity([torch.ones(3, 256, 176)] * 10, [torch.rand(3, 256, 176)] * 10, batch_size=2))
    # print(contextual_similarity([torch.zeros(3, 256, 176)] * 10, [torch.zeros(3, 256, 176)] * 10, batch_size=2))
    print("CX1. gen/ref: ----")
    for i in range(data_begin, len(data_keys)):
        dataset = DualDataset(imgs, indexs=[i, ref], img_height=img_height, img_width=img_width)
        print(data_keys[i], contextual_similarity(dataset, batch_size=48))
