import torch.nn as nn




# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def downsampling(in_planes, out_planes, step, last):

    layer = []
    for i in range(step):
        layer.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False))
        layer.append(nn.BatchNorm2d(out_planes))

        if i == 0:
            in_planes = out_planes
    if last == 1:
        layer.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layer)

def upsampling(in_planes, out_planes, step):

    layer = []

    layer.append(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
    layer.append(nn.BatchNorm2d(out_planes))
    layer.append(nn.Upsample(scale_factor=pow(2,step), mode='nearest'))

    return nn.Sequential(*layer)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class ExchangeResidualUnit(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_layer=None):
        super(ExchangeResidualUnit, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        stride = 1
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        out = self.relu(out)

        return out



class HRNet(nn.Module):

    def __init__(self, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(HRNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        # self.bn2 = norm_layer(self.inplanes)




        self.stage1 = self._make_layer(Bottleneck, 64, layers[0])
        ########### stage 2
        self.stage2_s1_b1 = self._make_layer_v2(ExchangeResidualUnit, 32, layers[1])

        # self.stage2_s1_b1 = self._make_layer(ExchangeResidualUnit, 64, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.stage2_s2_b1 = self._make_layer_v2(ExchangeResidualUnit, 64, layers[1])
        ########### stage 3
        self.stage3_s1_b1 = self._make_layer_v2(ExchangeResidualUnit, 32, layers[1])
        self.stage3_s1_b2 = self._make_layer_v2(ExchangeResidualUnit, 32, layers[1])
        self.stage3_s1_b3 = self._make_layer_v2(ExchangeResidualUnit, 32, layers[1])
        self.stage3_s1_b4 = self._make_layer_v2(ExchangeResidualUnit, 32, layers[1])

        self.stage3_s2_b1 = self._make_layer_v2(ExchangeResidualUnit, 64, layers[1])
        self.stage3_s2_b2 = self._make_layer_v2(ExchangeResidualUnit, 64, layers[1])
        self.stage3_s2_b3 = self._make_layer_v2(ExchangeResidualUnit, 64, layers[1])
        self.stage3_s2_b4 = self._make_layer_v2(ExchangeResidualUnit, 64, layers[1])

        self.stage3_s3_b1 = self._make_layer_v2(ExchangeResidualUnit, 128, layers[1])
        self.stage3_s3_b2 = self._make_layer_v2(ExchangeResidualUnit, 128, layers[1])
        self.stage3_s3_b3 = self._make_layer_v2(ExchangeResidualUnit, 128, layers[1])
        self.stage3_s3_b4 = self._make_layer_v2(ExchangeResidualUnit, 128, layers[1])
        ########### stage 4
        self.stage4_s1_b1 = self._make_layer_v2(ExchangeResidualUnit, 32, layers[1])
        self.stage4_s1_b2 = self._make_layer_v2(ExchangeResidualUnit, 32, layers[1])
        self.stage4_s1_b3 = self._make_layer_v2(ExchangeResidualUnit, 32, layers[1])

        self.stage4_s2_b1 = self._make_layer_v2(ExchangeResidualUnit, 64, layers[1])
        self.stage4_s2_b2 = self._make_layer_v2(ExchangeResidualUnit, 64, layers[1])
        self.stage4_s2_b3 = self._make_layer_v2(ExchangeResidualUnit, 64, layers[1])

        self.stage4_s3_b1 = self._make_layer_v2(ExchangeResidualUnit, 128, layers[1])
        self.stage4_s3_b2 = self._make_layer_v2(ExchangeResidualUnit, 128, layers[1])
        self.stage4_s3_b3 = self._make_layer_v2(ExchangeResidualUnit, 128, layers[1])

        self.stage4_s4_b1 = self._make_layer_v2(ExchangeResidualUnit, 256, layers[1])
        self.stage4_s4_b2 = self._make_layer_v2(ExchangeResidualUnit, 256, layers[1])
        self.stage4_s4_b3 = self._make_layer_v2(ExchangeResidualUnit, 256, layers[1])

        ##### stage 1 exchange
        self.exconv_s1tos1_stage1 = nn.Sequential(nn.Conv2d(256,32,kernel_size=3,stride=1, padding=1)
                                                  , nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.exconv_s1tos2_stage1 = downsampling(256,64,1,1)

        ##### stage 2 exchange
        self.exconv_s1tos2_stage2 = downsampling(32,64,1,0)
        self.exconv_s2tos1_stage2 = upsampling(64,32,1)
        self.exconv_s2tos3_stage2 = downsampling(64,128,1,1)

        ##### stage 3 exchange
        self.exconv_s1tos2_stage3_b1tob2 = downsampling(32, 64, 1,0)
        self.exconv_s1tos3_stage3_b1tob2 = downsampling(32, 128, 2,0)
        self.exconv_s2tos1_stage3_b1tob2 = upsampling(64, 32, 1)
        self.exconv_s2tos3_stage3_b1tob2 = downsampling(64, 128, 1,0)
        self.exconv_s3tos1_stage3_b1tob2 = upsampling(128, 32, 2)
        self.exconv_s3tos2_stage3_b1tob2 = upsampling(128, 64, 1)

        self.exconv_s1tos2_stage3_b2tob3 = downsampling(32, 64, 1,0)
        self.exconv_s1tos3_stage3_b2tob3 = downsampling(32, 128, 2,0)
        self.exconv_s2tos1_stage3_b2tob3 = upsampling(64, 32, 1)
        self.exconv_s2tos3_stage3_b2tob3 = downsampling(64, 128, 1,0)
        self.exconv_s3tos1_stage3_b2tob3 = upsampling(128, 32, 2)
        self.exconv_s3tos2_stage3_b2tob3 = upsampling(128, 64, 1)

        self.exconv_s1tos2_stage3_b3tob4 = downsampling(32, 64, 1,0)
        self.exconv_s1tos3_stage3_b3tob4 = downsampling(32, 128, 2,0)
        self.exconv_s2tos1_stage3_b3tob4 = upsampling(64, 32, 1)
        self.exconv_s2tos3_stage3_b3tob4 = downsampling(64, 128, 1,0)
        self.exconv_s3tos1_stage3_b3tob4 = upsampling(128, 32, 2)
        self.exconv_s3tos2_stage3_b3tob4 = upsampling(128, 64, 1)

        self.exconv_s3tos4_stage3 = downsampling(128,256,1,1)

        ##### stage 4 exchange

        self.exconv_s1tos2_stage4 = downsampling(32, 64, 1,0)
        self.exconv_s1tos3_stage4 = downsampling(32, 128, 2,1)
        self.exconv_s2tos1_stage4 = upsampling(64, 32, 1)
        self.exconv_s2tos3_stage4 = downsampling(64, 128, 1,0)
        self.exconv_s3tos1_stage4 = upsampling(128, 32, 2)
        self.exconv_s3tos2_stage4 = upsampling(128, 64, 1)



        self.exconv_s1tos2_stage4_b1tob2 = downsampling(32, 64, 1,0)
        self.exconv_s1tos3_stage4_b1tob2 = downsampling(32, 128, 2,0)
        self.exconv_s1tos4_stage4_b1tob2 = downsampling(32, 256, 3,0)
        self.exconv_s2tos1_stage4_b1tob2 = upsampling(64, 32, 1)
        self.exconv_s2tos3_stage4_b1tob2 = downsampling(64, 128, 1,0)
        self.exconv_s2tos4_stage4_b1tob2 = downsampling(64, 256, 2,0)
        self.exconv_s3tos1_stage4_b1tob2 = upsampling(128, 32, 2)
        self.exconv_s3tos2_stage4_b1tob2 = upsampling(128, 64, 1)
        self.exconv_s3tos4_stage4_b1tob2 = downsampling(128, 256, 1,0)
        self.exconv_s4tos1_stage4_b1tob2 = upsampling(256, 32, 3)
        self.exconv_s4tos2_stage4_b1tob2 = upsampling(256, 64, 2)
        self.exconv_s4tos3_stage4_b1tob2 = upsampling(256, 128, 1)

        self.exconv_s1tos2_stage4_b2tob3 = downsampling(32, 64, 1, 0)
        self.exconv_s1tos3_stage4_b2tob3 = downsampling(32, 128, 2, 0)
        self.exconv_s1tos4_stage4_b2tob3 = downsampling(32, 256, 3, 0)
        self.exconv_s2tos1_stage4_b2tob3 = upsampling(64, 32, 1)
        self.exconv_s2tos3_stage4_b2tob3 = downsampling(64, 128, 1, 0)
        self.exconv_s2tos4_stage4_b2tob3 = downsampling(64, 256, 2, 0)
        self.exconv_s3tos1_stage4_b2tob3 = upsampling(128, 32, 2)
        self.exconv_s3tos2_stage4_b2tob3 = upsampling(128, 64, 1)
        self.exconv_s3tos4_stage4_b2tob3 = downsampling(128, 256, 1, 0)
        self.exconv_s4tos1_stage4_b2tob3 = upsampling(256, 32, 3)
        self.exconv_s4tos2_stage4_b2tob3 = upsampling(256, 64, 2)
        self.exconv_s4tos3_stage4_b2tob3 = upsampling(256, 128, 1)


        self.exconv_s2tos1_stage4_b3tofinal = upsampling(64, 32, 1)
        self.exconv_s3tos1_stage4_b3tofinal = upsampling(128, 32, 2)
        self.exconv_s4tos1_stage4_b3tofinal = upsampling(256, 32, 3)

        self.final_conv = nn.Conv2d(32, 17, kernel_size=1, stride=1)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)





    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)



    def _make_layer_v2(self, block, planes, blocks):
        norm_layer = self._norm_layer
        self.inplanes = planes
        layers = []
        layers.append(block(self.inplanes, planes,  norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)



    def forward(self,x):

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        #### stage 1 ####
        out_s1 = self.stage1(x)
        down_s1tos2_stage1 = self.exconv_s1tos2_stage1(out_s1)
        s1tos1_stage1 = self.exconv_s1tos1_stage1(out_s1)

        ### stage 2 #####
        out_s1 = s1tos1_stage1
        out_s1 = self.stage2_s1_b1(out_s1)
        out_s2 = self.stage2_s2_b1(down_s1tos2_stage1)
        out_s3 = self.exconv_s2tos3_stage2(out_s2)


        down_s1tos2_stage2 = self.exconv_s1tos2_stage2(out_s1)
        up_s2tos1_stage2 = self.exconv_s2tos1_stage2(out_s2)

        ### stage 3 ####
        #### block 1 ####
        out_s1 = out_s1 + up_s2tos1_stage2
        out_s2 = down_s1tos2_stage2 + out_s2

        out_s1 = self.stage3_s1_b1(out_s1)
        out_s2 = self.stage3_s2_b1(out_s2)
        out_s3 = self.stage3_s3_b1(out_s3)

        #### block 2 ####
        down_s1tos2_stage3 = self.exconv_s1tos2_stage3_b1tob2(out_s1)
        down_s1tos3_stage3 = self.exconv_s1tos3_stage3_b1tob2(out_s1)
        up_s2tos1_stage3 = self.exconv_s2tos1_stage3_b1tob2(out_s2)
        down_s2tos3_stage3 = self.exconv_s2tos3_stage3_b1tob2(out_s2)
        up_s3tos1_stage3 = self.exconv_s3tos1_stage3_b1tob2(out_s3)
        up_s3tos2_stage3 = self.exconv_s3tos2_stage3_b1tob2(out_s3)

        out_s1 = out_s1 + up_s2tos1_stage3 + up_s3tos1_stage3
        out_s2 = down_s1tos2_stage3 + out_s2 + up_s3tos2_stage3
        out_s3 = down_s1tos3_stage3 + down_s2tos3_stage3 + out_s3

        out_s1 = self.stage3_s1_b2(out_s1)
        out_s2 = self.stage3_s2_b2(out_s2)
        out_s3 = self.stage3_s3_b2(out_s3)

        #### block 3 ####
        down_s1tos2_stage3 = self.exconv_s1tos2_stage3_b2tob3(out_s1)
        down_s1tos3_stage3 = self.exconv_s1tos3_stage3_b2tob3(out_s1)
        up_s2tos1_stage3 = self.exconv_s2tos1_stage3_b2tob3(out_s2)
        down_s2tos3_stage3 = self.exconv_s2tos3_stage3_b2tob3(out_s2)
        up_s3tos1_stage3 = self.exconv_s3tos1_stage3_b2tob3(out_s3)
        up_s3tos2_stage3 = self.exconv_s3tos2_stage3_b2tob3(out_s3)

        out_s1 = out_s1 + up_s2tos1_stage3 + up_s3tos1_stage3
        out_s2 = down_s1tos2_stage3 + out_s2 + up_s3tos2_stage3
        out_s3 = down_s1tos3_stage3 + down_s2tos3_stage3 + out_s3

        out_s1 = self.stage3_s1_b3(out_s1)
        out_s2 = self.stage3_s2_b3(out_s2)
        out_s3 = self.stage3_s3_b3(out_s3)

        #### block 4 ####
        down_s1tos2_stage3 = self.exconv_s1tos2_stage3_b3tob4(out_s1)
        down_s1tos3_stage3 = self.exconv_s1tos3_stage3_b3tob4(out_s1)
        up_s2tos1_stage3 = self.exconv_s2tos1_stage3_b3tob4(out_s2)
        down_s2tos3_stage3 = self.exconv_s2tos3_stage3_b3tob4(out_s2)
        up_s3tos1_stage3 = self.exconv_s3tos1_stage3_b3tob4(out_s3)
        up_s3tos2_stage3 = self.exconv_s3tos2_stage3_b3tob4(out_s3)


        out_s4 = self.exconv_s3tos4_stage3(out_s3)

        out_s1 = out_s1 + up_s2tos1_stage3 + up_s3tos1_stage3
        out_s2 = down_s1tos2_stage3 + out_s2 + up_s3tos2_stage3
        out_s3 = down_s1tos3_stage3 + down_s2tos3_stage3 + out_s3

        out_s1 = self.stage3_s1_b4(out_s1)
        out_s2 = self.stage3_s2_b4(out_s2)
        out_s3 = self.stage3_s3_b4(out_s3)


        ### stage 4 #####
        #### block 1 ####
        down_s1tos2_stage4 = self.exconv_s1tos2_stage4(out_s1)
        down_s1tos3_stage4 = self.exconv_s1tos3_stage4(out_s1)
        up_s2tos1_stage4 = self.exconv_s2tos1_stage4(out_s2)
        down_s2tos3_stage4 = self.exconv_s2tos3_stage4(out_s2)
        up_s3tos1_stage4 = self.exconv_s3tos1_stage4(out_s3)
        up_s3tos2_stage4 = self.exconv_s3tos2_stage4(out_s3)

        out_s1 = out_s1 + up_s2tos1_stage4 + up_s3tos1_stage4
        out_s2 = down_s1tos2_stage4 + out_s2 + up_s3tos2_stage4
        out_s3 = down_s1tos3_stage4 + down_s2tos3_stage4 + out_s3
        out_s4 = out_s4

        out_s1 = self.stage4_s1_b1(out_s1)
        out_s2 = self.stage4_s2_b1(out_s2)
        out_s3 = self.stage4_s3_b1(out_s3)
        out_s4 = self.stage4_s4_b1(out_s4)

        #### block 2 ####
        down_s1tos2_stage4 = self.exconv_s1tos2_stage4_b1tob2(out_s1)
        down_s1tos3_stage4 = self.exconv_s1tos3_stage4_b1tob2(out_s1)
        down_s1tos4_stage4 = self.exconv_s1tos4_stage4_b1tob2(out_s1)
        up_s2tos1_stage4 = self.exconv_s2tos1_stage4_b1tob2(out_s2)
        down_s2tos3_stage4 = self.exconv_s2tos3_stage4_b1tob2(out_s2)
        down_s2tos4_stage4 = self.exconv_s2tos4_stage4_b1tob2(out_s2)
        up_s3tos1_stage4 = self.exconv_s3tos1_stage4_b1tob2(out_s3)
        up_s3tos2_stage4 = self.exconv_s3tos2_stage4_b1tob2(out_s3)
        down_s3tos4_stage4 = self.exconv_s3tos4_stage4_b1tob2(out_s3)
        up_s4tos1_stage4 = self.exconv_s4tos1_stage4_b1tob2(out_s4)
        up_s4tos2_stage4 = self.exconv_s4tos2_stage4_b1tob2(out_s4)
        up_s4tos3_stage4 = self.exconv_s4tos3_stage4_b1tob2(out_s4)

        out_s1 = out_s1 + up_s2tos1_stage4 + up_s3tos1_stage4 + up_s4tos1_stage4
        out_s2 = down_s1tos2_stage4 + out_s2 + up_s3tos2_stage4 + up_s4tos2_stage4
        out_s3 = down_s1tos3_stage4 + down_s2tos3_stage4 + out_s3 + up_s4tos3_stage4
        out_s4 = down_s1tos4_stage4 + down_s2tos4_stage4 + down_s3tos4_stage4 + out_s4


        out_s1 = self.stage4_s1_b2(out_s1)
        out_s2 = self.stage4_s2_b2(out_s2)
        out_s3 = self.stage4_s3_b2(out_s3)
        out_s4 = self.stage4_s4_b2(out_s4)

        #### block 3 ####
        down_s1tos2_stage4 = self.exconv_s1tos2_stage4_b2tob3(out_s1)
        down_s1tos3_stage4 = self.exconv_s1tos3_stage4_b2tob3(out_s1)
        down_s1tos4_stage4 = self.exconv_s1tos4_stage4_b2tob3(out_s1)
        up_s2tos1_stage4 = self.exconv_s2tos1_stage4_b2tob3(out_s2)
        down_s2tos3_stage4 = self.exconv_s2tos3_stage4_b2tob3(out_s2)
        down_s2tos4_stage4 = self.exconv_s2tos4_stage4_b2tob3(out_s2)
        up_s3tos1_stage4 = self.exconv_s3tos1_stage4_b2tob3(out_s3)
        up_s3tos2_stage4 = self.exconv_s3tos2_stage4_b2tob3(out_s3)
        down_s3tos4_stage4 = self.exconv_s3tos4_stage4_b2tob3(out_s3)
        up_s4tos1_stage4 = self.exconv_s4tos1_stage4_b2tob3(out_s4)
        up_s4tos2_stage4 = self.exconv_s4tos2_stage4_b2tob3(out_s4)
        up_s4tos3_stage4 = self.exconv_s4tos3_stage4_b2tob3(out_s4)

        out_s1 = out_s1 + up_s2tos1_stage4 + up_s3tos1_stage4 + up_s4tos1_stage4
        out_s2 = down_s1tos2_stage4 + out_s2 + up_s3tos2_stage4 + up_s4tos2_stage4
        out_s3 = down_s1tos3_stage4 + down_s2tos3_stage4 + out_s3 + up_s4tos3_stage4
        out_s4 = down_s1tos4_stage4 + down_s2tos4_stage4 + down_s3tos4_stage4 + out_s4

        out_s1 = self.stage4_s1_b3(out_s1)
        out_s2 = self.stage4_s2_b3(out_s2)
        out_s3 = self.stage4_s3_b3(out_s3)
        out_s4 = self.stage4_s4_b3(out_s4)

        ############

        out_s2 = self.exconv_s2tos1_stage4_b3tofinal(out_s2)
        out_s3 = self.exconv_s3tos1_stage4_b3tofinal(out_s3)
        out_s4 = self.exconv_s4tos1_stage4_b3tofinal(out_s4)

        out = out_s1 + out_s2 + out_s3 + out_s4

        out = self.final_conv(out)



        return out



def hrnet(**kwargs):
    model = HRNet([4,4], **kwargs)
    return model

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext50_32x4d']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d']))
    return model