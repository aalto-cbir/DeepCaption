# Load resnet152.pth provided by https://github.com/ruotianluo/pytorch-resnet/
# Follow code from: https://github.com/ruotianluo/pytorch-resnet/blob/master/resnet.py
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet
import torch.utils.model_zoo as model_zoo

model_urls = {
    # RGB input, normalized same way as input for Torchvision models,
    # pixel values between 0 and 1:
    'resnet152caffe_torchvision': 'resnet152-caffe-torchvision-style-95e0e999.pth',
    # BGR input, caffe normalization, pixel values between 0 and 255:
    'resnet152caffe_original': 'resnet152-caffe-original-4d2cc6ef.pth'
}


class CaffeResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(CaffeResNet, self).__init__(block, layers, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        for i in range(2, 5):
            getattr(self, 'layer%d' % i)[0].conv1.stride = (2, 2)
            getattr(self, 'layer%d' % i)[0].conv2.stride = (1, 1)


def resnet152caffe_torchvision(pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CaffeResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152caffe_torchvision'],
                                                 model_dir='models/external'))

    return model


def resnet152caffe_original(pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CaffeResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152caffe_original'],
                                                 model_dir='models/external'))

    return model
