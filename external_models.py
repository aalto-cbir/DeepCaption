# Load models/external/resnet152.pth
# Follow code from: https://github.com/ruotianluo/pytorch-resnet/blob/master/resnet.py
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet152caffe': 'resnet152-caffe-95e0e999.pth'
}


class CaffeResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(CaffeResNet, self).__init__(block, layers, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        for i in range(2, 5):
            getattr(self, 'layer%d' % i)[0].conv1.stride = (2, 2)
            getattr(self, 'layer%d' % i)[0].conv2.stride = (1, 1)


def resnet152caffe(pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CaffeResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152caffe'],
                                                 model_dir='models/external'))

    return model
