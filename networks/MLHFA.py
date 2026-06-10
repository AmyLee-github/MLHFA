import torch
from torch import nn
from networks.resnet import resnet50
from networks.srm_conv import SRMConv2d_simple
from networks.x_attn import CrossAttention
from networks.SEAttention import SEAttention
import torch.nn.functional as F


class MLHFA(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        # 初始化模型组件
        self.cam = CrossAttention(num_channels=3, num_heads=1)
        self.se = SEAttention(channel=3,reduction=3)
        self.srm = SRMConv2d_simple()
        self.disc = resnet50(pretrained=True)
        self.disc.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.disc.fc = nn.Linear(2048, 1)

    def forward(self, x_f, x_p):
        """
        前向传播逻辑
        :param x_f: 特征输入
        :param x_p: 图像输入
        :return: 模型输出
        """
        x_f_p = self.cam(x_f, x_p)
        x = self.se(x_f_p)
        x = F.interpolate(x, (256, 256), mode='bilinear')
        x = self.srm(x)
        x = self.disc(x)
        return x


if __name__ == '__main__':
    # 示例输入
    x_f = torch.randn(64, 3, 64, 64)
    x_p = torch.randn(64, 3, 64, 64)
    model = MLHFA(pretrain=True)
    output = model(x_f, x_p)
    print(output.shape)
    print(model)
