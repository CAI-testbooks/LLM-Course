import torch
import torch.nn as nn
class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0)):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class VisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = 160
        self.Block1=nn.Sequential(
            nn.Conv2d(1, 16, (1, 41),padding=(0,20)),#F1=16
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16*4, (14, 1)),#D=16
            nn.BatchNorm2d(16*4),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(0.5)
        )

        self.Block2=nn.Sequential(
            SeparableConv2D(16*4, 16, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1,5)),
            nn.Dropout(0.5)
        )
        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 20)#80*50

        )

    def forward(self, x):
        x=self.Block1(x)
        x=self.Block2(x)
        x=self.fc(x)
        return x

class Model(nn.Module):
    def __init__(self,embsize=160):
        super().__init__()
        self.T = 160
        self.Block1=nn.Sequential(
            nn.Conv2d(1, 64, (1, 41),padding=(0,20)),#F1=16
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32 * 64, (128, 1), groups=64),  # D=16
            nn.BatchNorm2d(64 * 32),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(p=0.5),

        )

        self.Block2=nn.Sequential(
            SeparableConv2D(64 * 32, 64, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 5)),
            nn.Dropout(p=0.5),

        )
        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8, embsize)
        )
        self.classifier=nn.Sequential(nn.Linear(embsize,40))

    def forward(self, x):
        x=self.Block1(x)
        x=self.Block2(x)
        x=self.fc(x)
        emb=x
        y=self.classifier(emb)
        return emb,y


class EmbModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc=nn.Sequential(nn.Flatten(),nn.Linear(4096,160))
    def forward(self, x):
        x = self.fc(x)
        x=x/x.norm(dim=1, keepdim=True)
        return x

