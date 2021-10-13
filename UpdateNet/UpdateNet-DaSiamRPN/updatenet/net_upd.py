import torch 
import torch.nn as nn


#使用残差块,输出256通道
class UpdateResNet256(nn.Module):
    def __init__(self, config=None):
        super(UpdateResNet256, self).__init__()
        self.update = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),
        )
    def forward(self, x, x0):
        #t = torch.cat((x, y, z), 0)
        # x0 is residual
        response = self.update(x)        
        response += x0
        return response

if __name__ == '__main__':

    # network test
    net = UpdateResNet256()
    net.eval()
    

#使用残差块,输出512通道
class UpdateResNet512(nn.Module):
    def __init__(self, config=None):
        super(UpdateResNet512, self).__init__()
        self.update = nn.Sequential(
            nn.Conv2d(1536, 192, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 512, 1),
        )
    def forward(self, x, x0):
        #t = torch.cat((x, y, z), 0)
        # x0 is residual
        response = self.update(x)        
        response += x0
        return response

if __name__ == '__main__':

    # network test
    net = UpdateResNet()
    net.eval()

#没有残差块 输出256通道
class UpdateNet(nn.Module):
    def __init__(self, config=None):
        super(UpdateNet, self).__init__()
        self.update = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),            
            #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        #t = torch.cat((x, y, z), 0)
        response = self.update(x)
        return response

if __name__ == '__main__':

    # network test
    net = UpdateNet()
    net.eval()



