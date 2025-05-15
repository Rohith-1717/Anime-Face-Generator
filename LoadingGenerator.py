import torch
import torch.nn as nn

nz = 100  
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz,512,4,1,0, bias=False),nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512,256,4,2,1, bias=False),nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1, bias=False),nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,3,4,2,1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)
    
def load_generator(path="generator.pth"):
    model = Generator()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model



