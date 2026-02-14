import torch
from torch import nn

class GroupedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1):
        super(GroupedAttentionGate,self).__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size,stride=1,padding=kernel_size//2,groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size,stride=1,padding=kernel_size//2,groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1+x1)
        psi = self.psi(psi)

        return x*psi
    

# ------------张量测试---------------
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x1 = torch.randn(1, 32, 256, 256, device=device)
    g1 = torch.randn(1, 32, 256, 256, device=device)

    gag =  GroupedAttentionGate(32, 32, 32, kernel_size=3, groups=1).to(device)
    print(gag)
    y = gag(x1, g1)
    
    print("Input x1:", x1.shape)
    print("Input g1:", g1.shape)
    print("Output y:", y.shape)