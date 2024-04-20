import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, in_dim):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        batch_size, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(y).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(y).view(batch_size, channels, -1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out


if __name__ == '__main__':
    x=torch.randn(24, 256, 28, 28)
    x1 = torch.randn(24, 256, 28, 28)
    model=Attention(in_dim=256)
    y=model(x,x1)
    print(y.shape)