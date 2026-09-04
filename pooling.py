import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        if self.training:
            n, c, h, w = x.size()
            kh, kw = (self.kernel_size, self.kernel_size) if isinstance(self.kernel_size, int) else self.kernel_size
            stride_h, stride_w = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
            pad_h, pad_w = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding

            out_h = int((h + 2*pad_h - kh) / stride_h) + 1
            out_w = int((w + 2*pad_w - kw) / stride_w) + 1

            x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
            x_unf = x_unf.view(n, c, kh*kw, out_h, out_w)

            x_unf = F.relu(x_unf)
            probs = x_unf / (x_unf.sum(dim=2, keepdim=True) + 1e-8)

            probs_flat = probs.view(n, c, kh*kw, -1).permute(0,1,3,2).contiguous().view(-1, kh*kw)
            samples = torch.multinomial(probs_flat, 1).view(n, c, out_h, out_w)

            x_unf_flat = x_unf.view(n, c, kh*kw, -1).permute(0,1,3,2).contiguous().view(-1, kh*kw)
            idx = samples.view(-1, 1).expand(-1, 1)
            out = torch.gather(x_unf_flat, 1, idx).view(n, c, out_h, out_w).float()
            return out
        else:
            return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)