import torch
import bitsandbytes as bnb


a = torch.randn(256, 512).half().cuda()
b = torch.randn(256, 512).half().cuda()
c1 = torch.matmul(a, b.t())
c2 = bnb.matmul(a, b)
c3 = bnb.matmul(a, b, 3)

err1 = torch.abs(c1 - c2).mean().item()
err2 = torch.abs(c1 - c3).mean().item()
assert err1 < 0.2
assert err2 < 0.2
print(err1, err2)