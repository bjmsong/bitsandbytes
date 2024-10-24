import torch

from bitsandbytes.nn import Linear8bitLt
from bitsandbytes.nn.triton_based_modules import SwitchBackLinear

M, K, N = 256, 1024, 4096 
# torch
standard = torch.nn.Linear(K, N).cuda().half()
weight = torch.randn(N, K, dtype=torch.float16).cuda()
with torch.no_grad():  # 禁用梯度更新
    standard.weight.copy_(weight)
    standard.bias = None

# bitsandbytes in triton
vector_wise_quantization = True
switchback = SwitchBackLinear(K, N, vector_wise_quantization=vector_wise_quantization).cuda().half()
switchback.eval()  # training mode -> eval mode
switchback.weight.data.copy_(standard.weight)
switchback.bias = None

# bitsandbytes in cuda ?
baseline = Linear8bitLt(K, N).cuda().half()
baseline.weight.data.copy_(standard.weight)
baseline.bias = None

x1 = torch.randn(M, K, device='cuda', dtype=torch.float16)
x2 = x1.clone().detach()
x3 = x1.clone().detach()

out_standard = standard(x1)
out_sb = switchback(x2)
out_baseline = baseline(x3)

err_sb = (out_standard - out_sb).abs().mean()
err_baseline = (out_standard - out_baseline).abs().mean()
print("OUT", err_sb, err_baseline)
assert err_sb < 2 * err_baseline