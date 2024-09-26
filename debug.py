import torch

from bitsandbytes.nn import Linear8bitLt
from bitsandbytes.nn.triton_based_modules import SwitchBackLinear

vector_wise_quantization = True

standard = torch.nn.Linear(2, 2).cuda().half()
weight = torch.tensor([[0.1,0.3],[0.2,0.4]])
with torch.no_grad():  # 禁用梯度更新
    standard.weight.copy_(weight)
    standard.bias = None

switchback = SwitchBackLinear(2, 2, vector_wise_quantization=vector_wise_quantization).cuda().half()
switchback.eval()  # training mode -> eval mode
baseline = Linear8bitLt(2, 2).cuda().half()

switchback.weight.data.copy_(standard.weight)
switchback.bias = None
baseline.weight.data.copy_(standard.weight)
baseline.bias = None

x1 = torch.tensor([[0.1,0.2],[0.3,0.4]], device='cuda', dtype=torch.float16)
x2 = x1.clone().detach()
x3 = x1.clone().detach()

out_standard = standard(x1)

print(x2.dtype)
out_sb = switchback(x2)

out_baseline = baseline(x3)

err_sb = (out_standard - out_sb).abs().mean()
err_baseline = (out_standard - out_baseline).abs().mean()
print("OUT", err_sb, err_baseline)
assert err_sb < 2 * err_baseline