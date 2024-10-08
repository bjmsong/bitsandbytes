import torch

from bitsandbytes.nn import Linear8bitLt
from bitsandbytes.nn.triton_based_modules import SwitchBackLinear

# hyper parameter
dim = 256
batch = 32
vector_wise_quantization = True

standard = torch.nn.Linear(dim, 4 * dim).cuda().half()
switchback = SwitchBackLinear(dim, 4 * dim, vector_wise_quantization=vector_wise_quantization).cuda().half()
switchback.eval()  # training mode -> eval mode
baseline = Linear8bitLt(dim, 4 * dim).cuda().half()

switchback.weight.data.copy_(standard.weight)
switchback.bias.data.copy_(standard.bias)
baseline.weight.data.copy_(standard.weight)
baseline.bias.data.copy_(standard.bias)

x1 = torch.randn(batch, dim).cuda().half()
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