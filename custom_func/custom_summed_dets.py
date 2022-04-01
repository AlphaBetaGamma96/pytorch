import torch
from torch import Tensor
from typing import Tuple
from functorch import vmap, jacrev, grad
print(torch.__version__)

torch.manual_seed(0)

def NaiveLogSumExpEnvLogDomainStable(matrices: Tensor, log_envs: Tensor) -> Tuple[Tensor, Tensor]:
  sgns, logabss = torch.slogdet(matrices * torch.exp(log_envs))
  max_logabs_envs = torch.max(logabss, keepdim=True, dim=-1)[0] #grab the value only (no indices)

  scaled_dets = sgns*torch.exp( logabss - max_logabs_envs )  #max subtraction
  summed_scaled_dets = scaled_dets.sum(keepdim=True, dim=-1) #sum

  global_logabs = (max_logabs_envs + (summed_scaled_dets).abs().log()).squeeze(-1) #add back in max value and take logabs
  global_sgn = summed_scaled_dets.sign().squeeze(-1) #take sign
  return global_sgn, global_logabs



m = torch.randn(4096,1,2,2,requires_grad=True, device='cuda')
loge = torch.randn(4096,1,2,2,requires_grad=True, device='cuda')


#sgn, logabs = NaiveLogSumExpEnvLogDomainStable(m, loge)
#yeet = torch.autograd.grad(logabs, [m, loge], [torch.ones_like(logabs), torch.ones_like(logabs)])
#print(yeet)

def get_logabs(m, loge):
    sgn, logabs = torch.summed_dets(m, loge)
    return logabs

logabs = get_logabs(m, loge)
#out = vmap(jacrev(get_logabs, argnums=(0,1)), in_dims=(0, 0))(m, loge)
out = vmap(jacrev(jacrev(jacrev(get_logabs, argnums=(0,1)), argnums=(0,1)), argnums=(0,1)), in_dims=(0, 0))(m, loge)
print(out)
#out = vmap(jacrev(get_logabs(m, loge)))(m, loge)
"""
(tensor([[[[-0.0488, -2.2448],
          [-0.3612,  0.3111]]],


        [[[ 0.5555, -1.8408],
          [-1.7133,  0.2045]]]], device='cuda:0', grad_fn=<ViewBackward0>), tensor([[[[ 0.0452,  0.9548],
          [ 0.9548,  0.0452]]],


        [[[-0.0671,  1.0671],
          [ 1.0671, -0.0671]]]], device='cuda:0', grad_fn=<ViewBackward0>))
"""
