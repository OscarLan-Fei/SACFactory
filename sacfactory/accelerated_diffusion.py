"""Cosine DDPM schedule and conditional residual sampling on the selected CPU or CUDA device."""
import math
import torch
from functools import lru_cache


@lru_cache(maxsize=32)
def schedule(steps, device='cpu'):
    if not 10 <= steps <= 200:
        raise ValueError('Use 10 to 200 diffusion steps for the compact prototype.')
    s = torch.linspace(0, 1, steps+1, dtype=torch.float64)
    a = torch.cos((s+.008)/1.008*math.pi/2)**2
    a = a/a[0]
    beta = (1-a[1:]/a[:-1]).clamp(1e-5, .999).float()
    alpha = 1-beta
    abar = torch.cumprod(alpha, dim=0)
    previous = torch.cat([torch.ones(1), abar[:-1]])
    variance = beta*(1-previous)/(1-abar)
    return tuple(t.to(device) for t in (beta, alpha, abar, previous, variance))


def training_loss(model, x, z, steps, generator, reduction=True):
    _, _, abar, _, _ = schedule(steps, str(x.device))
    k = torch.randint(steps, (len(x),), generator=generator, device=x.device)
    noise = torch.randn(z.shape, generator=generator, device=x.device)
    level = abar[k, None]
    noisy = level.sqrt()*z+(1-level).sqrt()*noise
    predicted = model(noisy, k.float()/max(steps-1, 1), x)
    loss = ((noise-predicted)**2).mean(dim=1)
    return loss.mean() if reduction else loss


@torch.no_grad()
def sample(model, x, draws, steps, seed, clean_limit):
    model.eval()
    generator = torch.Generator(device=x.device).manual_seed(seed)
    repeated = x.repeat_interleave(draws, dim=0)
    condition = model.encoder(repeated)
    z = torch.randn((len(repeated), 1), generator=generator, device=x.device)
    beta, alpha, abar, previous, variance = schedule(steps, str(x.device))
    clipped = torch.zeros(len(repeated), device=x.device)
    for k in range(steps-1, -1, -1):
        index = torch.full((len(repeated),), k/max(steps-1, 1), device=x.device)
        noise = model.denoise(z, index, condition)
        clean = (z-(1-abar[k]).sqrt()*noise)/abar[k].sqrt()
        clipped += (clean[:, 0].abs() > clean_limit).float()
        # A disclosed training-only sampling guard; observed targets are never clipped.
        clean = clean.clamp(-clean_limit, clean_limit)
        mean = beta[k]*previous[k].sqrt()/(1-abar[k])*clean + (1-previous[k])*alpha[k].sqrt()/(1-abar[k])*z
        z = mean + variance[k].sqrt()*torch.randn(z.shape, generator=generator, device=x.device) if k else mean
    return z.reshape(len(x), draws).cpu().numpy(), (clipped.reshape(len(x), draws).mean(dim=1)/steps).cpu().numpy()
