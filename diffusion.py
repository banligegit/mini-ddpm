"""Forward noising, posterior-variance ancestral DDPM and eta=0 DDIM.

Code index t=0 denotes mathematical time 1; previous=-1 denotes clean x_0.
sample() uses respaced ancestral DDPM; sample_ddim() uses deterministic DDIM.
Run comparison: python compare_samplers.py. Outputs: runs/ddim_comparison/.
"""
import torch


class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        if timesteps < 2 or not 0 < beta_start <= beta_end < 1:
            raise ValueError('Invalid diffusion schedule')
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64, device=device)
        self.alpha_bar = (1 - self.betas).cumprod(0).float()
        self.betas = self.betas.float()

    def q_sample(self, x0, t, noise=None):
        """Draw x_t directly, without running t forward transitions."""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alpha_bar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def sampling_indices(self, steps):
        if not 2 <= steps <= self.timesteps:
            raise ValueError('steps must lie in [2, timesteps]')
        return torch.linspace(0, self.timesteps - 1, steps).round().long().tolist()

    def posterior(self, xt, eps, t, previous, clip=True):
        """q(x_s | x_t, estimated x_0), s < t, including skipped intervals."""
        at = self.alpha_bar[t]
        ap = self.alpha_bar[previous] if previous >= 0 else torch.ones_like(at)
        x0 = (xt - (1 - at).sqrt() * eps) / at.sqrt()
        if clip:
            x0 = x0.clamp(-1, 1)
        alpha = at / ap
        beta = 1 - alpha
        mean = ap.sqrt() * beta / (1 - at) * x0 + alpha.sqrt() * (1 - ap) / (1 - at) * xt
        variance = ((1 - ap) / (1 - at) * beta).clamp_min(0)
        return mean, variance

    def ddim_step(self, xt, eps, t, previous, clip=True):
        """One deterministic DDIM transition (eta=0) on the original time grid.

        A clipped clean-image prediction is paired with a re-derived epsilon,
        preserving xt = sqrt(at)*x0 + sqrt(1-at)*epsilon. This is the clipping
        convention used in OpenAI improved-diffusion's DDIM implementation.
        It is not the DDPM posterior mean with its random term removed.
        """
        at = self.alpha_bar[t]
        ap = self.alpha_bar[previous] if previous >= 0 else torch.ones_like(at)
        x0 = (xt - (1-at).sqrt() * eps) / at.sqrt()
        if clip:
            x0 = x0.clamp(-1, 1)
            eps = (xt - at.sqrt() * x0) / (1-at).sqrt()
        return ap.sqrt() * x0 + (1-ap).sqrt() * eps

    @torch.inference_mode()
    def sample_ddim(self, model, n=64, steps=1000, seed=2026, device='cpu'):
        """Same initial noise as sample(), but no random transition noise."""
        model.eval()
        indices = self.sampling_indices(steps)
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(n, 1, 28, 28, generator=generator, device=device)
        for i in range(len(indices)-1, -1, -1):
            t = indices[i]
            previous = indices[i-1] if i else -1
            times = torch.full((n,), t, device=device, dtype=torch.long)
            x = self.ddim_step(x, model(x, times), t, previous)
        return x.clamp(-1, 1)

    @torch.inference_mode()
    def sample(self, model, n=64, steps=1000, seed=2026, device='cpu'):
        model.eval()
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(n, 1, 28, 28, generator=generator, device=device)
        indices = self.sampling_indices(steps)
        # Network always receives the ORIGINAL time index (0..999).
        for i in range(len(indices) - 1, -1, -1):
            t = indices[i]
            previous = indices[i - 1] if i else -1
            times = torch.full((n,), t, device=device, dtype=torch.long)
            mean, variance = self.posterior(x, model(x, times), t, previous)
            if previous >= 0:
                z = torch.randn(x.shape, generator=generator, device=device)
                x = mean + variance.sqrt() * z
            else:
                x = mean  # Last transition is deterministic; never add noise here.
        return x.clamp(-1, 1)
