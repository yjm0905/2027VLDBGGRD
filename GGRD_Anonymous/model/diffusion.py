import torch

class Diffusion:
    def __init__(self, time_steps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.betas = torch.linspace(beta_start, beta_end, time_steps).float()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
  
    @staticmethod
    def _extract(data, batch_t, shape):
        batch_size = batch_t.shape[0] 
        data = data.to(batch_t.device)
        out = torch.gather(data, -1, batch_t.long())

        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start, batch_t, noise):
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, batch_t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, batch_t, x_start.shape)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise 
      
        return x_noisy
