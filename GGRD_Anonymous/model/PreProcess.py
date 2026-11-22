import torch
from torch import nn
import torch.nn.functional as F
class getMultiData_S(nn.Module):
    def __init__(self, R, L, patchsize):
        super(getMultiData_S, self).__init__()
        self.R = R
        self.L = L
        self.patchsize = patchsize
       
    def compute_patch_covariance(self,x):
        B, R, T, D = x.shape
        result = []

        for b in range(B):
            sample_r = []
            for r in range(R):
                patches = []
                seq = x[b, r]  # [T, D]
                num_patches = (T + self.patchsize - 1) // self.patchsize  # ceil(T / patch_size)

                for i in range(num_patches):
                    start = i * self.patchsize
                    end = min(start + self.patchsize, T)
                    patch = seq[start:end]  # [patch_size', D]
                    patch = patch / (patch.norm(p=2, dim=0, keepdim=True) + 1e-8)
                    patch_cov = patch.T @ patch  # [D, D]
                    patches.append(patch_cov)

                patches = torch.stack(patches, dim=0)  # [P, D, D]
                sample_r.append(patches)

            sample_r = torch.stack(sample_r, dim=0)  # [R, P, D, D]
            result.append(sample_r)

        result = torch.stack(result, dim=0)  # [B, R, P, D, D]
        return result

    def multi_resolution_avg_pool(self, data):

        B, T, D = data.shape
        results = []
        x = data.transpose(1, 2)  # (B, D, T)

        for r in range(1, self.R + 1):
            window_size = r * self.L
            pad_len = (window_size - (T % window_size)) % window_size  
            padded_x = F.pad(x, (0, pad_len), mode="constant", value=0)  

            pooled = F.avg_pool1d(padded_x, kernel_size=window_size, stride=window_size)  

            upsampled = F.interpolate(pooled, size=T, mode="linear", align_corners=False)  

            results.append(upsampled.transpose(1, 2))  # (B, T, D)

        return torch.stack(results, dim=1)  # (B, R, T, D)

    def multi_resolution_sliding_average(self,data):
        B, T, D = data.shape
        results = []
        for r in range(1, self.R + 1):
            window_size = r * self.L
            padded_T = T + window_size - 1
            repeat_factor = (padded_T + T - 1) // T
            padded_data = data.repeat(1, repeat_factor, 1)[:, :padded_T, :]  # (B, padded_T, D)

            smoothed = []
            for t in range(T):
                window = padded_data[:, t:t + window_size, :]  # (B, window_size, D)
                avg = window.mean(dim=1)  # (B, D)
                smoothed.append(avg)

            smoothed_tensor = torch.stack(smoothed, dim=1)  # (B, T, D)
            results.append(smoothed_tensor)

        return torch.stack(results, dim=1)  # (B, R, T, D)

    def forward(self, data):
    
        new_data = self.multi_resolution_sliding_average(data)
        #new_data = self.multi_resolution_avg_pool(data)
        S = self.compute_patch_covariance(new_data)

        return new_data, S# B R T D   和 B R P D D 
