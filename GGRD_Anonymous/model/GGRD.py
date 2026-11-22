import torch
from torch import nn
from model.diffusion import Diffusion
from model.reconstruction import Reconstruction
from model.PreProcess import getMultiData_S
from model.embedding import  PositionalEmbedding
class GGRD(nn.Module):
    def __init__(self, time_steps, beta_start, beta_end, window_size, model_dim, ff_dim, atten_dim, feature_num,
                 time_num, block_num, head_num, dropout, device,  t, R, L, patchsize):
        super(GGRD, self).__init__()
        self.device = device
        self.window_size = window_size
        self.t = t
        num_patches = (window_size + patchsize - 1) // patchsize
        self.getmultidata_s = getMultiData_S(
            R = R, 
            L = L,
            patchsize = patchsize
        )
        self.diffusion = Diffusion(
            time_steps=time_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )
        self.reconstruction = Reconstruction(
            window_size=window_size,
            model_dim=model_dim,
            ff_dim=ff_dim,
            atten_dim=atten_dim,
            feature_num=feature_num,
            time_num=time_num,
            block_num=block_num,
            head_num=head_num,
            dropout=dropout,
            R = R,
            num_patches = num_patches
        )
        self.position_embedding = PositionalEmbedding(feature_num)

    def forward(self, data, time):

        data = self.position_embedding(data) 
        MutiData, S = self.getmultidata_s(data)
        # B R T D   and  B R P D D 
        #torch.save(S, 'brpdd.pt')
        # torch.save(data, 'data.pt')
        bt = torch.full((data.shape[0],), self.t).to(self.device)
        sample_noise = torch.randn_like(MutiData).float().to(self.device)

        noise_data = self.diffusion.q_sample(MutiData, bt, sample_noise) # B R T D
        
        recon = self.reconstruction(data, noise_data, time, S)

        return  recon 