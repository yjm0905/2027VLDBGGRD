from torch import nn
from model.GGNet import  GGNet
class Reconstruction(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num, head_num,
                 dropout, R, num_patches):
        super(Reconstruction, self).__init__()
        self.ggnet = GGNet(window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num, head_num,dropout,num_patches)
        self.R = R
    def get_mean_S(self, S, r):#S : B R P D D 
        S_selected = S[:, r:]  # shape: [B, R-r, P, D, D]
        S_mean = S_selected.mean(dim=1)  
        return S_mean

    def forward(self, data, noise_data, time, S):#data: B T D  noise_data : B R T D        
        x_hat = None
        for r in range(self.R - 1, 0, -1):
            
            if  r == self.R -1:                #time  B 5           S: B R P D D     
                x_hat = self.ggnet( data, 
                                    noise_data[:, r, :, :],  # B T D 
                                    noise_data[:, r-1, :, :], 
                                    time, 
                                    S[: , r, :, :, :], #  B P D D 
                                    S[: , r-1, :, :, :],
                                    r
                                    )
            else:
                x_hat = self.ggnet( data, 
                                    x_hat, 
                                    noise_data[:, r-1, :, :], 
                                    time, 
                                    self.get_mean_S(S, r),#B P D D
                                    S[: , r-1, :, :, :],#B P D D
                                    r
                                    )                
        return x_hat
            


        
