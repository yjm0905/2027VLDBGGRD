from torch import nn
from model.block import Time_Feature_Encoder, GGNBlock, TimeAugmentEncoder
from model.embedding import TimeEmbedding, DataEmbedding
class GGNet(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num, head_num,
                 dropout, num_patches):
        super(GGNet, self).__init__()
        self.time_embed = TimeEmbedding(model_dim, time_num) #  Time B T 5
        self.data_embed = DataEmbedding(model_dim, feature_num) #  Data  B T D 
   
        self.TF_en = Time_Feature_Encoder(window_size, model_dim, ff_dim, atten_dim, head_num, dropout)  
        self.Time_Aug = TimeAugmentEncoder(model_dim, ff_dim, dropout)
        self.ggablocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.ggablocks.append(
                GGNBlock(feature_num, model_dim, ff_dim, atten_dim, head_num, dp, window_size, num_patches)
            )
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, feature_num)
        )

    def forward(self, data, XR, XR_1, time, S2, S1, r):
        mean_data = (XR + XR_1) / 2  # B T D 
        mean_S = (S2 + S1) / 2   # B P D D 
       
        new_data = self.data_embed(mean_data) 
        new_time = self.time_embed(time)  
        x = self.TF_en(new_data) 
       
        new_time = new_time + 0.01 * self.Time_Aug(new_time) 

        x = x + new_time
        #B T model/ B T D / B T D
        for block in self.ggablocks:
            x = block(x, data, mean_S)

        x = self.ffn(x)

        return x
