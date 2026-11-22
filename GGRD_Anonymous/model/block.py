import torch
import torch.nn.functional as F
from torch import nn
from model.attention import MultiHeadAttention, GGAttention

class TimeBlock(nn.Module):  
    def __init__(self, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(TimeBlock, self).__init__()
        self.MSA = MultiHeadAttention(model_dim, model_dim, model_dim, atten_dim, head_num, dropout, True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
    
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):#x B T model
        
        x = self.MSA(x, x)
        
        residual = x.clone()
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm(x + residual)

class FeatureBlock(nn.Module):   
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(FeatureBlock, self).__init__()
 
        self.MSA = MultiHeadAttention(window_size, window_size, window_size, atten_dim, head_num, dropout, True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):#x B T model

        
        x = x.permute(0, 2, 1)
        x = self.MSA(x, x)#x B  model  T
        x = x.permute(0, 2, 1) #x B T model

        residual = x.clone()
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm(x + residual)

class Time_Feature_Encoder(nn.Module):
    def __init__(self,  window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(Time_Feature_Encoder, self).__init__()
        self.time_block = TimeBlock(model_dim, ff_dim, atten_dim, head_num, dropout)
        self.feature_block = FeatureBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dropout)

        self.conv1 = nn.Conv1d(in_channels=2 * model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm1 = nn.LayerNorm(2 * model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        time_x = self.time_block(x)  #B T model 
        feature_x = self.feature_block(x)#B T model 

        x = self.norm1(torch.cat([time_x, feature_x], dim=-1))

        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm2(x)

class GGNBlock(nn.Module):
    def __init__(self,  feature_num, model_dim, ff_dim, atten_dim, head_num, dropout, window_size, num_patches):
        super(GGNBlock, self).__init__()
        self.gga = GGAttention(model_dim, feature_num, model_dim, atten_dim, head_num, dropout, True, window_size, num_patches)
        #query_dim, key_dim, value_dim, d_model, num_heads, dropout, residual
        #self.msa0 = MultiHeadAttention(model_dim, feature_num, model_dim, atten_dim, head_num, dropout, True)
        self.msa = MultiHeadAttention(model_dim, feature_num, model_dim, atten_dim, head_num, dropout, True)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim)
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x, data, S):
        x = self.gga(x, data, S)
        #x = self.msa0(x, data)
        a = x 

        x = self.ffn(x)

        x = self.norm1( a + x )

        b = x

        x = self.msa(x, data)

        x = x + b

        return self.norm2(x)

class TimeAugmentEncoder(nn.Module):  #B T model  
    def __init__(self, model_dim, ff_dim, dropout):
        super(TimeAugmentEncoder, self).__init__()
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels=model_dim, out_channels=model_dim, kernel_size=3, padding=1, groups=model_dim),  # 深度可分离卷积
            nn.GELU(),
            nn.BatchNorm1d(model_dim)
        )

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, model_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):#x B T model
        
        residual = x.clone()
        x = self.local_conv(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        x = self.norm1(x + residual)

        x_ffn = self.ffn(x)
      
        return self.norm2(x + self.dropout(x_ffn))
