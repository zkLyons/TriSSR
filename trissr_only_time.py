import torch
import copy
from torch import nn
from mamba_ssm import Mamba2
from recbole.model.abstract_recommender import SequentialRecommender
import torch.nn.functional as F
from s5 import S5, S5Block
import math


class TriSSR(SequentialRecommender):
    def __init__(self, config, dataset):
        super(TriSSR, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"] 
        self.num_layers = config["num_layers"]   
        self.dropout_prob = config["dropout_prob"] 
        self.beta = config["beta"]
        self.norm_embedding = config['norm_embedding']
        
        # Hyperparameters for SSD Block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]  
        self.expand = config["expand"]  
        self.headdim = config['headdim']

        #s5
        # 最小时间间隔
        self.dt_min = config["dt_min"]
        # 最大时间间隔
        self.dt_max = config["dt_max"]
        # 状态宽度
        self.d_P = config["d_P"]
        # 隐藏层大小
        self.d_H = config["d_H"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size)  # 0 -> mask_token

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.BiSSD_layers = nn.ModuleList([
            BiSSDLayer(
                beta = self.beta,
                d_model=self.hidden_size,  
                d_state=self.d_state,      
                d_conv=self.d_conv,        
                expand=self.expand,        
                dropout=self.dropout_prob, 
                num_layers=self.num_layers,
                headdim = self.headdim,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                d_P=self.d_P,
                d_H=self.hidden_size,
            ) for _ in range(self.num_layers)
        ])

        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, cum_item_length, item_idx, flip_index,time_diff):
        # print(time_diff.shape)
        item_emb = self.item_embedding(item_seq)
        # 是否对初始嵌入归一化
        if self.norm_embedding == True:
            item_emb = self.dropout(item_emb)
            item_emb = self.LayerNorm(item_emb)

        for i in range(self.num_layers):
            item_emb = self.BiSSD_layers[i](item_emb, item_idx, flip_index,time_diff)

        # gather_last_token_output
        gather_index = cum_item_length - 1 # [B]
        seq_output = item_emb[0, gather_index, :]

        return seq_output
    
    def calculate_loss(self, item_id, item_id_list, cum_item_length, item_idx, flip_index,time_diff):
        # item_id应该是每一个用户交互过的最后一个物品id[itemX,itemY,itemZ....]  [batch_size]数据集在创建的时候使用了滑动窗口,例如u1:[i1,i2,i3,i4,i5],划分后的序列为：,u1:[i1],u1:[i1,i2],u1:[i1,i2,i3],u1:[i1,i2,i3,i4],各对应的目标物品为[i2],[i3],[i4],[i5]
        # item_id_list 应该是每个批次，用户交互的物品id列表拼接起来形成的，u1：[item1,item2,item3],u2:[item4,item5]...--->[item1,item2,item3,....item4,item5...]
        # cum_item_length:表示该批次当中每个用户交互的物品的累加长度(假设物品的seq设置为50，u1交互了3个物品，所以为3，长度补到50，u2交互了2个，所以值为52)，[3,52]
        item_seq = item_id_list.unsqueeze(0)     # [1, cat_dim such as 13297]
        item_idx = item_idx.unsqueeze(0)

        seq_output = self.forward(item_seq, cum_item_length, item_idx, flip_index,time_diff) # [B, hidden_size]
        pos_items = item_id                      # [B]

        test_item_emb = self.item_embedding.weight # [item_num, hidden_size]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) # [B, item_num]

        loss = self.loss_fct(logits, pos_items)
        return loss


    def full_sort_predict(self, item_id_list, cum_item_length, item_idx, flip_index,positive_i,time_diff):
        # positive_i:目标物品
        item_seq = item_id_list.unsqueeze(0)    # [1, cat_dim such as 13297]
        item_idx = item_idx.unsqueeze(0)
        
        seq_output = self.forward(item_seq, cum_item_length, item_idx, flip_index,time_diff) # [B, hidden_size]
        test_items_emb = self.item_embedding.weight # [item_num, hidden_size]

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        valid_loss = self.loss_fct(scores, positive_i)

        return scores,valid_loss



    
# 双向mamba2+FFN
class BiSSDLayer(nn.Module):
    def __init__(self, beta, d_model, d_state, d_conv, expand, dropout, num_layers, headdim, dt_min, dt_max, d_P, d_H):
        super().__init__()
        self.beta = beta
        # mamba2作为mamba的改进，加入了注意力机制，所以需要传入注意力头数。
        self.num_layers = num_layers
        # self.forward_ssd = Mamba2(
        #         # This module uses roughly 3 * expand * d_model^2 parameters
        #         d_model=d_model,  #模型特征维度
        #         d_state=d_state,  #状态维度
        #         headdim = headdim,  #注意力维度
        #         d_conv=d_conv,     # 卷积核尺寸
        #         expand=expand,     #特征拓展因子
        #     )
        
        # self.S5 = S5(
        #         width=d_H,
        #         state_width=d_P,
        #         dt_min = dt_min,
        #         dt_max =dt_max,
        #         )  

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
       
        self.dropout = nn.Dropout(dropout)
        #滤波层
        # self.Fc=FrequencyLayer(d_model=d_model)

        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
        self.inner_size=d_model*4
        self.timefourier=TimeFourier(d_model)
        # self.tri_expert = TriExpertFusion(d_model)
        
    def forward(self, item_emb, item_idx, flip_index,time_diff):
        # out_pass=self.Fc(item_emb)
        time_out=self.timefourier(time_diff)
         # forward ssd
        # forward_hidden_state = self.forward_ssd(item_emb, seq_idx=item_idx)
        # # backward ssd
        # filp_emb = item_emb[:, flip_index, :]
        # backward_hidden_state = self.forward_ssd(filp_emb, seq_idx=item_idx)
        # # backward_hidden_state=backward_hidden_state[:,flip_index,:]

        # hidden_states = forward_hidden_state + backward_hidden_state * self.beta + item_emb
               
        # out=self.tri_expert(out_pass,hidden_states,time_out)
        # hidden_states=out

        # SFC增强输出（低频基座+高频细节）
        # hidden_states=hidden_states+out_pass
        hidden_states=time_out
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.ffn(hidden_states)

        return hidden_states
class TimeFourier(nn.Module):
    def __init__(self, hidden_size, max_freq=10.0, num_bands=16):
        super().__init__()
        # 生成 num_bands 个频率，log-space 分布
        # 生成从1-10的16的等间隔的点。
        # 这样做是为了生成一组呈指数增长的频率（例如 1, 10, 100... 如果 max_freq 很大），而不是线性增长的频率。这种对数间隔的频率分布在时间编码中很常见，因为它允许模型同时捕获短期（高频）和长期（低频）的时间模式

        freqs = torch.logspace(0.0, math.log10(max_freq), num_bands)
        # 将freqs注册为缓冲区，不需要使用梯度下降进行学习。
        self.register_buffer('freqs', freqs)  # [num_bands]
        self.proj = nn.Linear(2 * num_bands, hidden_size)

    def forward(self, time_diff):
        # time_diff: [B, L]
        # 拓展维度： [B, L, num_bands]
        x = time_diff.unsqueeze(-1) * self.freqs  # 广播
        sin = torch.sin(x)
        cos = torch.cos(x)
        feat = torch.cat([sin, cos], dim=-1)      # [B, L, 2*num_bands]
        return self.proj(feat)                    # [B, L, D]



# class TriExpertFusion(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.freq_ffn  = FeedForward(d_model, d_model * 4)
#         self.mamba_ffn = FeedForward(d_model, d_model * 4)
#         self.time_ffn  = FeedForward(d_model, d_model * 4)

#         self.alpha_freq  = nn.Parameter(torch.tensor(1.0))
#         self.alpha_mamba = nn.Parameter(torch.tensor(1.0))
#         self.alpha_time  = nn.Parameter(torch.tensor(1.0))

#         self.fuse_proj = nn.Linear(3 * d_model, d_model)
#         self.LayerNorm = nn.LayerNorm(d_model)

#     def forward(self, freq_emb, mamba_emb,time_emb):
#         freq_out  = self.freq_ffn(freq_emb)
#         mamba_out = self.mamba_ffn(mamba_emb)
#         time_out  = self.time_ffn(time_emb)

#         # 加权拼接
#         fused = torch.cat((
#             self.alpha_freq  * freq_out,
#             self.alpha_mamba * mamba_out,
#             self.alpha_time  * time_out
#         ), dim=-1)

#         # 降维并规范化
#         out = self.fuse_proj(fused)
#         out = self.LayerNorm(out)
#         return out

class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.LeakyReLU()
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        # Feed-Forward Network
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)

        # residual connection
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
    
  
#     # 频域滤波层
# class FrequencyLayer(nn.Module):
#     def __init__(self, d_model,dropout=0.2):
#         super(FrequencyLayer, self).__init__()
#         self.out_dropout = nn.Dropout(dropout)
#         self.d_model=d_model
#         self.LayerNorm = nn.LayerNorm(self.d_model, eps=1e-12)
#         # 三个通道分别建模
#         self.low_branch = nn.Sequential(
#             nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)
#         )
#         self.mid_branch = nn.Sequential(
#             nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)
#         )
#         self.high_branch = nn.Sequential(
#             nn.Linear(d_model, d_model), Swish(), nn.Dropout(dropout)
#         )
#          # 可学习权重用于融合（通过 softmax 归一化）
#         self.fuse_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
#          # 分频比例：可以调整（默认三等分）
#         # self.low_ratio = 0.2
#         # self.mid_ratio = 0.4
#         self.low_ratio = nn.Parameter(torch.tensor(0.2))
#         self.mid_ratio = nn.Parameter(torch.tensor(0.4))
#         self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3,padding=1)
#         self.gru = nn.GRU(d_model, d_model, num_layers=1, bias=False,batch_first=True)

#     def forward(self, x):
#         if torch.isnan(x).any():
#             x = torch.where(torch.isnan(x), torch.zeros_like(x), x)      


#         #  1. 直接使用1D卷积在特征维度
#         g1 = self.conv1d(x.permute(0, 2, 1))  # 从B,L,D -> B,D,L
#         g1 = g1.permute(0, 2, 1)              # 回B,L,D
#         gru_output, _ = self.gru(g1)

#         B, L, D = x.shape
#         freq = torch.fft.rfft(x, dim=1, norm='ortho')  # [B, F, D]
#         F_size = freq.size(1)
#         l_end = int(F_size * self.low_ratio)
#         m_end = int(F_size * self.mid_ratio)

#         # 构建频域掩码
#         low, mid, high = torch.zeros_like(freq), torch.zeros_like(freq), torch.zeros_like(freq)
#         low[:, :l_end, :] = freq[:, :l_end, :]
#         mid[:, l_end:m_end, :] = freq[:, l_end:m_end, :]
#         high[:, m_end:, :] = freq[:, m_end:, :]

#         # 逆傅里叶变换 -> 时域信号
#         low = torch.fft.irfft(low, n=L, dim=1, norm='ortho')
#         mid = torch.fft.irfft(mid, n=L, dim=1, norm='ortho')
#         high = torch.fft.irfft(high, n=L, dim=1, norm='ortho')
   


#         # 三个通道分别处理
#         low_out = self.low_branch(low)
#         mid_out = self.mid_branch(mid)
#         high_out = self.high_branch(high)

#         # 融合通道
#         weight = torch.softmax(self.fuse_weights, dim=0)
#         fuse = weight[0] * low_out + weight[1] * mid_out + weight[2] * high_out
#         fuse=fuse+gru_output


#         # 残差 & 标准化
#         return fuse  # 返回融合结果 + 高频用于 loss


class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

