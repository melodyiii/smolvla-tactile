import torch
import torch.nn as nn
import torch.nn.functional as F

class TactileEncoder(nn.Module):
    def __init__(self, proj_dim=512, hid=128, d_model=256, tau=0.07):
        super().__init__()
        # CNN 保持不变: 提取每一帧的空间特征
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3,1,1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128,hid,3,1,1), nn.BatchNorm2d(hid), nn.GELU(),
        )
        
        # GRU 保持不变: 处理时序
        self.gru = nn.GRU(hid*16*16, d_model, batch_first=True)
        
        # Proj 保持不变: 映射到 CLIP 维度
        # 注意: Linear 和 LayerNorm 支持 [B, T, d_model] 输入，会自动处理最后一维
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, proj_dim))
        
        # 温度系数
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0/tau)))

    def forward(self, x, return_seq=False, return_feat_map=False, return_spatial_seq=False): 
        """
        x: [B, T, 16, 16]
        return_seq:         Stage 2/3 需要序列特征用于 Attention
        return_feat_map:    返回最后一帧 CNN 特征图 [B, hid, 16, 16]
        return_spatial_seq: 返回每一帧 CNN 特征图 [B, T, hid, 16, 16]
        """
        B, T, H, W = x.shape
       
        # 1. CNN 提取空间特征
        x = x.view(B*T, 1, H, W)                 # [B*T, 1, 16, 16]
        f_map = self.cnn(x)                      # [B*T, hid, 16, 16]  ← 中间特征图
        f = f_map.flatten(1)                     # [B*T, hid*16*16]
        f = f.view(B, T, -1)                     # [B, T, hid*16*16]

        # 2. GRU 提取时序特征
        seq_out, h = self.gru(f)                 
        # seq_out: [B, T, d_model]
        # h: [1, B, d_model]

        # 3. 全局特征
        z_global = F.normalize(self.proj(h[-1]), dim=-1)  # [B, 512]
        s = self.logit_scale.exp().clamp(1e-3, 1e3)

        # 4. 空间特征（last frame & full sequence）
        spatial_seq = f_map.view(B, T, *f_map.shape[1:])  # [B,T,hid,16,16]
        feat_map = spatial_seq[:, -1, :, :, :]            # [B,hid,16,16]

        if return_seq:
            z_seq = F.normalize(self.proj(seq_out), dim=-1)  # [B, T, 512]
            if return_feat_map and return_spatial_seq:
                return z_seq, z_global, s, feat_map, spatial_seq
            if return_feat_map:
                return z_seq, z_global, s, feat_map
            if return_spatial_seq:
                return z_seq, z_global, s, spatial_seq
            return z_seq, z_global, s
        else:
            if return_feat_map and return_spatial_seq:
                return z_global, s, feat_map, spatial_seq
            if return_feat_map:
                return z_global, s, feat_map
            if return_spatial_seq:
                return z_global, s, spatial_seq
            return z_global, s


def improved_multi_pos_infonce(z_q, pos_list, all_keys, logit_scale, method="weighted"):
   
    if method == "simple":
        logits_all = (z_q @ all_keys.t()) * logit_scale
       
        pos_logits = torch.stack([(z_q * p).sum(-1)*logit_scale for p in pos_list], dim=1)
        return (torch.logsumexp(logits_all,1) - torch.logsumexp(pos_logits,1)).mean()
    
    elif method == "weighted":
      
        logits_all = (z_q @ all_keys.t()) * logit_scale
        
        losses = []
        for i, pos_emb in enumerate(pos_list):
            # 计算每个正例的相似度
            # 确保 pos_emb 是 [B, Dim]
            pos_sim = (z_q * pos_emb).sum(-1) * logit_scale
            
            # 创建目标：当前正例为正样本，其他为负样本
            exp_pos = torch.exp(pos_sim)
            #  包含了正样本自身，通常 InfoNCE 分母包含正样本
            exp_neg = torch.exp(logits_all).sum(1) - exp_pos
            
            # 加权：第一个正例（完整句子）权重更高
            weight = 1.0 if i == 0 else 0.7
            # log(pos / (pos + neg))
            loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8)) * weight
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    elif method == "hard":
        # 困难负例挖掘版本
        logits_all = (z_q @ all_keys.t()) * logit_scale
        
        with torch.no_grad():
            pos_mask = torch.zeros_like(logits_all, dtype=torch.bool)
            for pos_emb in pos_list:
                # 这里的 == 比较可能需要 float 的近似比较，或者直接依赖索引
                # 如果 all_keys 很大，这个操作比较慢
                pos_indices = (all_keys.unsqueeze(0) == pos_emb.unsqueeze(1)).all(-1)
                pos_mask |= pos_indices
            
            # 负例的logits
            neg_logits = logits_all.clone()
            neg_logits[pos_mask] = -1e9 
            hard_neg_idx = neg_logits.argmax(1)  
        
        losses = []
        for pos_emb in pos_list:
            pos_sim = (z_q * pos_emb).sum(-1) * logit_scale
            hard_neg_sim = logits_all[torch.arange(z_q.size(0)), hard_neg_idx]
            
            # 使用最难负例计算损失
            loss = F.cross_entropy(
                torch.stack([pos_sim, hard_neg_sim], dim=1), 
                torch.zeros(z_q.size(0), dtype=torch.long, device=z_q.device)
            )
            losses.append(loss)
        
        return torch.stack(losses).mean()