"""
cbert_cross.py
────────────────────────────────────────────────────────────────────────────
Full “C-BERT” reference implementation
 • CodeBERT + BERT backbones
 • JOERN token-type embedding
 • Per-layer CLS-query swap (Q-only) for cross-modal interaction
 ────────────────────────────────────────────────────────────────────────────
"""

import torch, math
import torch.nn as nn
from transformers import AutoModel
from config import cfg as global_cfg

# ── mean-pool utility ───────────────────────────────────────────────
def masked_mean_pool(hidden, mask):
    """
    hidden : (B, L, D)
    mask   : (B, L)   ─ 1 for real token, 0 for pad
    returns: (B, D)
    """
    mask = mask.float().unsqueeze(-1)          # (B, L, 1)
    summed = (hidden * mask).sum(dim=1)        # (B, D)
    denom  = mask.sum(dim=1).clamp(min=1e-6)   # avoid div-by-0
    return summed / denom


# ────────────────────────────────────────────────────────────────────────────
# 1.  Self-Attention that **replaces Query of CLS_cross (index 1)** only
# ────────────────────────────────────────────────────────────────────────────
class CustomSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head   = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // self.n_head
        self.inner    = self.n_head * self.head_dim

        self.query = nn.Linear(cfg.hidden_size, self.inner)
        self.key   = nn.Linear(cfg.hidden_size, self.inner)
        self.value = nn.Linear(cfg.hidden_size, self.inner)
        self.dropout = nn.Dropout(cfg.attention_probs_dropout_prob)

    # helper
    def _transpose(self, x):                         # (B,L,E) → (B,H,L,D)
        B, L, _ = x.size()
        return x.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x, attn_mask=None, cross_cls_sent=None):
        """
        x            : (B,L,D) - hidden_states
        cross_cls_sent : (B,D) - 상대 모달 CLS_sent
        """
        q = self.query(x)                            # (B,L,E)
        if cross_cls_sent is not None:               # Q-swap for CLS_cross
            q_cross = self.query(cross_cls_sent).unsqueeze(1)  # (B,1,E)
            q[:, 1:2, :] = q_cross

        k = self.key(x)
        v = self.value(x)

        q, k, v = map(self._transpose, (q, k, v))    # (B,H,L,D)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        probs   = self.dropout(scores.softmax(-1))
        ctx     = torch.matmul(probs, v)             # (B,H,L,D)
        ctx     = ctx.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return ctx                                   # (B,L,D)

# ────────────────────────────────────────────────────────────────────────────
# 2.  Transformer layer wrapper:  Self-Attention(Q-swap) + FFN
# ────────────────────────────────────────────────────────────────────────────
from deepspeed.moe.layer import MoE
from deepspeed.moe.experts import Experts
class FeedForwardExpert(nn.Module):
    # exactly the same hidden sizes as the old FFN
    def __init__(self, hidden_size, intermediate_size, activation):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = activation
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        return self.fc2(self.act(self.fc1(x)))

class CrossLayer(nn.Module):
    def __init__(self, bert_layer, cfg, num_experts = 8, top_k = 2):
        super().__init__()
        # self-attention
        self.self_attn = CustomSelfAttention(cfg)
        self.qout      = bert_layer.attention.output.dense
        self.q_ln      = bert_layer.attention.output.LayerNorm          
        self.dropout   = nn.Dropout(cfg.hidden_dropout_prob)
        
        # FFN or MoE? (cfg.use_moe)
        global global_cfg
        self.use_moe = global_cfg.use_moe
        
        if self.use_moe:
            # build the MoE, replace the dense FFN
            self.moe = MoE (
                hidden_size = cfg.hidden_size,
                expert = FeedForwardExpert(
                    cfg.hidden_size, 
                    cfg.intermediate_size, 
                    activation = nn.functional.gelu
                ),
                num_experts= global_cfg.moe_num_experts,
                k = global_cfg.moe_top_k,
            )        
            # warm-start every expert from the dense FFN
            for exp in self.moe.deepspeed_moe.experts.deepspeed_experts:
                exp.fc1.weight.data.copy_(bert_layer.intermediate.dense.weight)
                exp.fc1.bias.data.copy_(bert_layer.intermediate.dense.bias)
                exp.fc2.weight.data.copy_(bert_layer.output.dense.weight)
                exp.fc2.bias.data.copy_(bert_layer.output.dense.bias)
                    
        else:
            # just use regular FFN
            self.inter     = bert_layer.intermediate
            self.out       = bert_layer.output.dense
            self.act_fn    = self.inter.intermediate_act_fn
    
        self.out_ln = bert_layer.output.LayerNorm
        
        # attention weight copy
        for tgt, src in [
            (self.self_attn.query, bert_layer.attention.self.query),
            (self.self_attn.key,   bert_layer.attention.self.key),
            (self.self_attn.value, bert_layer.attention.self.value),
        ]:
            tgt.weight.data = src.weight.data.clone()
            tgt.bias.data   = src.bias.data.clone()

    def forward(self, x, attn_mask, cross_cls):
        # Self-Attention (with Q-swap)
        attn_out = self.self_attn(x, attn_mask, cross_cls)
        x = self.q_ln(x + self.dropout(self.qout(attn_out)))
        
        # feedforward sublayer
        if self.use_moe:
            ffn_out, moe_loss, _ = self.moe(x)          # (B,L,D)
            x = self.out_ln(x + self.dropout(ffn_out))
        else:
            ffn_out = self.act_fn(self.inter(x))
            x = self.out_ln(x + self.dropout(self.out(ffn_out)))
            moe_loss = 0                  # MoE loss is unused: fill with scalar 0
        return x, moe_loss

def convert_to_cross_encoder(model):
    """Replace every BertLayer with CrossLayer in-place."""
    new_layers = nn.ModuleList()
    for layer in model.encoder.layer:
        new_layers.append(CrossLayer(layer, model.config))
    model.encoder.layer = new_layers
    return model

# ────────────────────────────────────────────────────────────────────────────
# 3.  C-BERT main module
# ────────────────────────────────────────────────────────────────────────────
class CBERT(nn.Module):
    """
    Inputs
    -------
      text_ids, text_mask                 : (B, L_t)
      code_ids, code_type_ids, code_mask  : (B, L_c)
      explicit_feat                       : (B, 4)  (난이도 추정 명시적 특징)
    Note
    ----
      • 시퀀스 앞 2 토큰은 [CLS_sent] [CLS_cross] 용으로 **비워둔 채** 전달
    """
    def __init__(self, num_labels,
                 num_tag,
                 text_model_name="bert-base-uncased",
                 code_model_name="microsoft/codebert-base",
                 dropout=0.0,
                 convert=True):
        super().__init__()
        self.convert = convert
        if convert:
            # backbones → cross-layer 변환
            self.text = convert_to_cross_encoder(AutoModel.from_pretrained(text_model_name))
            self.code = convert_to_cross_encoder(AutoModel.from_pretrained(code_model_name))

        D = self.text.config.hidden_size
        assert D == self.code.config.hidden_size
        #self.type_emb = nn.Embedding(num_code_types, D)

        self.embed_tag = nn.Sequential(
                            nn.Linear(num_tag,num_tag), 
                            nn.ReLU(),
                            nn.Linear(num_tag,16))
                            
        self.classifier = nn.Sequential(
            nn.Linear(D * 2 + 3 + 16, D*2),
            nn.ReLU(),
            nn.Linear(D*2, D),
            nn.ReLU(),
            nn.Linear(D, num_labels),
        )
    
    # helper: get extended mask (B,1,1,L)
    @staticmethod
    def _ext_mask(mask, model):
        return model.get_extended_attention_mask(mask, mask.shape, mask.device)

    def forward(self,
                text_ids, text_mask,
                code_ids, code_mask,
                feats, tags):
        B, Lt = text_ids.shape
        _, Lc = code_ids.shape
        dev   = text_ids.device
        D     = self.text.config.hidden_size
    
        # ── Text Embedding
        t_tok = self.text.embeddings.word_embeddings(text_ids)
        t_pos = self.text.embeddings.position_embeddings(torch.arange(Lt, device=dev))
        t_hid = t_tok + t_pos                               # (B,Lt,D)
    
        # ── Code Embedding  (+ JOERN token-type)
        c_tok = self.code.embeddings.word_embeddings(code_ids)
        c_pos = self.code.embeddings.position_embeddings(torch.arange(Lc, device=dev))
        c_hid = c_tok + c_pos                      # (B,Lc,D)
    
        # 두 모달 모두 앞 2 토큰 CLS 초기화
        cls_vec = self.text.embeddings.word_embeddings.weight[0]
        t_hid[:, :2, :] = cls_vec
        c_hid[:, :2, :] = cls_vec
    
        # mask
        t_mask = self._ext_mask(text_mask, self.text)
        c_mask = self._ext_mask(code_mask,  self.code)
    
        # ───────────────────────────────────────────────
        #  Layer-by-Layer  (Q-swap per layer)
        # ───────────────────────────────────────────────
        total_moe_loss = 0.
        for t_layer, c_layer in zip(self.text.encoder.layer, self.code.encoder.layer):
            # 서로의 CLS_sent(Query) 주입
            #t_hid = t_layer(t_hid, t_mask, cross_cls=None)#torch.zeros_like(c_hid[:, 0, :]).to(dev))
            #c_hid = c_layer(c_hid, c_mask, cross_cls=None)#torch.zeros_like(c_hid[:, 0, :]).to(dev))
            t_hid_buf = t_hid.clone()
            c_hid_buf = c_hid.clone()
            t_hid, t_loss = t_layer(t_hid, t_mask, cross_cls=c_hid_buf[:, 0, :])
            c_hid, c_loss = c_layer(c_hid, c_mask, cross_cls=t_hid_buf[:, 0, :])
            total_moe_loss = total_moe_loss + t_loss + c_loss
        
        txt_vec  = masked_mean_pool(t_hid, text_mask)   # (B, D)
        code_vec = masked_mean_pool(c_hid, code_mask)   # (B, D)
        
        tag_embedding = self.embed_tag(tags)
        # print(txt_vec.shape, code_vec.shape, feats.shape, tag_embedding.shape)
        logits = self.classifier(torch.cat([txt_vec, code_vec, feats, tag_embedding], dim=1))
        return logits, total_moe_loss
        
        
        