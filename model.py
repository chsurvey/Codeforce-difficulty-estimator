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
class FeedForwardExpert(nn.Module):
    # exactly the same hidden sizes as the old FFN
    def __init__(self, hidden_size, intermediate_size, activation):
        super().__init__(hidden_size, intermediate_size, activation)
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
        
        # build the MoE, replace the dense FFN
        self.moe = MoE (
            hidden_size = cfg.hidden_size,
            expert = FeedForwardExpert(
                cfg.hidden_size, 
                cfg.intermediate_size, 
                activation = nn.functional.gelu
            ),
            num_experts= num_experts,
            k = top_k,
        )
        self.out_ln = bert_layer.output.LayerNorm
        
        # # FFN
        # self.inter     = bert_layer.intermediate
        # self.out       = bert_layer.output.dense
        # self.out_ln    = bert_layer.output.LayerNorm
        # self.dropout   = nn.Dropout(cfg.hidden_dropout_prob)
        # self.act_fn    = self.inter.intermediate_act_fn
        
        # (optional) copy pretrained dense weights into every expert
        for exp in self.moe.experts:
            exp.fc1.weight.data.copy_(bert_layer.intermediate.dense.weight)
            exp.fc1.bias.data.copy_(bert_layer.intermediate.dense.bias)
            exp.fc2.weight.data.copy_(bert_layer.output.dense.weight)
            exp.fc2.bias.data.copy_(bert_layer.output.dense.bias)
        
        # weight copy
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
        # # FFN
        # ffn = self.act_fn(self.inter(x))
        # x = self.out_ln(x + self.dropout(self.out(ffn)))
        moe_out, moe_loss, _ = self.moe(x)          # DeepSpeed returns (y, loss, counts)
        x = self.out_ln(x + self.dropout(moe_out))
        return x, moe_loss

def convert_to_cross_encoder(model):
    """Replace every BertLayer with CrossLayer in-place."""
    new_layers = nn.ModuleList()
    for layer in model.encoder.layer:
        new_layers.append(CrossLayer(layer, model.config))
    model.encoder.layer = new_layers
    return model

# ────────────────────────────────────────────────────────────────────────────
model = AutoModel.from_pretrained("bert-base-uncased")
for layer in model.encoder.layer:
    print(type(layer))
# ────────────────────────────────────────────────────────────────────────────
    

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
                 text_model_name="bert-base-uncased",
                 code_model_name="microsoft/codebert-base",
                 num_code_types=20,
                 dropout=0.0):
        super().__init__()
        # backbones → cross-layer 변환
        self.text = convert_to_cross_encoder(AutoModel.from_pretrained(text_model_name))
        self.code = convert_to_cross_encoder(AutoModel.from_pretrained(code_model_name))

        D = self.text.config.hidden_size
        assert D == self.code.config.hidden_size
        self.type_emb = nn.Embedding(num_code_types, D)

        self.classifier = nn.Sequential(
            nn.Linear(D * 2 + 4, D),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(D, num_labels),
        )

    # helper: get extended mask (B,1,1,L)
    @staticmethod
    def _ext_mask(mask, model):
        return model.get_extended_attention_mask(mask, mask.shape, mask.device)

    def forward(self,
                text_ids, text_mask,
                code_ids, code_type_ids, code_mask,
                explicit_feat):
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
        c_type= self.type_emb(code_type_ids)
        c_hid = c_tok + c_pos + c_type                      # (B,Lc,D)

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
        for t_layer, c_layer in zip(self.text.encoder.layer, self.code.encoder.layer):
            # 서로의 CLS_sent(Query) 주입
            t_hid = t_layer(t_hid, t_mask, cross_cls=c_hid[:, 0, :])
            c_hid = c_layer(c_hid, c_mask, cross_cls=t_hid[:, 0, :])

        # final CLS_sent (index 0)
        t_cls, c_cls = t_hid[:, 0], c_hid[:, 0]
        logits = self.classifier(torch.cat([t_cls, c_cls, explicit_feat], dim=-1))
        return logits