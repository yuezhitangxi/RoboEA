import torch
from torch import nn
from torch.nn import functional as F
import math

from transformers import apply_chunking_to_forward
from transformers.activations import ACT2FN

class MFusion(nn.Module):
    def __init__(self, args, modal_num):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        
    def forward(self, embs):
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        modal_num = len(embs)

        hidden_states = torch.stack(embs, dim=1)
        bs = hidden_states.shape[0]
        for i, layer_module in enumerate(self.fusion_layer):
            layer_outputs = layer_module(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]
        attention_pro = torch.sum(layer_outputs[1], dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        weight_norm = F.softmax(attention_pro_comb, dim=-1)
        embs = [weight_norm[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num)]
        joint_emb = torch.cat(embs, dim=1)
        
        
        return joint_emb

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)

        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        if self.config.use_intermediate:
            self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states,
            output_attentions=output_attentions,
        )
        if not self.config.use_intermediate:
            return (self_attention_outputs[0], self_attention_outputs[1])

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output, outputs)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class StructureGuidedFusion(nn.Module):
    def __init__(self, graph_dim, fused_modal_dim, dropout=0.1):
        super().__init__()
        self.modal_num = 3
        self.modal_dim = fused_modal_dim // self.modal_num
        self.graph_to_modal = nn.Linear(graph_dim, self.modal_dim)
        self.reliability_mlp = nn.Sequential(
            nn.Linear(self.modal_dim * 4, self.modal_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.modal_dim, 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(graph_dim + fused_modal_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, fused_modal_dim),
        )
        self.reliability_beta = 0.35

    def _reliability_posterior(self, g_norm, modal_embs):
        g_modal = F.normalize(self.graph_to_modal(g_norm), dim=-1)
        reliability_logits = []
        for modal_emb in modal_embs:
            m_modal = F.normalize(modal_emb, dim=-1)
            reliability_feat = torch.cat(
                [g_modal, m_modal, torch.abs(g_modal - m_modal), g_modal * m_modal],
                dim=-1,
            )
            reliability_logits.append(self.reliability_mlp(reliability_feat))
        return F.softmax(torch.cat(reliability_logits, dim=1), dim=1)

    def _apply_reliability_scaling(self, modal_fused, reliability):
        modal_num = reliability.size(1)
        chunks = torch.chunk(modal_fused, modal_num, dim=-1)
        weighted_chunks = []
        for idx, chunk in enumerate(chunks):
            scale = 1.0 + self.reliability_beta * (modal_num * reliability[:, idx:idx + 1] - 1.0)
            weighted_chunks.append(chunk * scale)
        return torch.cat(weighted_chunks, dim=-1)

    def forward(self, gph_emb, modal_fused, modal_embs=None):
        g_norm = F.normalize(gph_emb, dim=-1)
        if modal_embs is not None:
            modal_embs = [emb for emb in modal_embs if emb is not None]
            modal_num = len(modal_embs)
            if modal_num > 0 and modal_fused.size(1) == modal_num * self.modal_dim:
                reliability = self._reliability_posterior(g_norm, modal_embs)
                modal_fused = self._apply_reliability_scaling(modal_fused, reliability)

        m_norm = F.normalize(modal_fused, dim=-1)
        gate_raw = self.gate_mlp(torch.cat([g_norm, m_norm], dim=-1))
        gate = 1.0 + 0.08 * torch.tanh(gate_raw)
        joint_emb = torch.cat([gph_emb, gate * modal_fused], dim=-1)
        return joint_emb
