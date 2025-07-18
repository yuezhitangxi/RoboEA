import torch
from torch import nn
from torch.nn import functional as F
import math
import dgl
import dgl.function as fn

from transformers import apply_chunking_to_forward
from transformers.activations import ACT2FN

class MCD(nn.Module):
    CACHE_KEY = "MCD_weight"

    def __init__(self, k, alpha, beta, x_drop_rate, edge_drop_rate, z_drop_rate):
        super().__init__()

        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = torch.tensor(self.compute_gamma(alpha, beta, k)).float()
        self.gammas = [torch.tensor(self.compute_gamma(alpha, beta, i)).float() for i in range(1, k + 1)]

        self.x_dropout = nn.Dropout(x_drop_rate)
        self.edge_dropout = nn.Dropout(edge_drop_rate)
        self.z_dropout = nn.Dropout(z_drop_rate)

    @staticmethod
    def compute_gamma(alpha, beta, k):
        return beta ** k + alpha * sum(beta ** i for i in range(k))

    @classmethod
    def build_graph(cls, adj):
        src, dst = adj.nonzero(as_tuple=True)
        g = dgl.graph((src, dst), num_nodes=adj.shape[0])
        g = g.to('cpu')
        g = dgl.add_self_loop(g)
        g = g.to('cuda')
        return g

    @classmethod
    @torch.no_grad()
    def norm_adj(cls, g):
        degs = g.in_degrees().float().pow(-0.5)
        g.ndata["norm"] = degs
        g.apply_edges(fn.u_mul_v("norm", "norm", cls.CACHE_KEY))

    def forward(self, g, x_dict, return_all=False):
        self.norm_adj(g)
        edge_weight = g.edata[self.CACHE_KEY]
        dropped_edge_weight = self.edge_dropout(edge_weight)

        out_dict = {}
        for modality, x in x_dict.items():
            if x is None:
                out_dict[modality] = None
                continue

            h0 = self.x_dropout(x)
            h = h0

            if return_all:
                h_list = []

            with g.local_scope():
                g.edata[self.CACHE_KEY] = dropped_edge_weight

                for _ in range(self.k):
                    g.ndata["h"] = h
                    g.update_all(fn.u_mul_e("h", self.CACHE_KEY, "m"), fn.sum("m", "h"))
                    h = g.ndata.pop("h")
                    h = h * self.beta + h0 * self.alpha

                    if return_all:
                        h_list.append(h)

            if not return_all:
                h = h / self.gamma
                h = self.z_dropout(h)
                out_dict[modality] = h
            else:
                h_list = [h / gamma for h, gamma in zip(h_list, self.gammas)]
                h_list = [self.z_dropout(h) for h in h_list]
                out_dict[modality] = h_list

        return out_dict

class MFusion(nn.Module):
    def __init__(self, args, modal_num, with_weight=1):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        self.type_id = torch.tensor([0, 1, 2, 3, 4, 5]).cuda()
        

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fusion_weights)
        nn.init.zeros_(self.fusion_bias)
        
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

        batch_size = hidden_states.shape[0]
        modal_num = hidden_states.shape[1]
        dim = hidden_states.shape[2]
        hidden_states_reshaped = hidden_states.view(batch_size, modal_num * dim)
        

        return hidden_states_reshaped

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