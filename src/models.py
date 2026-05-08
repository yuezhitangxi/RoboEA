from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Mfusion import *


class RRGAT(nn.Module):
    def __init__(
        self, args, node_size, rel_size, triple_size, node_dim, depth=1
    ):
        super(RRGAT, self).__init__()

        self.args=args
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.activation = torch.nn.Tanh()
        self.depth = depth
        self.attn_kernels = nn.ParameterList()

        for l in range(self.depth):
            attn_kernel = torch.nn.Parameter(
                data=torch.empty(self.node_dim, 1, dtype=torch.float32)
            )
            nn.init.xavier_uniform_(attn_kernel)
            self.attn_kernels.append(attn_kernel)
        self.modal_num = 3

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj = inputs[2]
        r_index = inputs[3]
        r_val = inputs[4]

        features = self.activation(features)
        outputs.append(features)

        layer_outputs = []
        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
            tri_rel = torch.sparse_coo_tensor(
                indices=r_index,
                values=r_val,
                size=[self.triple_size, self.rel_size],
                dtype=torch.float32,
            )
            tri_rel = torch.sparse.mm(tri_rel, rel_emb)
            neighs = features[adj[1, :].long()]
            tri_rel = F.normalize(tri_rel, dim=1, p=2)

            # neighs = (
            #     neighs - 2 * torch.sum(neighs * tri_rel, dim=1, keepdim=True) * tri_rel
            # )

            att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)
            att = torch.sparse_coo_tensor(
                indices=adj, values=att, size=[self.node_size, self.node_size]
            )
            att = torch.sparse.softmax(att, dim=1)

            _src = neighs * torch.unsqueeze(att.coalesce().values(), dim=-1)
            _idx = adj[0, :].long()
            new_features = torch.zeros(self.node_size, _src.shape[-1], device=_src.device, dtype=_src.dtype)
            new_features.scatter_add_(0, _idx.unsqueeze(-1).expand_as(_src), _src)
            features = self.activation(new_features)
            outputs.append(features)
        # outputs = torch.cat(outputs, dim=-1)
        # final_outputs = outputs
        # return final_outputs
        return torch.cat(outputs, dim=-1)

class MultiModalEncoderMrFusion(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units

    """

    def __init__(
        self,
        args,
        ent_num,
        rel_size,
        triple_size,
        img_feature_dim,
        adj_matrix,
        r_index,
        r_val,
        rel_matrix,
        ent_matrix,
    ):
        super(MultiModalEncoderMrFusion, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        self.ENT_NUM = ent_num

        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(1000, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)

        if "Dualmodal" in self.args.structure_encoder:
            self.node_hidden = self.input_dim
            self.rel_hidden = self.input_dim
            self.node_size = ent_num
            self.rel_size = rel_size
            self.triple_size = triple_size
            self.depth = 2
            self.adj_list = adj_matrix
            self.r_index = r_index
            self.r_val = r_val
            self.rel_adj = rel_matrix
            self.ent_adj = ent_matrix

            self.final_fusion_type = self.args.final_fusion_type

            self.ent_embedding = nn.Embedding(self.node_size, self.node_hidden)
            self.rel_embedding = nn.Embedding(self.rel_size, self.rel_hidden)
            nn.init.xavier_uniform_(self.ent_embedding.weight)
            nn.init.xavier_uniform_(self.rel_embedding.weight)

            self.e_encoder = RRGAT(
                args=self.args,
                node_size=self.node_size,
                rel_size=self.rel_size,
                triple_size=self.triple_size,
                node_dim=self.node_hidden,
                depth=self.depth,
            )
        self.fusion1 = MFusion(args, modal_num=3)


    def avg(self, adj, emb, size: int):
        adj = torch.sparse_coo_tensor(
            indices=adj,
            values=torch.ones_like(adj[0, :], dtype=torch.float32),
            size=[self.node_size, size],
        )
        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)

    def forward(
        self,
        input_idx,
        adj,
        adj2,
        img_features=None,
        rel_features=None,
        att_features=None,
        name_features=None,
        char_features=None,
    ):

        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None


        if self.args.w_gcn:
            if self.args.structure_encoder == "Dualmodal-joint-LMF":
                joint_emb = []
                if img_emb is not None:
                    joint_emb.append(img_emb)

                if rel_emb is not None:
                    joint_emb.append(rel_emb)
    
                if att_emb is not None:
                    joint_emb.append(att_emb)

                ent_feature = self.avg(self.ent_adj, self.ent_embedding.weight, self.node_size)
                opt = [self.rel_embedding.weight, self.adj_list, self.r_index, self.r_val]
                gph_emb1=ent_feature
                gph_emb = self.e_encoder([ent_feature] + opt)

                if self.final_fusion_type == -1:
                    joint_emb = gph_emb
                else:
                    joint_emb = self.fusion1(joint_emb)
                    joint_emb = torch.cat([gph_emb, joint_emb], dim=-1)
        else:
            gph_emb = None
        name_emb, char_emb = None, None
        return gph_emb1, gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb
