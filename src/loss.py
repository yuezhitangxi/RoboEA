import torch
from torch import nn

from models import *
from utils import *
import torch.nn.functional as F


class CustomMultiLossLayer(nn.Module):
    def __init__(self, loss_num, device=None):
        super(CustomMultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(
            torch.zeros(
                self.loss_num,
            ),
            requires_grad=True,
        )

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss


class InfoNCE_loss(nn.Module):
    def __init__(self, device, temperature=0.05) -> None:
        super().__init__()
        self.device = device
        self.t = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def sim(self, emb_left, emb_right):
        return emb_left.mm(emb_right.t())

    def forward(self, emb, train_links):
        emb = F.normalize(emb)
        emb_train_left = emb[train_links[:, 0]]
        emb_train_right = emb[train_links[:, 1]]
        score = self.sim(emb_train_left, emb_train_right)
        bsize = emb_train_left.size()[0]
        label = torch.arange(bsize, dtype=torch.long).cuda(self.device)
        loss = self.ce_loss(score / self.t, label)
        return loss




def volume_computation4(structure, visual, attribute, relation, K=10):

    
    device = structure.device
    B = structure.shape[0]
    similarity = structure @ visual.T

    topk_val, topk_idx = torch.topk(similarity, K, dim=1)

    cur_struct = structure.unsqueeze(1).expand(-1, K, -1)
    cur_attr = attribute.unsqueeze(1).expand(-1, K, -1)
    cur_rela = relation.unsqueeze(1).expand(-1, K, -1)

    cur_visuals = visual[topk_idx]
    cur_attrs = attribute[topk_idx]
    cur_relas = relation[topk_idx]

    all_modals = torch.stack([cur_struct, cur_visuals, cur_attrs, cur_relas], dim=2)

    G = torch.einsum('bkif,bkjf->bkij', all_modals, all_modals)
    gram_det = torch.linalg.det(G)
    vol = torch.sqrt(torch.clamp(torch.abs(gram_det), min=1e-8))

    return topk_idx, vol

class GRAM_Loss(nn.Module):
    def __init__(self, temperature=0.05, K=10):
        super().__init__()
        self.temperature = temperature
        self.K = K

    def forward(self, struct, visual, attr, rel, train_links):
        
        struct_left  = F.normalize(struct[train_links[:, 0]], dim=1)
        visual_left  = F.normalize(visual[train_links[:, 0]], dim=1)
        attr_left    = F.normalize(attr[train_links[:, 0]], dim=1)
        rel_left     = F.normalize(rel[train_links[:, 0]], dim=1)

        struct_right = F.normalize(struct[train_links[:, 1]], dim=1)
        visual_right = F.normalize(visual[train_links[:, 1]], dim=1)
        attr_right   = F.normalize(attr[train_links[:, 1]], dim=1)
        rel_right    = F.normalize(rel[train_links[:, 1]], dim=1)

        B = train_links.shape[0]
        topk_idx, topk_volumes = volume_computation4(struct_left, visual_right, attr_right, rel_right, K=self.K)
        targets = torch.arange(B, device=struct.device)
        loss1 = sparse_gram_loss(topk_idx, topk_volumes, targets, temperature=self.temperature)
       
        loss2=0


        return (loss1 + loss2) / 2

def sparse_gram_loss(topk_idx, volumes, targets, temperature=0.05):

    device = targets.device
    B, K = topk_idx.shape

    targets_exp = targets.unsqueeze(1).expand(-1, K)
    mask = (topk_idx == targets_exp)
    has_pos = mask.any(dim=1)

    valid_idx = has_pos.nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        raise ValueError("ERROR！")

    valid_volumes = volumes[valid_idx]
    valid_mask = mask[valid_idx]
    valid_pos_idx = valid_mask.float().argmax(dim=1)

    valid_volumes = valid_volumes / temperature
    log_probs = F.log_softmax(-valid_volumes, dim=1)
    loss = -log_probs[torch.arange(valid_volumes.size(0), device=device), valid_pos_idx].mean()

    return loss


