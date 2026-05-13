import torch
from torch import nn
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
    """Bidirectional InfoNCE with optional temperature."""
    def __init__(self, device=None, temperature=0.05, bidirectional=False) -> None:
        super().__init__()
        self.t = temperature
        self.bidirectional = bidirectional
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, emb, train_links):
        emb = F.normalize(emb, dim=-1)
        if not torch.is_tensor(train_links):
            train_links = torch.LongTensor(train_links).to(emb.device)
        else:
            train_links = train_links.to(emb.device)
        left = emb[train_links[:, 0]]
        right = emb[train_links[:, 1]]
        score = left.mm(right.t())
        label = torch.arange(score.size(0), dtype=torch.long, device=score.device)
        loss_l2r = self.ce_loss(score / self.t, label)
        if self.bidirectional:
            loss_r2l = self.ce_loss(score.t() / self.t, label)
            loss = 0.5 * (loss_l2r + loss_r2l)
        else:
            loss = loss_l2r
        return loss


class CosFaceMarginLoss(nn.Module):
    """CosFace-style margin contrastive loss with focal weighting,
    temperature scheduling, and hard negative repulsion."""
    def __init__(self, temperature=0.05, margin=0.15, scale=16.0,
                 hard_neg_topk=10, hard_neg_weight=0.05, hard_neg_margin=0.2,
                 focal_gamma=2.0, t_max=0.15, bidirectional=True):
        super().__init__()
        self.t_min = temperature
        self.t_max = t_max
        self.margin = margin
        self.scale = scale
        self.hard_neg_topk = hard_neg_topk
        self.hard_neg_weight = hard_neg_weight
        self.hard_neg_margin = hard_neg_margin
        self.focal_gamma = focal_gamma
        self.bidirectional = bidirectional
        self.structure_loss_weight = 0.03
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def _structure_margins(self, gph_emb, train_links, margin_eff):
        if gph_emb is None:
            return None, None

        gph_emb = F.normalize(gph_emb, dim=-1)
        left_g = gph_emb[train_links[:, 0]]
        right_g = gph_emb[train_links[:, 1]]
        struct_sim = left_g.mm(right_g.t()).clamp(-1.0, 1.0)
        struct_affinity = (struct_sim + 1.0) * 0.5
        struct_strength = (margin_eff / max(self.margin, 1e-8)) ** 2

        pos_margin = margin_eff * (0.9 + 0.2 * struct_affinity.diag())
        row_base = struct_affinity.mean(dim=1, keepdim=True)
        neg_margin = 0.15 * margin_eff * struct_strength * F.relu(struct_affinity - row_base)
        eye = torch.eye(struct_affinity.size(0), dtype=torch.bool, device=struct_affinity.device)
        neg_margin = neg_margin.masked_fill(eye, 0.0)
        return pos_margin, neg_margin

    def _one_side(self, score, B, margin_eff, t_eff, pos_margin=None, neg_margin=None):
        label = torch.arange(B, dtype=torch.long, device=score.device)
        if pos_margin is None:
            pos_margin = score.new_full((B,), margin_eff)
        score_m = score.clone()
        score_m[label, label] = score_m[label, label] - pos_margin
        logits = self.scale * score_m / t_eff
        loss_per_sample = self.ce_loss(logits, label)
        if self.focal_gamma > 0:
            probs = F.softmax(logits, dim=1)
            p_pos = probs[range(B), label].detach()
            focal_weight = (1.0 - p_pos) ** self.focal_gamma
            loss_ce = (focal_weight * loss_per_sample).mean()
        else:
            loss_ce = loss_per_sample.mean()

        eye = torch.eye(B, dtype=torch.bool, device=score.device)
        hard_score = score
        if neg_margin is not None:
            hard_score = hard_score + neg_margin
        neg = hard_score.masked_fill(eye, -1e9)
        pos = score.diag()
        k = min(self.hard_neg_topk, B - 1)
        hard, hard_idx = torch.topk(neg, k=k, dim=1)
        hm_scale = torch.clamp(pos_margin / max(self.margin, 1e-8), max=1.0)
        hm = self.hard_neg_margin * hm_scale.unsqueeze(1)
        if neg_margin is not None:
            hard_struct = torch.gather(neg_margin, 1, hard_idx)
            hm = hm * (1.0 + torch.clamp(hard_struct / max(margin_eff, 1e-8), max=1.0))
        loss_hard = F.softplus(
            (hard - pos.unsqueeze(1) + hm) * 10.0
        ).mean()
        return loss_ce, loss_hard

    def forward(self, emb, train_links, epoch_ratio=1.0, gph_emb=None):
        emb = F.normalize(emb, dim=-1)
        if not torch.is_tensor(train_links):
            train_links = torch.LongTensor(train_links).to(emb.device)
        else:
            train_links = train_links.to(emb.device)
        left = emb[train_links[:, 0]]
        right = emb[train_links[:, 1]]
        B = left.size(0)
        if B <= 1:
            return left.sum() * 0.0, left.sum() * 0.0, left.sum() * 0.0

        margin_eff = self.margin * min(epoch_ratio, 1.0)
        t_eff = self.t_max - (self.t_max - self.t_min) * min(epoch_ratio, 1.0)
        ratio = min(epoch_ratio, 1.0)
        score = left.mm(right.t())
        pos_margin, neg_margin = self._structure_margins(gph_emb, train_links, margin_eff)

        loss_ce_l2r, loss_hard_l2r = self._one_side(
            score, B, margin_eff, t_eff, pos_margin=pos_margin, neg_margin=neg_margin
        )
        if self.bidirectional:
            pos_margin_r = pos_margin
            neg_margin_r = None if neg_margin is None else neg_margin.t()
            loss_ce_r2l, loss_hard_r2l = self._one_side(
                score.t(), B, margin_eff, t_eff,
                pos_margin=pos_margin_r, neg_margin=neg_margin_r
            )
            loss_ce = 0.5 * (loss_ce_l2r + loss_ce_r2l)
            loss_hard = 0.5 * (loss_hard_l2r + loss_hard_r2l)
        else:
            loss_ce = loss_ce_l2r
            loss_hard = loss_hard_l2r

        total = loss_ce + self.hard_neg_weight * loss_hard
        if gph_emb is not None and self.structure_loss_weight > 0:
            gph_norm = F.normalize(gph_emb, dim=-1)
            left_g = gph_norm[train_links[:, 0]]
            right_g = gph_norm[train_links[:, 1]]
            struct_score = left_g.mm(right_g.t())
            struct_ce_l2r, _ = self._one_side(
                struct_score, B, 0.5 * margin_eff, t_eff,
                pos_margin=None, neg_margin=None
            )
            if self.bidirectional:
                struct_ce_r2l, _ = self._one_side(
                    struct_score.t(), B, 0.5 * margin_eff, t_eff,
                    pos_margin=None, neg_margin=None
                )
                struct_ce = 0.5 * (struct_ce_l2r + struct_ce_r2l)
            else:
                struct_ce = struct_ce_l2r
            total = total + self.structure_loss_weight * (ratio ** 2) * struct_ce
        return total, loss_ce.detach(), loss_hard.detach()
