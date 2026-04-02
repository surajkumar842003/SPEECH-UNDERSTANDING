
import torch
import torch.nn.functional as F


def reconstruction_loss(e, e_hat):
    return F.l1_loss(e_hat, e.detach())


def triplet_env_loss(g1, g2, g3, margin=0.2):
    g1 = F.normalize(g1, dim=1)
    g2 = F.normalize(g2, dim=1)
    g3 = F.normalize(g3, dim=1)
    pos = (g1 * g2).sum(dim=1)
    neg = (g1 * g3).sum(dim=1)
    return F.relu(neg - pos + margin).mean()


def mapc_loss(e_spk, e_env):
    B = e_spk.shape[0]
    if B < 2:
        return e_spk.sum() * 0.0
    s = e_spk - e_spk.mean(dim=0, keepdim=True)
    v = e_env - e_env.mean(dim=0, keepdim=True)
    s = s / (s.norm(dim=0, keepdim=True) + 1e-8)
    v = v / (v.norm(dim=0, keepdim=True) + 1e-8)
    return ((s.T @ v) / B).abs().mean()


def contrastive_speaker_loss(emb_list, labels, temperature=0.1):
    
    embs = torch.cat(emb_list, dim=0).float()
    embs = F.normalize(embs, dim=1)

    B          = labels.shape[0]
    n          = len(emb_list)
    N          = n * B
    all_labels = labels.repeat(n)
    temperature = max(float(temperature), 0.07)

    sim  = (embs @ embs.T) / temperature      

    eye  = torch.eye(N, dtype=torch.bool, device=embs.device)
    sim.masked_fill_(eye, -1e4)               
    pos_mask = (all_labels.unsqueeze(0) == all_labels.unsqueeze(1))
    pos_mask.masked_fill_(eye, False)

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=embs.device, requires_grad=True)

    log_prob = F.log_softmax(sim, dim=1)

    loss = -(log_prob * pos_mask.float()).sum() / pos_mask.float().sum()
    if torch.isnan(loss):
        return torch.tensor(0.0, device=embs.device, requires_grad=True)
    return loss


def total_disentangled_loss(loss_spk, loss_r, loss_ee, loss_adv, loss_corr,
                             loss_contrast=None,
                             lam_spk=1.0, lam_r=1.0, lam_ee=1.0,
                             lam_adv=0.3, lam_c=0.5, lam_contrast=0.3):
    total = (lam_spk * loss_spk +
             lam_r   * loss_r   +
             lam_ee  * loss_ee  +
             lam_adv * loss_adv +
             lam_c   * loss_corr)
    if loss_contrast is not None:
        total = total + lam_contrast * loss_contrast
    return total
