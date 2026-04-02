import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None

def grad_reverse(x, alpha=1.0):
    return GradReversalFn.apply(x, alpha)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid())

    def forward(self, x):
        s = x.mean(dim=2)
        return x * self.fc(s).unsqueeze(2)


class Res2Conv1d(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, scale=8):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size,
                      dilation=dilation,
                      padding=(kernel_size - 1) * dilation // 2)
            for _ in range(scale - 1)])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.width) for _ in range(scale - 1)])

    def forward(self, x):
        chunks = torch.chunk(x, self.scale, dim=1)
        out, sp = [], None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            sp = chunks[i] if i == 0 else sp + chunks[i]
            sp = F.relu(bn(conv(sp)))
            out.append(sp)
        out.append(chunks[-1])
        return torch.cat(out, dim=1)


class SERes2Block(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, scale=8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.bn1   = nn.BatchNorm1d(channels)
        self.res2  = Res2Conv1d(channels, kernel_size, dilation, scale)
        self.bn2   = nn.BatchNorm1d(channels)
        self.conv3 = nn.Conv1d(channels, channels, 1)
        self.bn3   = nn.BatchNorm1d(channels)
        self.se    = SEBlock(channels)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.res2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.se(x) + r


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck, 1),
            nn.ReLU(),
            nn.Conv1d(bottleneck, in_dim, 1),
            nn.Softmax(dim=2))

    def forward(self, x):
        w  = self.attn(x)
        mu = (x * w).sum(dim=2)
        sg = ((x * x * w).sum(dim=2) - mu ** 2).clamp(min=1e-8).sqrt()
        return torch.cat([mu, sg], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, emb_dim=192):
        super().__init__()
        self.layer1   = nn.Conv1d(in_channels, channels, 5, padding=2)
        self.bn1      = nn.BatchNorm1d(channels)
        self.layer2   = SERes2Block(channels, kernel_size=3, dilation=2)
        self.layer3   = SERes2Block(channels, kernel_size=3, dilation=3)
        self.layer4   = SERes2Block(channels, kernel_size=3, dilation=4)
        self.cat_conv = nn.Conv1d(channels * 3, channels * 3, 1)
        self.bn_cat   = nn.BatchNorm1d(channels * 3)
        self.pool     = AttentiveStatsPool(channels * 3, bottleneck=128)
        self.bn_pool  = nn.BatchNorm1d(channels * 6)
        self.fc       = nn.Linear(channels * 6, emb_dim)
        self.bn_emb   = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x  = F.relu(self.bn1(self.layer1(x)))
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        x  = F.relu(self.bn_cat(self.cat_conv(
            torch.cat([x1, x2, x3], dim=1))))
        x  = self.bn_pool(self.pool(x))
        return self.bn_emb(self.fc(x))


class AAMSoftmax(nn.Module):
    def __init__(self, emb_dim, num_classes, margin=0.2, scale=30):
        super().__init__()
        self.m  = margin
        self.s  = scale
        self.W  = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.W)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        cos     = F.normalize(x, dim=1) @ F.normalize(self.W, dim=1).T
        one_hot = torch.zeros_like(cos).scatter_(1, labels.view(-1,1), 1.0)
        return self.ce(self.s * (cos - one_hot * self.m), labels)


class DisentanglerAE(nn.Module):
    def __init__(self, emb_dim, spk_dim, env_dim):
        super().__init__()
        self.enc_spk = nn.Sequential(
            nn.Linear(emb_dim, spk_dim),
            nn.BatchNorm1d(spk_dim))
        self.enc_env = nn.Sequential(
            nn.Linear(emb_dim, env_dim),
            nn.BatchNorm1d(env_dim))
        self.decoder = nn.Sequential(
            nn.Linear(spk_dim + env_dim, emb_dim),
            nn.BatchNorm1d(emb_dim))

    def forward(self, e):
        e_s = self.enc_spk(e)
        e_e = self.enc_env(e)
        e_r = self.decoder(torch.cat([e_s, e_e], dim=1))
        return e_s, e_e, e_r


class EnvDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


class ProjectionHead(nn.Module):
    
    def __init__(self, spk_dim, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spk_dim, spk_dim),
            nn.BatchNorm1d(spk_dim),
            nn.ReLU(),
            nn.Linear(spk_dim, proj_dim))

    def forward(self, x):
        return self.net(x)


class BaselineSpeakerModel(nn.Module):
    def __init__(self, cfg, num_speakers):
        super().__init__()
        m          = cfg["model"]
        channels   = m.get("channels",      512)
        emb_dim    = m.get("embedding_dim", 192)
        margin     = cfg["loss"].get("aam_margin",
                     cfg["loss"].get("margin",  0.2))
        scale      = cfg["loss"].get("aam_scale",
                     cfg["loss"].get("scale",   30))
        self.encoder    = ECAPA_TDNN(
            in_channels=cfg["data"]["n_mels"],
            channels=channels, emb_dim=emb_dim)
        self.classifier = AAMSoftmax(
            emb_dim=emb_dim, num_classes=num_speakers,
            margin=margin, scale=scale)

    def forward(self, mel):
        return F.normalize(self.encoder(mel), dim=1)

    def compute_loss(self, mel, labels):
        emb = self.encoder(mel)
        return self.classifier(emb, labels), emb


class DisentangledSpeakerModel(nn.Module):
    
    def __init__(self, cfg, num_speakers, use_proj=False):
        super().__init__()
        m        = cfg["model"]
        lc       = cfg["loss"]

        emb_dim    = m.get("embedding_dim", 192)
        spk_dim    = m.get("spk_dim",       128)
        env_dim    = m.get("env_dim",       128)
        channels   = m.get("channels",      512)
        disc_hid   = m.get("disc_hidden",   256)
        disc_out   = m.get("disc_out",      128)
        proj_dim   = m.get("proj_dim",       64)

        aam_margin = lc.get("aam_margin", lc.get("margin", 0.2))
        aam_scale  = lc.get("aam_scale",  lc.get("scale",  30))

        self.use_proj    = use_proj
        self.backbone    = ECAPA_TDNN(
            in_channels=cfg["data"]["n_mels"],
            channels=channels, emb_dim=emb_dim)
        self.ae          = DisentanglerAE(emb_dim, spk_dim, env_dim)
        self.spk_cls     = AAMSoftmax(
            emb_dim=spk_dim, num_classes=num_speakers,
            margin=aam_margin, scale=aam_scale)
        self.env_disc_ee = EnvDiscriminator(env_dim, disc_hid, disc_out)
        self.env_disc_es = EnvDiscriminator(spk_dim, disc_hid, disc_out)

        self.proj_head = ProjectionHead(spk_dim, proj_dim) if use_proj else None

    def forward(self, mel):
        
        e            = self.backbone(mel)
        e_s, e_e, e_r = self.ae(e)
        z = self.proj_head(e_s) if self.use_proj else None
        return e_s, e_e, e_r, e, z

    def get_embedding(self, mel):
        e     = self.backbone(mel)
        e_s, *_ = self.ae(e)
        return F.normalize(e_s, dim=1)
