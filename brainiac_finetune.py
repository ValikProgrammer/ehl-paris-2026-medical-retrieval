from __future__ import annotations
# /// script
# requires-python = ">=3.10"
# dependencies = ["monai>=1.3.0","nibabel>=5.3","numpy","torch","scipy","scikit-learn"]
# ///

"""Fine-tune the pretrained BrainIAC ViT with a projection head and symmetric
InfoNCE to align T1 post-contrast queries with T2 targets, trained on the
geom+contrast augmented dataset1 manifest. Then embed all pools, score by cosine,
apply Hungarian assignment, and write a Kaggle submission.

Frozen-cosine BrainIAC failed (~0.15) because one frozen encoder maps the two
modalities into unmatched spaces. Fine-tuning learns the cross-modal alignment;
the heavy augmentation provides geom/contrast invariance for dataset2/3.
"""

import argparse, csv, hashlib, math, re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT
from scipy.optimize import linear_sum_assignment

IMAGE_SIZE = (96, 96, 96)
DATA_ROOT = Path("/app/ehl-paris-medical-image-retrieval")


# ---------------- data ----------------

def read_csv(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

def write_csv(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"]); w.writeheader(); w.writerows(rows)

def resolve(data_root, image_path):
    p = Path(image_path)
    if not p.is_absolute(): p = data_root / p
    if p.exists(): return p
    if p.name.endswith(".nii.gz"):
        fb = p.with_name(p.name[:-3])
        if fb.exists(): return fb
    raise FileNotFoundError(p)

def norm_nonzero(v):
    v = np.nan_to_num(v.astype(np.float32), nan=0., posinf=0., neginf=0.)
    m = np.abs(v) > 1e-6; fg = v[m]
    if fg.size == 0: return v
    mu, sd = float(fg.mean()), float(fg.std())
    out = v.copy(); out[m] = (out[m]-mu)/sd if sd > 1e-6 else out[m]-mu; out[~m] = 0.
    return out

class TCache:
    def __init__(self, data_root, cache_dir):
        self.data_root = data_root; self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.mem = {}
    def get(self, image_id, image_path):
        if image_id in self.mem: return self.mem[image_id]
        f = self.cache / f"{image_id}.npy"
        if f.exists():
            t = torch.from_numpy(np.load(f))
        else:
            img = nib.load(str(resolve(self.data_root, image_path)))
            v = np.asanyarray(img.dataobj).astype(np.float32)
            if v.ndim > 3: v = v[..., 0]
            v = norm_nonzero(v)
            t = torch.from_numpy(v).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=IMAGE_SIZE, mode="trilinear", align_corners=False).squeeze(0).contiguous()
            np.save(f, t.numpy())
        self.mem[image_id] = t
        return t


def base_subject(qid): return re.sub(r"_aug\d+$", "", qid)

def subject_split(pairs, frac, seed):
    groups = {}
    for r in pairs: groups.setdefault(base_subject(r["query_id"]), []).append(r)
    keyed = sorted(groups.items(), key=lambda kv: hashlib.sha256(f"{seed}:{kv[0]}".encode()).hexdigest())
    hn = max(1, min(len(keyed)-1, round(len(keyed)*frac)))
    hold = [r for _, rs in keyed[:hn] for r in rs]; train = [r for _, rs in keyed[hn:] for r in rs]
    return train, hold


def augment(x):  # x: (B,1,D,H,W) light TTA flips
    for ax in (2, 3, 4):
        if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[ax])
    return x


# ---------------- model ----------------

class BrainIACEncoder(nn.Module):
    def __init__(self, ckpt, proj=256, freeze_layers=0):
        super().__init__()
        self.backbone = ViT(in_channels=1, img_size=IMAGE_SIZE, patch_size=(16,16,16),
                            hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12, save_attn=False)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        bsd = {k[9:]: v for k, v in sd.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(bsd, strict=False)
        self.head = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, proj))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))
    def forward(self, x):
        feats = self.backbone(x)
        pooled = feats[0].mean(dim=1)
        return F.normalize(self.head(pooled), dim=1)


# ---------------- eval ----------------

@torch.no_grad()
def embed_ids(model, cache, ids, paths, device, bs=8):
    model.eval(); out = []
    for i in range(0, len(ids), bs):
        xs = torch.stack([cache.get(ids[j], paths[j]) for j in range(i, min(i+bs, len(ids)))]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
            out.append(model(xs).float().cpu())
    return torch.cat(out).numpy()

def mrr(scores, q_ids, t_ids, truth):
    s = 0.
    for i, q in enumerate(q_ids):
        order = np.argsort(-scores[i]); tgt = truth[q]
        rank = int(np.where(np.asarray(t_ids)[order] == tgt)[0][0]) + 1
        s += 1.0/rank
    return s/len(q_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--checkpoint", type=Path, default=Path("/app/BrainIAC.ckpt"))
    ap.add_argument("--train-pair-csv", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("/app/.bic_cache"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=Path("/app/submissions/brainiac_finetune_submission.csv"))
    ap.add_argument("--run-dir", type=Path, default=Path("/app/runs/bic"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260627)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    cache = TCache(args.data_root, args.cache_dir)

    pairs = read_csv(args.train_pair_csv)
    train_pairs, hold = subject_split(pairs, args.holdout_frac, 20260627)
    print(f"pairs={len(pairs)} train={len(train_pairs)} holdout={len(hold)}", flush=True)

    model = BrainIACEncoder(args.checkpoint).to(device)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
        {"params": [model.logit_scale], "lr": args.lr_head},
    ], weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    h_q = [r["query_id"] for r in hold]; h_qp = [r["query_image"] for r in hold]
    h_t = [r["target_id"] for r in hold]; h_tp = [r["target_image"] for r in hold]
    truth = {r["query_id"]: r["target_id"] for r in hold}

    best = -1.0
    order = list(range(len(train_pairs)))
    for ep in range(1, args.epochs+1):
        model.train(); np.random.shuffle(order); tot = 0.; seen = 0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i+args.batch_size]
            if len(idx) < 2: continue
            q = augment(torch.stack([cache.get(train_pairs[j]["query_id"], train_pairs[j]["query_image"]) for j in idx])).to(device)
            t = augment(torch.stack([cache.get(train_pairs[j]["target_id"], train_pairs[j]["target_image"]) for j in idx])).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
                qe = model(q); te = model(t)
                scale = model.logit_scale.clamp(max=math.log(100)).exp()
                logits = scale * qe @ te.T
                lbl = torch.arange(len(idx), device=device)
                loss = 0.5*(ce(logits, lbl) + ce(logits.T, lbl))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.detach().cpu())*len(idx); seen += len(idx)
        qv = embed_ids(model, cache, h_q, h_qp, device, args.batch_size)
        tv = embed_ids(model, cache, h_t, h_tp, device, args.batch_size)
        m = mrr(qv @ tv.T, h_q, h_t, truth)
        print(f"epoch {ep:02d} loss={tot/max(1,seen):.4f} holdout_all_gallery_mrr={m:.4f}", flush=True)
        if m > best:
            best = m; torch.save(model.state_dict(), args.run_dir/"best.pt")
    print(f"best holdout_mrr={best:.4f}", flush=True)

    # ---- generate submission with best model ----
    model.load_state_dict(torch.load(args.run_dir/"best.pt", map_location=device))
    rows = []
    for ds in ("dataset1", "dataset2", "dataset3"):
        for split in ("val", "test"):
            qf = args.data_root/ds/f"{split}_queries.csv"; gf = args.data_root/ds/f"{split}_gallery.csv"
            if not qf.exists(): continue
            qr = read_csv(qf); gr = read_csv(gf)
            qids = [r["query_id"] for r in qr]; tids = [r["target_id"] for r in gr]
            qv = embed_ids(model, cache, qids, [r["query_image"] for r in qr], device, args.batch_size)
            tv = embed_ids(model, cache, tids, [r["target_image"] for r in gr], device, args.batch_size)
            S = (qv @ tv.T).astype(np.float64)
            if S.shape[0] == S.shape[1]:
                ri, ci = linear_sum_assignment(-S)
                asg = np.empty(S.shape[0], dtype=np.int64); asg[ri] = ci
                S[np.arange(S.shape[0]), asg] = S.max()+1e6
            ta = np.asarray(tids)
            for i, q in enumerate(qids):
                rows.append({"query_id": q, "target_id_ranking": " ".join(ta[np.argsort(-S[i])].tolist())})
            print(f"embedded+ranked {ds}/{split}", flush=True)
    write_csv(args.out, rows)
    print(f"wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
