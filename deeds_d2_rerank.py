from __future__ import annotations
"""Deeds registration-residual re-ranker for dataset2.
Prefilter top-K candidates per query from the template ranking, deformably
register query(T1c)->candidate(T2) with deedsBCV, score by post-registration MIND
residual, then Hungarian over the assembled score matrix. Writes a d2 submission.
No synthetic validation — real d2, submit directly."""
import csv, subprocess, sys, time
from pathlib import Path
import nibabel as nib, numpy as np
from scipy.ndimage import map_coordinates, uniform_filter
from scipy.optimize import linear_sum_assignment

ROOT = Path("ehl-paris-medical-image-retrieval")
DEEDS = Path("deedsBCV")
WS = Path("/tmp/deeds_d2"); WS.mkdir(exist_ok=True)
SIZE = 96
TOPK = int(sys.argv[1]) if len(sys.argv) > 1 else 6
TEMPLATE_RANK = Path("submissions/d2_template_g44_hungarian.csv")
OUT = Path("submissions/d2_deeds_rerank.csv")


def resolve(ip):
    p = ROOT/ip
    if p.exists(): return p
    if p.name.endswith(".nii.gz"):
        fb = p.with_name(p.name[:-3])
        if fb.exists(): return fb
    raise FileNotFoundError(p)

def load_ds(ip, size=SIZE):
    v = np.asanyarray(nib.load(str(resolve(ip))).dataobj).astype(np.float32)
    if v.ndim > 3: v = v[..., 0]
    v = np.nan_to_num(v); sh = np.array(v.shape, float)
    g = np.stack(np.meshgrid(*[np.linspace(0, s-1, size) for s in sh], indexing="ij"), 0)
    return map_coordinates(v, g, order=1).astype(np.float32)

def mind(vol, d=2, patch=2):
    offs = [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0),(0,0,d),(0,0,-d)]
    Dp = np.stack([uniform_filter((vol-np.roll(vol,o,(0,1,2)))**2, 2*patch+1) for o in offs]).astype(np.float32)
    V = Dp.mean(0)+1e-6; m = np.exp(-Dp/V); m /= m.max(0, keepdims=True)+1e-6
    return m

def save_vol(v, p):
    if not p.exists(): nib.save(nib.Nifti1Image(v, np.eye(4)), str(p))

def deeds_residual(qid, cid, q_path_tag, fix_mind):
    o = WS/f"d_{qid}_{cid}"
    subprocess.run([str(DEEDS/"linearBCV"), "-F", str(WS/f"{cid}.nii.gz"),
                    "-M", str(WS/f"{qid}.nii.gz"), "-O", str(o)+"lin"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([str(DEEDS/"deedsBCV"), "-F", str(WS/f"{cid}.nii.gz"),
                    "-M", str(WS/f"{qid}.nii.gz"), "-A", str(o)+"lin_matrix.txt", "-O", str(o)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    df = Path(str(o)+"_deformed.nii.gz")
    if not df.exists(): return 9.9
    res = float(np.abs(fix_mind - mind(np.asanyarray(nib.load(str(df)).dataobj).astype(np.float32))).mean())
    for f in WS.glob(f"d_{qid}_{cid}*"): f.unlink()
    return res

def read_csv(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

def process(split, rows_out):
    qr = read_csv(ROOT/"dataset2"/f"{split}_queries.csv")
    gr = read_csv(ROOT/"dataset2"/f"{split}_gallery.csv")
    qids = [r["query_id"] for r in qr]; tids = [r["target_id"] for r in gr]
    qpath = {r["query_id"]: r["query_image"] for r in qr}
    tpath = {r["target_id"]: r["target_image"] for r in gr}
    # template ranking -> per-query ordered candidates (prefilter + fallback order)
    rank = {r["query_id"]: r["target_id_ranking"].split() for r in read_csv(TEMPLATE_RANK) if r["query_id"] in set(qids)}
    # save all volumes once
    for q in qids: save_vol(load_ds(qpath[q]), WS/f"{q}.nii.gz")
    fix_mind = {}
    for t in tids:
        save_vol(load_ds(tpath[t]), WS/f"{t}.nii.gz")
        fix_mind[t] = mind(np.asanyarray(nib.load(str(WS/f"{t}.nii.gz")).dataobj).astype(np.float32))
    n = len(qids); tindex = {t: j for j, t in enumerate(tids)}
    S = np.full((n, n), -1e9, np.float64)
    t0 = time.time()
    for i, q in enumerate(qids):
        cands = rank[q][:TOPK]
        for c in cands:
            r = deeds_residual(q, c, q, fix_mind[c])
            S[i, tindex[c]] = 100.0 - 100.0*r          # lower residual -> higher score
        # fallback: non-candidates keep template order, below all candidates
        for pos, c in enumerate(rank[q]):
            if c not in cands:
                S[i, tindex[c]] = -1e6 - pos
        print(f"  {split} q{i+1}/{n} done ({time.time()-t0:.0f}s)", flush=True)
    # Hungarian bijection
    ri, ci = linear_sum_assignment(-S)
    asg = np.empty(n, np.int64); asg[ri] = ci
    S[np.arange(n), asg] = S.max() + 1e6
    ta = np.asarray(tids)
    for i, q in enumerate(qids):
        rows_out.append({"query_id": q, "target_id_ranking": " ".join(ta[np.argsort(-S[i])].tolist())})
    print(f"{split}: {n}x{len(tids)} done", flush=True)

def main():
    rows = []
    for split in ("val", "test"):
        process(split, rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"]); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
