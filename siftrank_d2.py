"""Full SIFT-Rank d2 (and optionally d1/d3) retrieval on the server.
featExtract each volume -> .key; descriptor voting (Lowe ratio) on GPU -> NxN vote
matrix -> Hungarian -> ranking. Writes a submission CSV."""
import csv, subprocess, sys
from pathlib import Path
import numpy as np, torch
from scipy.optimize import linear_sum_assignment

ROOT = Path("/app/ehl-paris-medical-image-retrieval")
APP = Path("/app")
KEYS = Path("/app/.siftkeys"); KEYS.mkdir(exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.65
DATASETS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["dataset2"]
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/app/submissions/siftrank_d2.csv")

def resolve(ip):
    p = ROOT/ip
    if p.exists(): return p
    if p.name.endswith(".nii.gz"):
        fb = p.with_name(p.name[:-3])
        if fb.exists(): return fb
    raise FileNotFoundError(p)

def keyfile(img_id, img_path):
    kf = KEYS/f"{img_id}.key"
    if not kf.exists() or kf.stat().st_size < 100:
        subprocess.run([str(APP/"featExtract"), str(resolve(img_path)), str(kf)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return kf

def load_desc(path):
    rows = []
    started = False
    for line in open(path):
        if line.startswith("Features:"): started = True; continue
        if not started or line.startswith("Scale-space"): continue
        parts = [p for p in line.rstrip().replace("\t"," ").split(" ") if p != ""]
        if len(parts) < 64: continue
        rows.append([float(x) for x in parts[-64:]])
    if not rows: return torch.zeros((0,64), device=DEV)
    return torch.tensor(rows, dtype=torch.float32, device=DEV)

def votes(Q, T, ratio):
    if Q.shape[0]==0 or T.shape[0]==0: return 0.0
    D = torch.cdist(Q, T)            # nq x nt
    if T.shape[0] < 2: return 0.0
    d, _ = torch.topk(D, 2, dim=1, largest=False)
    good = d[:,0] < ratio*d[:,1]
    return float(good.sum().item())

def read_csv(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

def process(ds, split, rows_out):
    qr = read_csv(ROOT/ds/f"{split}_queries.csv"); gr = read_csv(ROOT/ds/f"{split}_gallery.csv")
    qids=[r["query_id"] for r in qr]; tids=[r["target_id"] for r in gr]
    print(f"{ds}/{split} extracting {len(qids)}+{len(tids)}", flush=True)
    Q = [load_desc(keyfile(r["query_id"], r["query_image"])) for r in qr]
    T = [load_desc(keyfile(r["target_id"], r["target_image"])) for r in gr]
    nq=np.array([q.shape[0] for q in Q]); nt=np.array([t.shape[0] for t in T])
    n=len(qids); M=np.zeros((n,len(tids)))
    for i in range(n):
        for j in range(len(tids)):
            M[i,j]=votes(Q[i], T[j], RATIO)
        if (i+1)%10==0: print(f"  matched {i+1}/{n}", flush=True)
    J = M/(nq[:,None]+nt[None,:]-M+1e-6)
    S = J.astype(np.float64)
    if S.shape[0]==S.shape[1]:
        ri,ci=linear_sum_assignment(-S); asg=np.empty(S.shape[0],np.int64); asg[ri]=ci
        S[np.arange(S.shape[0]),asg]=S.max()+1e6
    ta=np.asarray(tids)
    for i,q in enumerate(qids):
        rows_out.append({"query_id":q,"target_id_ranking":" ".join(ta[np.argsort(-S[i])].tolist())})
    np.save(str(OUT)+f".{ds}.{split}.votes.npy", M)  # save raw votes for fusion
    print(f"{ds}/{split} done", flush=True)

def main():
    rows=[]
    for ds in DATASETS:
        for split in ("val","test"):
            process(ds, split, rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["query_id","target_id_ranking"]);w.writeheader();w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}", flush=True)

if __name__=="__main__":
    main()
