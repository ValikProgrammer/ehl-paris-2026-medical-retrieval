"""Reimplement SIFT-Rank keypoint voting (the crashing featMatchMultiple) in numpy.
Pairwise similarity = number of Lowe-ratio descriptor matches between two keypoint
sets. Validates cross-modal discrimination on the d1 .key files in /tmp/sift."""
import glob, sys
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
WS = "/tmp/sift"

def load_desc(path):
    rows = []
    started = False
    for line in open(path):
        if line.startswith("Features:"): started = True; continue
        if not started: continue
        if line.startswith("Scale-space"): continue
        parts = [p for p in line.rstrip().replace("\t", " ").split(" ") if p != ""]
        if len(parts) < 64: continue
        rows.append([float(x) for x in parts[-64:]])
    d = np.asarray(rows, dtype=np.float32)
    return d

def votes(qd, td, ratio):
    if len(qd) == 0 or len(td) == 0: return 0
    qn = (qd*qd).sum(1)[:, None]; tn = (td*td).sum(1)[None, :]
    D = qn + tn - 2.0*qd @ td.T
    part = np.partition(D, 1, axis=1)[:, :2]
    d1 = part.min(1); d2 = part.max(1)
    return int((d1 < (ratio*ratio)*d2).sum())

def main():
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
    qd = [load_desc(f"{WS}/q{i}.key") for i in range(N)]
    td = [load_desc(f"{WS}/t{i}.key") for i in range(N)]
    nq = np.array([len(x) for x in qd]); nt = np.array([len(x) for x in td])
    print(f"ratio={ratio} desc counts q:", nq[:5], "...", flush=True)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = votes(qd[i], td[j], ratio)
    J = M / (nq[:, None] + nt[None, :] - M + 1e-6)   # jaccard-normalized
    ok_raw = sum(int(np.argmax(M[i]))==i for i in range(N))
    ok_j = sum(int(np.argmax(J[i]))==i for i in range(N))
    for i in range(N):
        ar, aj = int(np.argmax(M[i])), int(np.argmax(J[i]))
        print(f"q{i}: raw_am={ar} jac_am={aj} true={i} {'rawOK' if ar==i else ''} {'jacOK' if aj==i else ''}")
    print(f"\nratio={ratio}: raw {ok_raw}/{N}, jaccard {ok_j}/{N}")

if __name__ == "__main__":
    main()
