from __future__ import annotations
"""Decisive test: does deeds post-registration MIND residual discriminate
same-subject? On N real dataset1 pairs, register query_i(T1c) -> candidate_j(T2)
and score by MIND residual. Check the true match (diagonal) is min per row."""
import csv, subprocess, sys
from pathlib import Path
import nibabel as nib, numpy as np
from scipy.ndimage import map_coordinates, uniform_filter

ROOT = Path("ehl-paris-medical-image-retrieval")
DEEDS = Path("deedsBCV")
WS = Path("/tmp/deedsws"); WS.mkdir(exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SIZE = 96

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
    Dp = [uniform_filter((vol-np.roll(vol,o,(0,1,2)))**2, 2*patch+1) for o in offs]
    Dp = np.stack(Dp).astype(np.float32); V = Dp.mean(0)+1e-6
    m = np.exp(-Dp/V); m /= m.max(0, keepdims=True)+1e-6
    return m

def save(v, p): nib.save(nib.Nifti1Image(v, np.eye(4)), str(p))

def main():
    pairs = list(csv.DictReader(open(ROOT/"dataset1/train_pairs.csv")))[:N]
    Q = [load_ds(p["query_image"]) for p in pairs]
    T = [load_ds(p["target_image"]) for p in pairs]
    Tm = [mind(t) for t in T]
    for j, t in enumerate(T): save(t, WS/f"fix{j}.nii.gz")
    for i, q in enumerate(Q): save(q, WS/f"mov{i}.nii.gz")
    R = np.zeros((N, N), np.float32)
    for i in range(N):
        for j in range(N):
            o = WS/f"d_{i}_{j}"
            subprocess.run([str(DEEDS/"linearBCV"), "-F", str(WS/f"fix{j}.nii.gz"),
                            "-M", str(WS/f"mov{i}.nii.gz"), "-O", str(o)+"lin"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run([str(DEEDS/"deedsBCV"), "-F", str(WS/f"fix{j}.nii.gz"),
                            "-M", str(WS/f"mov{i}.nii.gz"), "-A", str(o)+"lin_matrix.txt",
                            "-O", str(o)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            def_ = np.asanyarray(nib.load(str(o)+"_deformed.nii.gz").dataobj).astype(np.float32)
            R[i, j] = float(np.abs(Tm[j] - mind(def_)).mean())  # MIND residual; lower=better
        am = int(np.argmin(R[i])); print(f"q{i}: argmin={am} true={i} {'OK' if am==i else 'MISS'}  row={np.round(R[i],4)}", flush=True)
    diag_min = sum(int(np.argmin(R[i])) == i for i in range(N))
    print(f"\ndeeds MIND-residual: true-match is min for {diag_min}/{N} queries")

if __name__ == "__main__":
    main()
