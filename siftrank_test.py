import csv, subprocess, os, glob
from pathlib import Path
import numpy as np

ROOT = Path("/app/ehl-paris-medical-image-retrieval")
APP = Path("/app")
WS = Path("/tmp/sift"); WS.mkdir(exist_ok=True)
N = 12

def resolve(ip):
    p = ROOT/ip
    if p.exists(): return p
    if p.name.endswith(".nii.gz"):
        fb = p.with_name(p.name[:-3])
        if fb.exists(): return fb
    raise FileNotFoundError(p)

def extract(img, key):
    if not Path(key).exists():
        subprocess.run([str(APP/"featExtract"), str(img), str(key)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    pairs = list(csv.DictReader(open(ROOT/"dataset1/train_pairs.csv")))[:N]
    qkeys, tkeys = [], []
    for i, p in enumerate(pairs):
        qk = WS/f"q{i}.key"; tk = WS/f"t{i}.key"
        extract(resolve(p["query_image"]), qk); qkeys.append(str(qk))
        extract(resolve(p["target_image"]), tk); tkeys.append(str(tk))
        print(f"extracted {i+1}/{N}", flush=True)
    listfile = WS/"list.txt"
    listfile.write_text("\n".join(qkeys + tkeys) + "\n")
    os.chdir(WS)
    subprocess.run([str(APP/"featMatchMultiple"), "-f", str(listfile)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mv = WS/"matching_votes.txt"
    if not mv.exists():
        cand = glob.glob(str(WS/"*votes*")) + glob.glob(str(WS/"*.txt"))
        print("matching_votes.txt missing; files:", cand); return
    M = np.loadtxt(mv)
    print("vote matrix shape", M.shape)
    block = M[:N, N:]   # queries x targets
    ok = 0
    for i in range(N):
        am = int(np.argmax(block[i])); ok += (am == i)
        print(f"q{i}: argmax={am} true={i} {'OK' if am==i else 'MISS'}  selfvote={block[i,i]:.1f} max={block[i].max():.1f}")
    print(f"\nSIFT-Rank cross-modal: true-match is top for {ok}/{N}")

if __name__ == "__main__":
    main()
