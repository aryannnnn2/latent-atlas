import os
import matplotlib.pyplot as plt
from common import *

SEEDS = 5
MODELS = [("gpt2", 124, "data/feats_travel.npy"),
          ("gpt2-medium", 355, "data/feats_gpt2-medium_travel.npy")]

rows = []
for label, size, path in MODELS:
    if not os.path.exists(path):
        print("skipping", label, "(file not found)")
        continue
    X_all = np.load(path).astype(np.float32)
    n = X_all.shape[1]
    layers = sorted(set(range(0, n, 2)) | {n - 1})          # every 2nd layer + the last
    best = None
    for l in layers:
        m, h = ci([np.median(cv_errors(X_all[:, l, :], s)) for s in range(SEEDS)])
        print(f"{label:12s} layer {l:2d}/{n-1}  {m:6.0f} +/- {h:4.0f} km", flush=True)
        if best is None or m < best[1]:
            best = (l, m, h)
    rows.append((label, size, *best))

print("\n=== SCALE SUMMARY (best layer, median km +/- 95% CI) ===")
for label, size, l, m, h in rows:
    print(f"{label:12s} {size:4d}M params  layer {l:2d}: {m:6.0f} +/- {h:.0f}")

plt.figure(figsize=(6, 4))
plt.errorbar([r[1] for r in rows], [r[3] for r in rows], yerr=[r[4] for r in rows],
             marker="o", capsize=4)
for label, size, l, m, h in rows:
    plt.annotate(label, (size, m), textcoords="offset points", xytext=(5, 8))
plt.xscale("log")
plt.xlabel("parameters (millions, log scale)")
plt.ylabel("median error (km), best layer")
plt.title("Does a bigger GPT-2 build a better map?")
plt.tight_layout()
plt.savefig("results/scale.png", dpi=150)
print("saved results/scale.png")