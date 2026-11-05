import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geonamescache import GeonamesCache
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline

SEEDS = 10            # lower to 5 if it runs too slowly
ALPHA = 1000.0
FEATS = {"bare": "data/feats_city.npy",
         "travel": "data/feats_travel.npy",
         "isin": "data/feats_isin.npy"}

df = pd.read_csv("data/cities.csv", keep_default_na=False)
cc = df["country_code"].values
countries = np.unique(cc)
LAT, LON = df["lat"].values, df["lon"].values


def to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)


def to_latlon(v):
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1))), np.degrees(np.arctan2(v[:, 1], v[:, 0]))


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


XYZ = to_xyz(LAT, LON)

# primary language per country (first code in geonamescache's list)
C = GeonamesCache().get_countries()
lang = np.array([(C[c]["languages"].split(",")[0].split("-")[0] if C[c]["languages"] else "none")
                 for c in cc])


def folds(seed, k=5):
    """each seed = a different random assignment of whole countries to folds"""
    perm = np.random.default_rng(seed).permutation(countries)
    fold_of = {c: i % k for i, c in enumerate(perm)}
    f = np.array([fold_of[c] for c in cc])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def cv_errors(X, seed, kind="ridge", shuffle=False):
    rng = np.random.default_rng(seed + 1000)
    pred = np.zeros((len(df), 3))
    for tr, te in folds(seed):
        y = XYZ[tr]
        if shuffle:
            y = y[rng.permutation(len(tr))]
        if kind == "mean":                      # always guess the training mean
            pred[te] = y.mean(axis=0)
            continue
        if kind == "lang":                      # language-only baseline
            m = make_pipeline(OneHotEncoder(handle_unknown="ignore"), Ridge(alpha=1.0))
        else:
            m = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
        m.fit(X[tr], y)
        pred[te] = m.predict(X[te])
    la, lo = to_latlon(pred)
    return haversine(LAT, LON, la, lo)


def ci(a):
    a = np.array(a, dtype=float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))


# ---------- probe every layer, every template, many seeds ----------
res = {}
for name, path in FEATS.items():
    X_all = np.load(path)
    means, cis = [], []
    for l in range(X_all.shape[1]):
        meds = [np.median(cv_errors(X_all[:, l, :], s)) for s in range(SEEDS)]
        m, h = ci(meds)
        means.append(m); cis.append(h)
        print(f"{name:7s} layer {l:2d}  {m:6.0f} +/- {h:4.0f} km", flush=True)
    res[name] = (np.array(means), np.array(cis))

# ---------- controls (use the bare-name template at its best layer) ----------
Xb_all = np.load(FEATS["bare"])
best = int(np.argmin(res["bare"][0]))
Xb = Xb_all[:, best, :]
dummy = np.zeros((len(df), 1))

mean_base = ci([np.median(cv_errors(dummy, s, kind="mean")) for s in range(SEEDS)])
shuf_ctrl = ci([np.median(cv_errors(Xb, s, shuffle=True)) for s in range(SEEDS)])
lang_base = ci([np.median(cv_errors(lang.reshape(-1, 1), s, kind="lang")) for s in range(SEEDS)])

print("\n=== SUMMARY (median km, mean +/- 95% CI over seeds) ===")
for name in FEATS:
    m, h = res[name]
    b = int(np.argmin(m))
    print(f"{name:7s} best layer {b:2d}: {m[b]:6.0f} +/- {h[b]:.0f}")
print(f"guess-the-mean        : {mean_base[0]:6.0f} +/- {mean_base[1]:.0f}")
print(f"shuffled-label probe  : {shuf_ctrl[0]:6.0f} +/- {shuf_ctrl[1]:.0f}")
print(f"language-only baseline: {lang_base[0]:6.0f} +/- {lang_base[1]:.0f}")

# ---------- can the hidden state decode the language? ----------
counts = pd.Series(lang).value_counts()
y_lang = np.where(np.isin(lang, counts[counts >= 30].index), lang, "other")
accs, majs = [], []
for tr, te in folds(0):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.01))
    clf.fit(Xb[tr], y_lang[tr])
    seen = np.isin(y_lang[te], np.unique(y_lang[tr]))      # skip languages absent from training
    accs.append((clf.predict(Xb[te][seen]) == y_lang[te][seen]).mean())
    top = pd.Series(y_lang[tr]).value_counts().index[0]
    majs.append((y_lang[te][seen] == top).mean())
print(f"\nlanguage-ID probe accuracy: {np.mean(accs):.2f}  (majority-class: {np.mean(majs):.2f})")

# ---------- per-continent error, best layer, seed 0 ----------
df["err_km"] = cv_errors(Xb, 0)
print("\nBy continent (median km):\n", df.groupby("continent")["err_km"].median().sort_values())

# ---------- plot ----------
plt.figure(figsize=(8, 5))
for name, (m, h) in res.items():
    plt.plot(m, marker="o", label=f"probe: {name}")
    plt.fill_between(range(len(m)), m - h, m + h, alpha=0.2)
plt.axhline(mean_base[0], color="gray", ls="--", label="guess the mean")
plt.axhline(shuf_ctrl[0], color="red", ls=":", label="shuffled labels")
plt.axhline(lang_base[0], color="green", ls="-.", label="language-only baseline")
plt.xlabel("GPT-2 layer"); plt.ylabel("median error (km), 3D probe")
plt.title("Probe error by layer and prompt (mean, 95% CI over seeds)")
plt.legend(); plt.tight_layout()
plt.savefig("results/v2_layers.png", dpi=150)
print("saved results/v2_layers.png")