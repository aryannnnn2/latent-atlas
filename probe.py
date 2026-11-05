import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv("data/cities.csv", keep_default_na=False)
X_all = np.load("data/feats.npy")          # (N, layers+1, dim)
Y = df[["lat", "lon"]].values
groups = df["country_code"].values
rng = np.random.default_rng(0)


def haversine(lat1, lon1, lat2, lon2):
    """distance in km between two points on Earth"""
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def cv_predict(X, Y, groups, alpha=1000.0, shuffle=False):
    """5-fold CV where whole countries are held out"""
    pred = np.zeros_like(Y, dtype=float)
    for tr, te in GroupKFold(n_splits=5).split(X, Y, groups):
        Ytr = rng.permutation(Y[tr]) if shuffle else Y[tr]   # shuffle = control
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        m.fit(X[tr], Ytr)
        pred[te] = m.predict(X[te])
    pred[:, 0] = np.clip(pred[:, 0], -90, 90)
    pred[:, 1] = np.clip(pred[:, 1], -180, 180)
    return pred


def errors(pred):
    return haversine(Y[:, 0], Y[:, 1], pred[:, 0], pred[:, 1])


# --- probe every layer ---
n_layers = X_all.shape[1]
real_med, ctrl_med = [], []
for l in range(n_layers):
    X = X_all[:, l, :]
    real_med.append(np.median(errors(cv_predict(X, Y, groups))))
    ctrl_med.append(np.median(errors(cv_predict(X, Y, groups, shuffle=True))))
    print(f"layer {l:2d}  probe: {real_med[-1]:7.0f} km   shuffled: {ctrl_med[-1]:7.0f} km")

# baseline: always guess the average location
mean_pt = Y.mean(axis=0)
mean_err = np.median(haversine(Y[:, 0], Y[:, 1], mean_pt[0], mean_pt[1]))

best = int(np.argmin(real_med))
print("best layer:", best, "median error km:", real_med[best], "| guess-the-mean:", mean_err)

# --- plot 1: error per layer ---
plt.figure(figsize=(7, 4))
plt.plot(real_med, marker="o", label="probe (real labels)")
plt.plot(ctrl_med, marker="x", label="control (shuffled labels)")
plt.axhline(mean_err, color="gray", linestyle="--", label="always guess average")
plt.xlabel("GPT-2 layer"); plt.ylabel("median error (km)")
plt.title("Where does GPT-2 know geography?")
plt.legend(); plt.tight_layout()
plt.savefig("results/error_per_layer.png", dpi=150)

# --- best-layer predictions ---
pred = cv_predict(X_all[:, best, :], Y, groups)
df["err_km"] = errors(pred)
df["pred_lat"], df["pred_lon"] = pred[:, 0], pred[:, 1]
df.to_csv("results/predictions.csv", index=False)

# --- plot 2: real map vs GPT-2's imagined map ---
colors = df["continent"].astype("category").cat.codes
fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharex=True, sharey=True)
ax[0].scatter(df["lon"], df["lat"], c=colors, cmap="tab10", s=6)
ax[0].set_title("Real world")
ax[1].scatter(df["pred_lon"], df["pred_lat"], c=colors, cmap="tab10", s=6)
ax[1].set_title(f"GPT-2's imagined world (layer {best}, unseen countries)")
for a in ax:
    a.set_xlabel("longitude"); a.set_ylabel("latitude")
plt.tight_layout()
plt.savefig("results/real_vs_imagined.png", dpi=150)

# --- plot 3: error per continent ---
by_cont = df.groupby("continent")["err_km"].median().sort_values()
plt.figure(figsize=(6, 4))
by_cont.plot(kind="bar")
plt.ylabel("median error (km)"); plt.title("Where is GPT-2's map worst?")
plt.tight_layout()
plt.savefig("results/error_per_continent.png", dpi=150)

print("\nWorst 15 cities:")
print(df.sort_values("err_km", ascending=False)[["city", "country", "err_km"]].head(15))
print("\nBy continent:\n", by_cont)