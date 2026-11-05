import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def ridge_probe():
    return make_pipeline(StandardScaler(), Ridge(alpha=1000.0))


def folds(seed, k=5):
    """each seed = a different random assignment of whole countries to folds (same as probe_v2.py)"""
    perm = np.random.default_rng(seed).permutation(countries)
    fold_of = {c: i % k for i, c in enumerate(perm)}
    f = np.array([fold_of[c] for c in cc])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def cv_errors(X, seed, make_model=ridge_probe):
    pred = np.zeros((len(df), 3))
    for tr, te in folds(seed):
        m = make_model()
        m.fit(X[tr], XYZ[tr])
        pred[te] = m.predict(X[te])
    la, lo = to_latlon(pred)
    return haversine(LAT, LON, la, lo)


def ci(a):
    a = np.array(a, dtype=float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))