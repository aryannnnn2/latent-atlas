import plotly.graph_objects as go
from common import *

X = np.load("data/feats_travel.npy")[:, 12, :].astype(np.float32)   # GPT-2 small, travel prompt, layer 12
pred = np.zeros((len(df), 3))
for tr, te in folds(0):                       # out-of-fold: every city is predicted by a probe that never saw its country
    m = ridge_probe()
    m.fit(X[tr], XYZ[tr])
    pred[te] = m.predict(X[te])
plat, plon = to_latlon(pred)
err = haversine(LAT, LON, plat, plon)

hover = (df["city"] + ", " + df["country"] + "<br>error: " + err.round().astype(int).astype(str) + " km").values
fig = go.Figure()
fig.add_trace(go.Scattergeo(lat=LAT, lon=LON, mode="markers", text=hover, hoverinfo="text",
                            marker=dict(size=4, color=err, colorscale="Turbo", cmin=0, cmax=10000,
                                        colorbar=dict(title="error (km)", y=0.4, len=0.6)),
                            name="true location"))

worst = np.argsort(-err)[:40]
lats, lons = [], []
for i in worst:
    lats += [LAT[i], plat[i], None]
    lons += [LON[i], plon[i], None]
fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode="lines", line=dict(width=0.6, color="red"),
                            hoverinfo="skip", name="40 worst: true -> predicted"))
fig.add_trace(go.Scattergeo(lat=plat[worst], lon=plon[worst], mode="markers",
                            marker=dict(size=5, color="red"), text=hover[worst], hoverinfo="text",
                            name="predicted (40 worst)"))
fig.update_layout(title="Latent Atlas: where GPT-2 small thinks cities are (test countries unseen)",
                  geo=dict(projection_type="natural earth"), height=650,
                  margin=dict(l=0, r=0, t=50, b=0),
                  legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
fig.write_html("docs/index.html", include_plotlyjs="cdn")
print("saved docs/index.html", "| median error:", np.median(err))