import numpy as np
import pandas as pd
import torch
from geonamescache import GeonamesCache
from transformers import AutoTokenizer, AutoModel

MODEL = "gpt2"
N_CITIES = 3000

# --- build city table (no download needed) ---
gc = GeonamesCache()
countries = gc.get_countries()
rows = []
for c in gc.get_cities().values():
    cc = c["countrycode"]
    if cc not in countries:
        continue
    rows.append({
        "city": c["name"],
        "country_code": cc,
        "country": countries[cc]["name"],
        "continent": countries[cc]["continentcode"],
        "lat": c["latitude"],
        "lon": c["longitude"],
        "pop": c["population"],
    })

df = pd.DataFrame(rows).sort_values("pop", ascending=False)
df = df.drop_duplicates(["city", "country_code"]).head(N_CITIES).reset_index(drop=True)
print(df.head(), len(df))

# --- model ---
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, output_hidden_states=True).eval()

texts = df["city"].tolist()
feats = []
with torch.no_grad():
    for i, t in enumerate(texts):
        inputs = tok(t, return_tensors="pt")
        out = model(**inputs)
        # hidden state of the LAST token at every layer -> (layers+1, dim)
        hs = torch.stack(out.hidden_states)[:, 0, -1, :]
        feats.append(hs.numpy())
        if i % 500 == 0:
            print(i, "/", len(texts))

feats = np.stack(feats)  # (N, layers+1, dim)
np.save("data/feats_city.npy", feats)
df.to_csv("data/cities.csv", index=False)
print("saved", feats.shape)