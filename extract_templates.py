import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

TEMPLATES = {"travel": "I traveled to {}", "isin": "{} is a city in"}

df = pd.read_csv("data/cities.csv", keep_default_na=False)
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2", output_hidden_states=True).eval()

for name, t in TEMPLATES.items():
    feats = []
    with torch.no_grad():
        for i, city in enumerate(df["city"]):
            ids = tok(t.format(city), return_tensors="pt")
            hs = torch.stack(model(**ids).hidden_states)[:, 0, -1, :]
            feats.append(hs.numpy())
            if i % 500 == 0:
                print(name, i, "/", len(df), flush=True)
    np.save(f"data/feats_{name}.npy", np.stack(feats))
    print("saved", name)