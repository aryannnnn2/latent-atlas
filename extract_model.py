import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

name = sys.argv[1]                      # e.g. gpt2-medium
TEMPLATE = "I traveled to {}"
df = pd.read_csv("data/cities.csv", keep_default_na=False)
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name, output_hidden_states=True).eval()

feats = []
with torch.no_grad():
    for i, city in enumerate(df["city"]):
        ids = tok(TEMPLATE.format(city), return_tensors="pt")
        hs = torch.stack(model(**ids).hidden_states)[:, 0, -1, :]
        feats.append(hs.numpy().astype(np.float16))     # float16 keeps the file small
        if i % 500 == 0:
            print(name, i, "/", len(df), flush=True)

np.save(f"data/feats_{name}_travel.npy", np.stack(feats))
print("saved", name)