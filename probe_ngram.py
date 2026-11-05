from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from common import *

SEEDS = 10
names = df["city"].values.astype(str)

best = None
for alpha in [0.3, 1.0, 3.0, 10.0]:
    def make():
        return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4)),
                             Ridge(alpha=alpha))
    m, h = ci([np.median(cv_errors(names, s, make)) for s in range(SEEDS)])
    print(f"alpha {alpha:5.1f}: {m:6.0f} +/- {h:4.0f} km", flush=True)
    if best is None or m < best[0]:
        best = (m, h, alpha)

print(f"\nBEST char n-gram baseline: {best[0]:.0f} +/- {best[1]:.0f} km (alpha={best[2]})")
print("for comparison, bare-name GPT-2 probe: 3845 +/- 42 km")