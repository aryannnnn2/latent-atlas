# Latent Atlas: does GPT-2 have a mental map of the world?

I trained linear probes on GPT-2's hidden states to predict the location of 3,000 cities. The test only uses **countries the probe never saw during training**.

**Headline result:** with a 3D probe on GPT-2 small's final layer, median location error is **~3,500 km** (prompt "I traveled to {city}") and **~3,850 km** from the bare city name, versus **~6,250 km** for guessing the average. GPT-2 medium gets **~2,840 km** with the same prompt. Two baselines that use no hidden states get ~4,060 km (primary language only) and ~4,420 km (character n-grams of the name), so much of the signal is language and spelling, but the GPT-2 probes are clearly better than both.

**[Interactive map demo](https://aryannnnn2.github.io/latent-atlas/)**

## Why this matters

GPT-2 only ever read text, and nobody told it where cities are. If a simple linear model can read location out of its hidden states, the model has picked up some geography from language alone. The baselines below show how much of that is just recognizing a name's language or spelling.

![error per layer](results/city_error_per_layer.png)

![real vs imagined](results/city_real_vs_imagined.png)

![error per continent](results/city_error_per_continent.png)

## How it works

1. Take the 3,000 most populous cities (from `geonamescache`).
2. Feed each city name to GPT-2 and save the hidden state of the last token at every layer.
3. For each layer, train a Ridge regression to predict location.
4. Evaluate with splits by country, so test countries are fully unseen. A random split would let the probe memorize "India means this area".
5. Controls: a probe trained on shuffled labels, an "always guess the average" baseline, a language-only baseline, and a character n-gram baseline.

## Findings

- Location information builds up with depth: with the first (lat/lon) probe, error falls from ~4,930 km at layer 0 to ~4,120 km at layer 12. The gain is gradual, not a sharp peak in the middle layers.
- A first experiment used "City, Country" as input. The last token was then the country, so every city in a country got nearly the same vector. Error was ~2,260 km at every layer, which only shows that country word embeddings encode geography. Switching to the city name alone gave the more honest test.
- Best on Europe and Asia, worst on Oceania (~15,700 km).
- Worst cities: Townsville, Melbourne, Sydney, and many Philippine cities (17,000-19,000 km off with the first probe).

## Update: 3D probe, prompts, seeds, language control

- Probes predict a 3D unit vector, which removes the longitude wrap-around problem. Errors are haversine km.
- 10 seeds (different random country-to-fold assignments). Results are mean ± 95% CI. Seeds vary only the split.
- Best median error: bare name 3,845 ± 42 km, "{city} is a city in" 3,654 ± 50 km, "I traveled to {city}" 3,490 ± 44 km.
- Controls: guess-the-mean 6,234 ± 201 km, shuffled labels 6,334 ± 167 km.
- **Language shortcut:** a baseline that predicts location from the country's primary language alone (no GPT-2) gets 4,058 ± 129 km. The bare-name probe is only ~200 km better, so much of what the probe reads is language identity. Context prompts widen the gap to ~570 km.
- A probe can decode primary language from hidden states (accuracy 0.33 vs 0.13 majority class). This shows language is present in the hidden states, not that the probe relies on it.

![v2 layers](results/v2_layers.png)

## Update 2: scale, spelling baseline, demo

- **Scale:** with the "I traveled to {city}" prompt, best-layer median error is 3,497 ± 78 km for GPT-2 small (124M) and 2,840 ± 79 km for GPT-2 medium (355M). This is a two-point comparison, and each model's best layer is chosen on the test folds, which slightly favors the larger model.
- **Spelling baseline:** a TF-IDF character n-gram model on the city name alone (no GPT-2) gets 4,423 ± 66 km, compared with 3,845 ± 42 km for the bare-name GPT-2 probe and 4,058 ± 129 km for the language-only baseline. Spelling carries much of the signal, but the GPT-2 probes are still clearly better.
- **Oceania:** the 3D probe did not fix it (~15,700 km). My guess is that few training countries are near it, but I have not tested that.
- **Demo:** the [interactive map](https://aryannnnn2.github.io/latent-atlas/) shows true locations coloured by error, with the 40 worst predictions drawn as lines from the true to the predicted location.

![scale](results/scale.png)

## Related work

This project follows Gurnee & Tegmark (2023), *Language Models Represent Space and Time* (arXiv:2310.02207), which probes LLM hidden states for place coordinates. Probes follow Alain & Bengio (2016); the shuffled-label control follows Hewitt & Liang (2019).

## What this repo does

This is a small, CPU-scale replication on GPT-2 small and medium. Its setup choices are:

- evaluation on held-out countries, with several random country-to-fold assignments and confidence intervals;
- a check showing that "City, Country" inputs leak the answer through the country token;
- two baselines that use no hidden states (primary language, and character n-grams of the name).

## Limitations

- Small models (GPT-2 small and medium), 3,000 cities.
- Seeds vary only the country-to-fold split, so the confidence intervals reflect split variability, not model or data variability.
- The best layer is chosen on the same test folds, which slightly favors models with more layers. The scale result has only two points.
- "Primary language" is a crude proxy for language family. The language-only baseline was not tuned, and the character n-gram baseline's regularization was tuned on the test folds.
- GPT-2 medium features were stored as float16 to save disk space.
- Few training examples near isolated regions (Oceania), so errors there are large.

## Bug story

Namibia's country code is "NA", which pandas silently reads as missing data. This crashed the country split until I added `keep_default_na=False`.

## Run it

```bash
pip install -r requirements.txt plotly
python extract_city.py
python probe_city.py
python extract_templates.py
python probe_v2.py
python probe_ngram.py
python extract_model.py gpt2-medium
python probe_scale.py
python make_demo.py
```

`common.py` holds shared helper functions. `extract.py` and `probe.py` run the "City, Country" version.
