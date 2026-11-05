Latent Atlas: does GPT-2 have a mental map of the world?

I trained linear probes on GPT-2’s hidden states to predict the location of 3,000 cities. The test only uses countries the probe never saw during training.

Headline result: with a 3D probe on GPT-2’s final layer, median location error is ~3,500 km (prompt “I traveled to {city}”) and ~3,850 km from the bare city name, versus ~6,250 km for guessing the average. A baseline that uses only each country’s primary language gets ~4,060 km, so much of the signal is language identity.

Why this matters

GPT-2 only ever read text, and nobody told it where cities are. If a simple linear model can read location out of its hidden states, the model has picked up some geography from language alone. The language-only baseline below shows how much of that is just recognizing a name’s language.

error per layer
real vs imagined
error per continent
How it works
Take the 3,000 most populous cities (from geonamescache).
Feed each city name to GPT-2 and save the hidden state of the last token at every layer.
For each layer, train a Ridge regression to predict location.
Evaluate with GroupKFold-style splits by country, so test countries are fully unseen. A random split would let the probe memorize “India means this area”.
Controls: a probe trained on shuffled labels, an “always guess the average” baseline, and a language-only baseline.
Findings
Location information builds up with depth: with the first (lat/lon) probe, error falls from ~4,930 km at layer 0 to ~4,120 km at layer 12. The gain is gradual, not a sharp peak in the middle layers.
A first experiment used “City, Country” as input. The last token was then the country, so every city in a country got nearly the same vector. Error was ~2,260 km at every layer, which only shows that country word embeddings encode geography. Switching to the city name alone gave the more honest test.
Best on Europe and Asia, worst on Oceania (~15,700 km).
Worst cities: Townsville, Melbourne, Sydney, and many Philippine cities (17,000-19,000 km off with the first probe).
Update: 3D probe, prompts, seeds, language control
Probes predict a 3D unit vector, which removes the longitude wrap-around problem. Errors are haversine km.
10 seeds (different random country-to-fold assignments). Results are mean ± 95% CI. Seeds vary only the split.
Best median error: bare name 3,845 ± 42 km, “{city} is a city in” 3,654 ± 50 km, “I traveled to {city}” 3,490 ± 44 km.
Controls: guess-the-mean 6,234 ± 201 km, shuffled labels 6,334 ± 167 km.
Language shortcut: a baseline that predicts location from the country’s primary language alone (no GPT-2) gets 4,058 ± 129 km. The bare-name probe is only ~200 km better, so much of what the probe reads is language identity. Context prompts widen the gap to ~570 km.
A probe can decode primary language from hidden states (accuracy 0.33 vs 0.13 majority class). This shows language is present in the hidden states, not that the probe relies on it.
v2 layers
Related work

This project follows Gurnee & Tegmark (2023), Language Models Represent Space and Time (arXiv:2310.02207), which probes LLM hidden states for place coordinates. Probes follow Alain & Bengio (2016); the shuffled-label control follows Hewitt & Liang (2019). This repo is a small-scale replication on GPT-2 small. It adds a country-held-out split, a check that “City, Country” inputs leak the answer through the country token, and a language-only baseline.

Limitations
Small model (GPT-2 small), 3,000 cities.
Seeds vary only the country-to-fold split, so the confidence intervals reflect split variability, not model or data variability.
“Primary language” is a crude proxy for language family, and the language-only baseline was not tuned.
Few training examples near isolated regions (Oceania), so errors there are large.
Bug story

Namibia’s country code is “NA”, which pandas silently reads as missing data. This crashed the country split until I added keep_default_na=False.

Run it
bash
pip install -r requirements.txt
python extract_city.py
python probe_city.py
python extract_templates.py
python probe_v2.py

extract.py and probe.py run the “City, Country” version.