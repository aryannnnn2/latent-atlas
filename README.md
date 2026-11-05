\# Latent Atlas: does GPT-2 have a mental map of the world?



I trained simple linear probes on GPT-2's hidden states to predict the latitude and longitude of 3,000 cities, testing only on \*\*countries the probe never saw during training\*\*.



\*\*Headline result:\*\* from the city name alone, a probe on GPT-2's final layer predicts location with a median error of \*\*~4,100 km\*\*, versus \*\*~6,350 km\*\* for guessing the average location and \*\*~6,300 km\*\* for a probe trained on shuffled labels.



!\[error per layer](results/city\_error\_per\_layer.png)

!\[real vs imagined](results/city\_real\_vs\_imagined.png)

!\[error per continent](results/city\_error\_per\_continent.png)



\## How it works

1\. Take the 3,000 most populous cities (from `geonamescache`).

2\. Feed each city name to GPT-2 and save the hidden state of the last token at every layer.

3\. For each layer, train a Ridge regression to predict (lat, lon).

4\. Evaluate with \*\*GroupKFold by country\*\*, so test countries are fully unseen. A random split would let the probe memorize "India means this area".

5\. Controls: a probe trained on shuffled labels, and a "always guess the average" baseline.



\## Findings

\- Location information builds up with depth: error falls from ~4,930 km (layer 0) to ~4,120 km (layer 12). The gain is gradual, not a sharp peak in the middle layers.

\- A first experiment used "City, Country" as input. The last token was then the country, so every city in a country got nearly the same vector. Error was ~2,260 km at every layer, which only shows that word embeddings of countries encode geography. Switching to the city name alone gave the more honest test above.

\- Best on Asia (~3,270 km) and Europe (~3,650 km), worst on Oceania (~15,400 km).

\- Worst cities: Townsville, Melbourne, Sydney, and many Philippine cities (17,000-19,000 km off).



\## Limitations

\- Small model (GPT-2 small), 3,000 cities.

\- Linear probe on raw lat/lon, which ignores longitude wrap-around. Far-flung regions (Oceania) get pulled toward the middle of the map. Predicting 3D unit vectors would be better.

\- Few training examples near isolated regions, so errors there are large.

\- Single run, no confidence intervals.



\## Bug story

Namibia's country code is "NA", which pandas silently reads as missing data. This crashed the country split until I added `keep\_default\_na=False`.



\## Run it

```

pip install -r requirements.txt

python extract\_city.py

python probe\_city.py

```

(`extract.py` and `probe.py` run the "City, Country" version.)

