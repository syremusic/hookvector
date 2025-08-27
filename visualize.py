import json
import numpy as np
import umap
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import seaborn as sns
import pandas as pd
from itertools import islice

with open("data/all_lyrics.json", "r") as f:
	data = json.load(f)

lyrics_list = []
artist_list = []
for artist, lyrics_array in islice(data.items(), 2):
	for lyric in lyrics_array:
		lyrics_list.append(lyric)
		artist_list.append(artist)

model = SentenceTransformer("all-mpnet-base-v2")
X = model.encode(lyrics_list, show_progress_bar=True)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
X_umap = reducer.fit_transform(X)

df = pd.DataFrame({
	"UMAP1": X_umap[:,0],
	"UMAP2": X_umap[:,1],
	"artist": artist_list
})

plt.figure(figsize=(12,8))
sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="artist", palette="tab20", alpha=0.7)
plt.title("Lyrics Embeddings by Artist")
plt.show()