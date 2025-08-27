from util.list_artists import list_top_artists
from util.extract import extract_artist_lyrics

INF = float('inf')

db_path = "data/lrclib-db-dump-20250823T005259Z.sqlite3"
top_artists = list_top_artists(db_path=db_path, limit=100)

for idx, artist in enumerate(top_artists, start=1):
	print(f"Getting lyrics for {artist} ({idx}/{len(top_artists)})")
	artist_lyrics = extract_artist_lyrics(db_path=db_path, artist=artist, max_songs=INF)
	
	for x in artist_lyrics:
		print(x[:100], "...\n")