import sqlite3
import hashlib
import re
import unicodedata

MULTISPACE_RE = re.compile(r"[ \t]+")
PUNCTUATION_RE = re.compile(r"[^\w\s',]")

QUOTE_MAP = str.maketrans({
	"’": "'",
	"‘": "'",
	"“": '"',
	"”": '"',
})

def normalize_lyrics(lyrics):
	if not lyrics:
		return ""

	# Split into lines and strip empty ones
	lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
	text = "\n".join(lines)

	# Lowercase
	text = text.lower()

	# Normalize quotes
	text = text.translate(QUOTE_MAP)

	# Remove accents (normalize unicode)
	text = unicodedata.normalize("NFKD", text)
	text = text.encode("ascii", "ignore").decode("utf-8")

	# Remove all punctuation except apostrophes and commas
	text = PUNCTUATION_RE.sub("", text)

	# Collapse multiple spaces/tabs into single spaces, but preserve newlines
	text = "\n".join(MULTISPACE_RE.sub(" ", line) for line in text.splitlines())

	return text.strip()

def extract_artist_lyrics(db_path, artist, max_songs=10):
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	seen_hashes = set()
	unique_lyrics = []

	query = """
		SELECT lyrics.plain_lyrics
		FROM lyrics
		JOIN tracks ON lyrics.track_id = tracks.id
		WHERE tracks.artist_name_lower = ?
		AND lyrics.plain_lyrics IS NOT NULL
	"""
	cursor.execute(query, (artist.lower(),))
	rows = cursor.fetchall()

	for row in rows:
		lyrics_text = row[0]
		normalized = normalize_lyrics(lyrics_text)
		lyrics_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

		if lyrics_hash in seen_hashes:
			continue

		seen_hashes.add(lyrics_hash)
		unique_lyrics.append(normalized)

		if len(unique_lyrics) >= max_songs:
			break
	
	return unique_lyrics