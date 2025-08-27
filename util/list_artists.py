import sqlite3

def list_top_artists(db_path, limit=10):
	print(f"Listing {limit} artist(s) from {db_path} ...")
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	query = """
		SELECT artist_name_lower
		FROM tracks
		WHERE artist_name_lower IS NOT NULL AND artist_name_lower != 'various artists'
		GROUP BY artist_name_lower
		ORDER BY COUNT(*) DESC
		LIMIT ?;
	"""
	cursor.execute(query, (limit,))
	results = [row[0] for row in cursor.fetchall()]

	conn.close()
	return results