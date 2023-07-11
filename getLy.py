import lyricwikia
song_name = "The A Team"
artist_name = "Ed Sheeran"
def get_lyrics(song_name, artist_name):
	lyrics = lyricwikia.get_lyrics(artist_name, song_name)
    #print(lyrics)
	f= open("lyrics_got.txt","w")
	f.write(lyrics)
	f.close()

get_lyrics(song_name, artist_name)
