import tkinter as tk
from tkinter import messagebox
import lyricsgenius


def get_lyrics(api_token, artist, song):
    genius = lyricsgenius.Genius(api_token)

    try:
        song_lyrics = genius.search_song(song, artist)

        if song_lyrics:
            return song_lyrics.lyrics
        else:
            return "Lyrics not found."
    except Exception as e:
        return f"Error: {e}"


def extract_lyrics():
    api_token = "niUvdmT6FJ8qMrHPBP--nIFcG6ubjerMDumSzq43IlSAgWs3bKwpxItRcRlK4OAM"  # Replace with your Genius API token
    artist = artist_entry.get()
    song = song_entry.get()

    if not api_token or not artist or not song:
        messagebox.showinfo("Error", "https://genius.com/, artist, and song.")
        return

    lyrics = get_lyrics(api_token, artist, song)
    with open(str(song) + ".txt", 'a', encoding='UTF8') as file:
        file.write(lyrics)

    result_text.delete(1.0, tk.END)  # Clear previous results
    result_text.insert(tk.END, lyrics)


# Create main window
window = tk.Tk()
window.title("Flood-Lyrics Extractor")

# Set window size and position it in the center of the screen
window_width = 500
window_height = 600
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2
window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Set window background color to baby pink
window.configure(bg="#FFC0CB")  # Hex color for baby pink

# Create and place widgets
tk.Label(window, text="Artist:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
tk.Label(window, text="Song:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)

artist_entry = tk.Entry(window)
song_entry = tk.Entry(window)
artist_entry.grid(row=0, column=1, padx=10, pady=10)
song_entry.grid(row=1, column=1, padx=10, pady=10)

# Set button color to blue
extract_button = tk.Button(window, text="Extract Lyrics", command=extract_lyrics, bg="#0000FF", fg="white")
extract_button.grid(row=2, column=0, columnspan=2, pady=10)

# Make lyrics box longer in height and center it
result_text = tk.Text(window, height=20, width=40)
result_text.grid(row=3, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")

# Configure row and column weights to center the lyrics box
window.grid_rowconfigure(3, weight=1)
window.grid_columnconfigure(0, weight=1)

# Start the GUI main loop
window.mainloop()
