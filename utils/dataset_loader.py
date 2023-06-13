from constants import DATASET_DIR_NAME, DATASET_METADATA_DIR_NAME
from file_handler import is_path, list_dir
import pandas as pd
import os

""" 
Get metadata (genre, track id) from metadata/tracks.csv file
"""
def get_metadata(path = f"{DATASET_DIR_NAME}/{DATASET_METADATA_DIR_NAME}/tracks.csv"):
    if not is_path(path):
        raise ValueError(f"No metadata file found! Download one and provide valid path! Searching for: {path}")
    tracks = pd.read_csv(path, header = 2, low_memory = False)
    tracks_array = tracks.values
    tracks_id_array = tracks_array[: , 0]
    tracks_genre_array = tracks_array[: , 40]
    tracks_id_array = tracks_id_array.reshape(tracks_id_array.shape[0], 1)
    tracks_genre_array = tracks_genre_array.reshape(tracks_genre_array.shape[0], 1)
    return (tracks_id_array, tracks_genre_array)

"""
    Iterate over every directory in data and return list of songs.
    Name of the directory is label, which can be decoded using metadata 
"""

def load_songs():
    songs = []
    songs_directories = list_dir(DATASET_DIR_NAME)
    for songs_label_directory in songs_directories:
        song_label_directory_path = f"{DATASET_DIR_NAME}/{songs_label_directory}"
        files = [os.path.abspath(os.path.join(song_label_directory_path, f)) for f in list_dir(song_label_directory_path)]
        songs.extend(files)

    return songs[:20]
""" 
Load each 30s music file and turn it into mel-spectograms using spectogram.py
"""

def load_data():
    print(load_songs())

load_data()