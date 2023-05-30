from constants import DATASET_DIR_NAME
from file_handler import is_path
import pandas as pd


""" 
Get metadata (genre, track id) from metadata/tracks.csv file
"""
def get_metadata(path = f"{DATASET_DIR_NAME}/metadata/tracks.csv"):
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
Load each 30s music file and turn it into mel-spectograms using spectogram.py
"""

def load_data():
    return True