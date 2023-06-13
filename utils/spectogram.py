from matplotlib import pyplot as plt
import librosa
import os
import re
from PIL import Image
from dataset_loader import get_metadata, load_songs
from file_handler import get_extension, create_if_not_exists
from constants import SPECTOGRAM_IMAGAES_DIR_NAME

""" 
    Create spectogram based on mp3 file

    filepath -> path to mp3 file
    output -> image in spectogram_images directory 
"""
def create_spectogram(filepath):
    create_if_not_exists(SPECTOGRAM_IMAGAES_DIR_NAME)
    songs = load_songs()
    tracks_id_array, tracks_genre_array = get_metadata()
    counter = 0
    for song in songs:
        song_file_name = get_extension(song)
        track_index = list(tracks_id_array).index(int(song_file_name))
        if(str(tracks_genre_array[track_index, 0]) != '0'):
            y, sr = librosa.load(song)
            melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
            mel = librosa.power_to_db(melspectrogram_array)
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = float(mel.shape[1]) / float(100)
            fig_size[1] = float(mel.shape[0]) / float(100)
            plt.rcParams["figure.figsize"] = fig_size
            plt.axis('off')
            plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
            librosa.display.specshow(mel, cmap='gray_r')
            plt.savefig(f"{SPECTOGRAM_IMAGAES_DIR_NAME}/{str(counter)}_{str(tracks_genre_array[track_index,0])}.jpg", bbox_inches=None, pad_inches=0)
            plt.close()
            counter = counter + 1
    return None

create_spectogram("123")


