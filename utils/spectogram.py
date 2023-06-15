from utils.constants import GENRES, SPECTOGRAM_IMAGAES_DIR_NAME, SPECTOGRAM_SPLIT_DIR_NAME, TEST_SPECTOGRAM_IMAGAES_DIR_NAME, TEST_SPECTOGRAM_SPLIT_DIR_NAME, TRAINING_DATA_DIR_NAME, TEST_DATASET_DIR_NAME, TEST_SPECTOGRAM_IMAGAES_DIR_NAME, MEL_SPECTOGRAM_BANDS_SIZE
from utils.file_handler import get_song_name, create_if_not_exists, is_path
from utils.dataset_loader import get_genre, get_index, get_metadata, load_images, load_songs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.utils import np_utils
import librosa.display
from PIL import Image
import numpy as np
import librosa
import os
import cv2

""" 
    Create spectogram image based on mp3 file

    song -> path to mp3 file
    filename -> full path filename where spectogram should be saved eg '/specgoram_images/name.jpg'
"""
def create_spectogram_image(song, filename):
    y, sr = librosa.load(song)
    spectogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=MEL_SPECTOGRAM_BANDS_SIZE, fmax=8000)
    mel = librosa.power_to_db(spectogram)
    save_mel_fig(mel, filename)


"""
    Create spectograms for test purposes
"""
def create_test_spectograms():
    create_if_not_exists(TEST_SPECTOGRAM_IMAGAES_DIR_NAME)
    songs = load_songs(TEST_DATASET_DIR_NAME)
    for song in songs:
        song_name = get_song_name(song)
        filename = f'{TEST_SPECTOGRAM_IMAGAES_DIR_NAME}/{song_name}.jpg'
        create_spectogram_image(song, filename)
    return None


"""
    Create spectograms for train purposes
"""
def create_spectograms():
    create_if_not_exists(SPECTOGRAM_IMAGAES_DIR_NAME)
    songs = load_songs()
    tracks_id_array, tracks_genre_array = get_metadata()
    counter = 0
    for song in songs:
        song_file_name = get_song_name(song)
        track_index = list(tracks_id_array).index(int(song_file_name))
        if (str(tracks_genre_array[track_index, 0]) != '0'):
            filename = f"{SPECTOGRAM_IMAGAES_DIR_NAME}/{str(counter)}_{str(tracks_genre_array[track_index,0])}.jpg"
            create_spectogram_image(song, filename)
        counter = counter + 1
    return None


"""
    Split spectogram into chunks
"""
def create_split_spectogram(image, filename):
    img = Image.open(image)
    width, _ = img.size
    number_of_samples = int(width / MEL_SPECTOGRAM_BANDS_SIZE)
    for i in range(number_of_samples):
        start = i*MEL_SPECTOGRAM_BANDS_SIZE
        img_temporary = img.crop(
            (start, 0., start + MEL_SPECTOGRAM_BANDS_SIZE, MEL_SPECTOGRAM_BANDS_SIZE))
        img_temporary.save(filename)
    return

"""
    Split spectogram image into smaller parts for test
"""
def split_test_spectogram():
    create_if_not_exists(TEST_SPECTOGRAM_SPLIT_DIR_NAME)
    images = load_images(TEST_SPECTOGRAM_IMAGAES_DIR_NAME)
    counter = 0
    for image in images:
        song_name = get_song_name(image)
        filename = f'{TEST_SPECTOGRAM_SPLIT_DIR_NAME}/{counter}_{song_name}.jpg'
        create_split_spectogram(image, filename)
        counter = counter + 1

"""
    Split spectogram image into smaller parts for training
"""

def split_spectogram():

    create_if_not_exists(SPECTOGRAM_SPLIT_DIR_NAME)
    images = load_images(SPECTOGRAM_IMAGAES_DIR_NAME)
    counter = 0
    for image in images:
        genre = get_genre(image)
        filename = f'{SPECTOGRAM_SPLIT_DIR_NAME}/{counter}_{genre}.jpg'
        create_split_spectogram(image, filename)
        counter = counter + 1
    return


""" 
    Save spectogram image

    mel -> spectogram in db units array. Output of librosa.power_to_db
    figname -> Name of image (full path)
"""
def save_mel_fig(mel, figname):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = float(mel.shape[1]) / float(100)
    fig_size[1] = float(mel.shape[0]) / float(100)
    plt.rcParams["figure.figsize"] = fig_size
    plt.axis('off')
    plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(mel, cmap='gray_r')
    plt.savefig(figname, bbox_inches=None, pad_inches=0)
    plt.close()

"""
    Load split images for test purposes
"""
def load_splited_images():
    images_filenames = load_images(TEST_SPECTOGRAM_IMAGAES_DIR_NAME)
    images = []
    labels = []
    for f in images_filenames:
        song_variable = get_song_name(f)
        tempImg = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        images.append(cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY))
        labels.append(song_variable)

    images = np.array(images)

    return images, labels


"""
    Converts images and labels into training and testing matrices.
    Loads from file if TRAINING_DATA_DIR_NAME directory exists
"""
def load_dataset(datasetSize=0.7):

    images_all, labels_all = get_images_and_labels()
    
    images = []
    labels = []

    n_classes = len(GENRES)
    count_max = int(len(images_all) * datasetSize / n_classes)
    count_array = [0 for _ in range(n_classes)]

    for i in range(0, len(images_all)):
        if (count_array[labels_all[i]] < count_max):
            images.append(images_all[i])
            labels.append(labels_all[i])
            count_array[labels_all[i]] += 1

    images = np.array(images)
    labels = np.array(labels)

    images = np.array(images)
    labels = np.array(labels)

    labels = labels.reshape(labels.shape[0], 1)

    train_x, test_x, train_y, test_y = train_test_split( images, labels, test_size=1.0 - datasetSize, shuffle=True)

    train_y = np_utils.to_categorical(train_y)
    test_y = np_utils.to_categorical(test_y, num_classes=len(GENRES))
    genre_new = {value: key for key, value in GENRES.items()}

    if is_path(TRAINING_DATA_DIR_NAME):
        train_x, train_y, test_x, test_y = load_training_output()
        return train_x, train_y, test_x, test_y, n_classes, genre_new

    save_training_output(train_x, train_y, test_x, test_y)
    return train_x, train_y, test_x, test_y, n_classes, genre_new

"""
    Returns images and labels from SPECTOGRAM_SPLIT_DIR_NAME directory
"""
def get_images_and_labels():
    images = load_images(SPECTOGRAM_SPLIT_DIR_NAME)
    images_all = [None for _ in range(len(images))]
    labels_all = [None for _ in range(len(images))]

    for image in images:
        filename = get_song_name(image)
        index = get_index(filename)
        genre = get_genre(image)

        temp = cv2.imread(image, cv2.IMREAD_UNCHANGED)

        images_all[index] = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        labels_all[index] = GENRES[genre]

    return images_all, labels_all

"""
    Save training matrices to npy files in TRAINING_DATA_DIR_NAME
"""

def save_training_output(train_x, train_y, test_x, test_y, directory=TRAINING_DATA_DIR_NAME):
    create_if_not_exists(directory)
    np.save(f'{TRAINING_DATA_DIR_NAME}/train_x.npy', train_x)
    np.save(f'{TRAINING_DATA_DIR_NAME}/train_y.npy', train_y)
    np.save(f'{TRAINING_DATA_DIR_NAME}/test_x.npy', test_x)
    np.save(f'{TRAINING_DATA_DIR_NAME}/test_y.npy', test_y)

"""
    Loads  training matrices (npy files) from directory
"""
def load_training_output(directory=TRAINING_DATA_DIR_NAME):
    if is_path(directory):
        train_x = np.load(f'{TRAINING_DATA_DIR_NAME}/train_x.npy')
        train_y = np.load(f'{TRAINING_DATA_DIR_NAME}/train_y.npy')
        test_x = np.load(f'{TRAINING_DATA_DIR_NAME}/test_x.npy')
        test_y = np.load(f'{TRAINING_DATA_DIR_NAME}/test_y.npy')
        return train_x, train_y, test_x, test_y
