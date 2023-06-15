from pydub import AudioSegment
import os
from pathlib import Path

def is_path(path):
    search_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
    return os.path.exists(search_path)

def list_dir(directory):
    banned = ['.gitignore', 'metadata', 'split_songs.py', 'ffmpeg.exe', 'ffplay.exe', 'ffprobe.exe']

    if not is_path(directory):
        raise ValueError(f"Directory {directory} does not exist!")
    
    directories = os.listdir(directory)
    for name in banned:
        if name in directories:
            directories.remove(name)
    return directories

"""
    Get file name (without extension) from absolute path
"""

def get_extension(path):
    extension =  Path(path).stem
    return extension

for song in list_dir('C:\\Users\\KrzysztofKrawczyk\\Desktop\\Research\\Music-Recommendation-CNN\\test_data'):
    song_path = os.path.normpath(f'C:\\Users\\KrzysztofKrawczyk\\Desktop\\Research\\Music-Recommendation-CNN\\test_data\\{song}')
    song_name = str(song)
    song = AudioSegment.from_mp3(song_path)

    ten_seconds = 30 * 1000
    
    first_10_seconds = song[:ten_seconds]

    first_10_seconds.export(f'30s_{song_name}', format="mp3")