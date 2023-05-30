from constants import DATASET_DIR_NAME, DATASET_METADATA_DIR_NAME
import os
from pathlib import Path

""" 
    Check if directory or file exists
"""
def is_path(path):
    search_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
    return os.path.exists(search_path)

""" 
    Check if directory exists, create if not
"""
def create_if_not_exists(directory):
    if not is_path(directory):
        os.makedirs(directory)

"""
    List directories in given path
"""
def list_dir(directory):
    banned = ['.gitignore', 'metadata']

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