import os

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
