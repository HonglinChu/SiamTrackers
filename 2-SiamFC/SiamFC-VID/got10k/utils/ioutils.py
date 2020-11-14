from __future__ import absolute_import, division

import wget
import os
import shutil
import zipfile
import sys


def download(url, filename):
    r"""Download file from the internet.
    
    Args:
        url (string): URL of the internet file.
        filename (string): Path to store the downloaded file.
    """
    return wget.download(url, out=filename)


def extract(filename, extract_dir):
    r"""Extract zip file.
    
    Args:
        filename (string): Path of the zip file.
        extract_dir (string): Directory to store the extracted results.
    """
    if os.path.splitext(filename)[1] == '.zip':
        if not os.path.isdir(extract_dir):
            os.makedirs(extract_dir)
        with zipfile.ZipFile(filename) as z:
            z.extractall(extract_dir)
    else:
        raise Exception('Unsupport extension {} of the compressed file {}.'.format(
            os.path.splitext(filename)[1]), filename)


def compress(dirname, save_file):
    """Compress a folder to a zip file.
    
    Arguments:
        dirname {string} -- Directory of all files to be compressed.
        save_file {string} -- Path to store the zip file.
    """
    shutil.make_archive(save_file, 'zip', dirname)
