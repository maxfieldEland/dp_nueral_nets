# (setq python-shell-interpreter "~/python-environments/data-science/bin/python")
#
# mnist_utils.py
# Written by Nicholas Hanoian on 11/22/2019
"""Utilities to download and load the mnist dataset from Yann Lecun's
website: http://yann.lecun.com/exdb/mnist. Default data directory is
data

Usage:
from mnist_utils import load_mnist
train_images, train_labels, test_images, test_labels = load_mnist()

"""

import urllib.request
import os
import gzip
import numpy as np



# shouldn't be changed because these are the names of the files we
# download from the website
filenames = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz",  "t10k-labels-idx1-ubyte.gz"]

# change if you want
default_data_dir = "data"

def download_mnist(data_dir, filenames):
    """download mnist dataset from http://yann.lecun.com/exdb/mnist/, putting all files in data_dir"""

    root_url = "http://yann.lecun.com/exdb/mnist/"
    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        for filename in filenames:
            response = urllib.request.urlopen(root_url + filename)
            with open(os.path.join(data_dir, filename), "wb") as f:
                f.write(response.read())
        return True
    except Exception as e:
        print("Error:", e)
        return False

def load_idx_file(filename):
    """Read in an idx according to file format specifications from bottom
    of http://yann.lecun.com/exdb/mnist/. We only handle the case of
    the datatype being unsigned bytes, as this is the datatype for
    both image files and label files in mnist. The datatype of the
    data in an idx file is located in the header, and the code for
    unsigned bytes is \x08. Hence, if we find something other than
    this, we throw an error. Returns a numpy array of the data
    contained within the file.
    """

    ubyte_code = b"\x08" # indicates that the file contains unsigned bytes


    with gzip.open(filename, "r") as f:
        # read in "magic number"
        f.read(2) # empty
        datatype = f.read(1)
        n_dims = int.from_bytes(f.read(1), "big")

        # check that the datatype is what we expect
        if datatype != ubyte_code:
            print(f"Error! Expected datatype code to be {ubyte_code} for unsigned byte, but was {datatype}")
            return False

        # now we know number of dims, so figure out the size of those dims
        dims = []
        for i in range(n_dims):
            dims.append(int.from_bytes(f.read(4), "big"))

        # read in (product of all dimension sizes) number of bytes
        # each data point is one byte, so we dont need to multiply by
        # the number of bytes per data point
        buff = f.read(np.product(dims))

        # convert each data point to an integer, and reshape according to the given dims
        data = np.frombuffer(buff, dtype=np.ubyte).reshape(dims)

        return data





def load_mnist(data_dir=default_data_dir, filenames=filenames):
    """Check to see if MNIST files exist, and then load them in and return
    numpy arrays. If they do not exist, download them."""



    # check if all required files are found
    paths = [os.path.join(data_dir, filename) for filename in filenames]
    if sum([os.path.exists(path) for path in paths]) / len(paths) != 1:
        # one or more not found. download all files
        print(f"MNIST data not found. Downloading data to {data_dir}/")
        success = download_mnist(data_dir, filenames)
        if not success: # if download unsuccessful
            print("Unable to download MNIST.")
            return False

    # load in data from files
    train_images = load_idx_file(paths[0])
    train_labels = load_idx_file(paths[1])
    test_images  = load_idx_file(paths[2])
    test_labels  = load_idx_file(paths[3])

    return train_images, train_labels, test_images, test_labels
