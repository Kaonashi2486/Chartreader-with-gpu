import os
import numpy as np

class BASE():
    def __init__(self):
        self._split = None
        self._db_inds = []
        self._image_ids = []

        self._data = "chart"
        self._image_file = None
        # Stores the mean of image data for normalization.
        self._eig_val = np.ones((3, ), dtype=np.float32)
        # Stores eigen vectors
        self._eig_vec = np.zeros((3, 3), dtype=np.float32)

        self._mean = np.zeros((3, ), dtype=np.float32)
        # Stores standard deviation for normalization
        self._std = np.ones((3, ), dtype=np.float32)

        self._configs = {}

        self._data_rng = None

    @property
    def data(self):
        if self._data is None:
            raise ValueError("Data is not set")
        return self._data

    @property
    def configs(self):
        return self._configs

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def eig_val(self):
        return self._eig_val

    @property
    def eig_vec(self):
        return self._eig_vec

    @property
    def db_inds(self):
        return self._db_inds

    @property
    def split(self):
        return self._split

    def update_config(self, new):
        """ Update the config dictionary with new values. """
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

    def image_ids(self, ind):
        """ Return the image id at the specified index. """
        return self._image_ids[ind]

    def image_file(self, ind):
        """ Get the image file path for a given index. """
        if self._image_file is None:
            raise ValueError("Image path is not initialized")
        image_id = self._image_ids[ind]
        # Format the image file path based on image ID
        return self._image_file.format(image_id)

    def evaluate(self, name):
        """ Placeholder for evaluation function. """
        pass

    def shuffle_inds(self, quiet=False):
        """ Shuffle database indices. """
        if self._data_rng is None:
            # Use os.getpid() for random seed, works for both Windows and Linux
            self._data_rng = np.random.RandomState(os.getpid())

        if not quiet:
            print("Shuffling...")
        rand_perm = self._data_rng.permutation(len(self._db_inds))
        self._db_inds = np.array(self._db_inds)[rand_perm]
