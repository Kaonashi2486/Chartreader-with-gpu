import os
import numpy as np
import pickle
import sys
# Add the path to your config.py to sys.path
sys.path.append(r'C:\Users\saksh\OneDrive\Desktop\stuffs\Chartreader-with-gpu\db\datasets')
from db.datasets import dataset

class Config:
    def __init__(self):
        self._configs = {}
        self._configs["dataset_name"] = "Chart"  # Default dataset name
        self._configs["dataset"] =None
        self._configs["testing_function"] = None
        # Training Config
        self._configs["stepsize"] = 450000
        self._configs["learning_rate"] = 0.00025
        self._configs["decay_rate"] = 10
        self._configs["max_iter"] = 500000
        self._configs["val_iter"] = 100
        self._configs["batch_size"] = 2
        self._configs["snapshot_name"] = None
        self._configs["pretrain"] = None
        self._configs["opt_algo"] = "adam"

        # Directories (Use os.path.join for cross-platform compatibility)
        self._configs["data_dir"] = os.path.join(".", "data")
        self._configs["cache_dir"] = os.path.join(self._configs["data_dir"], "cache")
        self._configs["config_dir"] = os.path.join(".", "config")

        # Split
        self._configs["train_split"] = "trainval"
        self._configs["val_split"] = "minival"
        self._configs["test_split"] = "testdev"

        # Rng
        self._configs["data_rng"] = np.random.RandomState(123)

    @property
    def train_split(self):
        return self._configs["train_split"]

    @property
    def val_split(self):
        return self._configs["val_split"]

    @property
    def test_split(self):
        return self._configs["test_split"]

    @property
    def full(self):
        return self._configs

    @property
    def testing_function(self):
        return self._configs["testing_function"]

    @property
    def data_rng(self):
        return self._configs["data_rng"]

    @property
    def opt_algo(self):
        return self._configs["opt_algo"]

    @property
    def pretrain(self):
        return self._configs["pretrain"]

    @property
    def dataset(self):
        return self._configs["dataset"]

    @property
    def snapshot_name(self):
        return self._configs["snapshot_name"]

    @property
    def snapshot_dir(self):
        snapshot_dir = os.path.join(self.cache_dir, "nnet", self.snapshot_name)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        return snapshot_dir

    @property
    def snapshot_file(self):
        snapshot_file = os.path.join(self.snapshot_dir, f"{self.snapshot_name}_{{}}.pkl")
        return snapshot_file

    @property
    def config_dir(self):
        return self._configs["config_dir"]

    @property
    def batch_size(self):
        return self._configs["batch_size"]

    @property
    def max_iter(self):
        return self._configs["max_iter"]

    @property
    def learning_rate(self):
        return self._configs["learning_rate"]

    @property
    def decay_rate(self):
        return self._configs["decay_rate"]

    @property
    def stepsize(self):
        return self._configs["stepsize"]

    @property
    def val_iter(self):
        return self._configs["val_iter"]

    @property
    def data_dir(self):
        return self._configs["data_dir"]

    @property
    def cache_dir(self):
        if not os.path.exists(self._configs["cache_dir"]):
            os.makedirs(self._configs["cache_dir"])
        return self._configs["cache_dir"]

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

    @property
    def dataset_name(self):
        return self._configs["dataset_name"]

    @dataset_name.setter
    def dataset_name(self, value):
        self._configs["dataset_name"] = value

    def initialize_dataset(self):
        """
        Initialize the dataset after the configuration is loaded.
        This breaks the circular import.
        """
        from db.datasets import get_dataset
        self._configs["dataset"] = get_dataset(self._configs["dataset_name"])


# Create an instance of the Config class
system_configs = Config()


# Export (serialize) the instance to a file
with open('config_instance.pkl', 'wb') as file:
    pickle.dump(system_configs, file)
