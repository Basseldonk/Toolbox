#!/usr/bin/env Python3
"""
Datasets
"""
import os
import pandas as pd

from glob import glob
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """ Custom base class for pytorch datasets. """

    def __init__(self, name, data, **kwargs):
        """
        Returns a Dataset.

        Args:
            name (str):                 Name of the dataset.
            data (list):                Data in the dataset.

        Kwargs:
            split (dict, optional):     A dictionary specifying subsets. Keys are used
                                        subset names, values are used to assign a part
                                        of the available data to the subset. If values
                                        are integers, values describes the  number of
                                        images in the subset. If values are floats,
                                        the values are used as percentages.
        """
        self.name = name
        self.data = data
        self.is_superset = False
        if 'split' in kwargs.keys():
            self._split(kwargs['split'])

    def __getitem__(self, idx):
        """ Returns item with index idx. """
        if self.is_superset:
            for ds in self.data:
                if idx >= len(ds):
                    continue
                return ds[idx]
        else:
            return self.data[idx]

    def __len__(self):
        """ Returns the size of the dataset. """
        if self.is_superset:
            length = 0
            for ds in self.data:
                length += len(ds)
            return length
        else:
            return len(self.data)

    def __str__(self):
        """ Returns a string representation of the dataset. """
        desc = f'{self.name}\n' + '-' * len(self.name) + '\n'
        desc += f'Contains {len(self)} data entries.\n'
        if self.is_superset:
            for ds in self.data:
                desc += f'\tSubset {ds.name}: {len(ds)} entries.\n'
        return desc

    def _split(self, split):
        """
        Splits the data and distributes it over subsets defined in split.

        If split does not use all data, an additional subset 'rest' is created with
        the leftover data.

        Args:
            split (dict):       Dictionary defining subsets. Keys are used as subset
                                names and values indicate size of the subset. If values
                                are integers, values describes the  number of images
                                in the subset. If values are floats, the values are
                                used as percentages.
        """
        total = sum(split.values())
        # If split contains floats, convert to integers
        if isinstance(total, float):
            assert_msg = 'Not enough data! ' \
                            + f'Split requires a total of {total*100}%. ' \
                            + 'Split should not exceed 100%.'
            assert total <= 1, assert_msg
            # Add 'rest' subset if not all data is used in split
            if total < 1:
                split['rest'] = 1 - total
            split = self._float_split_to_int(split)
            total = sum(split.values())
        # Create subsets based off integer values
        if isinstance(total, int):
            assert_msg = 'Not enough data! ' \
                            + f'Split requires a total of {total} data entries ' \
                            + f'but only {len(self.data)} are available.'
            assert total <= len(self.data), assert_msg
            # Add 'rest' subset if not all data is used in split
            if total < len(self.data):
                split['rest'] = len(self.data) - total
            # Create subsets
            index = 0
            for name, length in split.items():
                subset_name = f'{self.name}.{name}'
                subset_data = self.data[index:index + length]
                setattr(self, name, Dataset(subset_name, subset_data))
                index += length
            # Replace data with references to subsets
            self.data = []
            for name in split.keys():
                self.data.append(getattr(self, name, None))
            # Indicate that this is a superset
            self.is_superset = True


    def _float_split_to_int(self, split):
        """ Converts a split with float values to a split with integers. """
        output = {}
        data_entries = len(self.data)
        subset_sizes = list(split.values())
        for i, (name, length) in enumerate(split.items()):
            output[name] = int(data_entries * (length / sum(subset_sizes[i:])))
            data_entries -= output[name]
        return output

    @classmethod
    def from_csv(cls, name, csv, **kwargs):
        """ Constructs a dataset based on the csv file. """
        data = pd.read_csv(csv, **kwargs)
        return Dataset(name, data, **kwargs)

    @classmethod
    def from_files(cls, name, folders, **kwargs):
        """ Constructs a dataset by collecting data form provided folders. """
        data = cls._collect_files(folders)
        return Dataset(name, data, **kwargs)

    @staticmethod
    def _collect_files(folders, extention='Default'):
        """
        Collects file paths for files with the correct extention in folders.

        Args:
            folders ([str]):                Paths to all folders containing folders that should be searched.
            extention ([str], optional):    String or list of strings specifying the file extentions to consider.
                                            Default is any, e.g. ['.*'].

        Returns:
            files ([str]):                  A list of paths to the found files.
        """
        if isinstance(extention, str):
            if extention.lower() == 'default':
                extention = ['.*']
            else:
                extention = [extention]
        files = []
        for f in folders:
            for e in extention:
                files += glob(os.path.join(f, f'*{e}'))
        return files
