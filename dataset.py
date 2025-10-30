"""
Dataset Management Module

This module provides a unified Dataset class for managing training and test data
with utilities for cross-validation and train/validation splitting.

The Dataset class encapsulates:
- Training data (features and labels)
- Optional test data (features only)
- K-fold cross-validation generation
- Train/validation split generation with randomization
"""
from typing import Generator
import numpy as np


class Dataset:
    """
    Container for machine learning datasets with cross-validation utilities.

    This class manages training and test data, providing convenient methods
    for generating k-fold cross-validation splits and random train/validation
    splits. All data shuffling uses NumPy's random permutation for reproducibility.

    Attributes:
        x_train: numpy array of shape (N, D). Training feature matrix.
        y_train: numpy array of shape (N,). Training labels.
        x_test: numpy array of shape (M, D) or None. Test feature matrix (labels unknown).
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, num_cont_features: int):
        """Initializes the Dataset with training and optional test data.

        Args:
            x_train: numpy array of shape (N, D). Training feature matrix where
                    N is the number of samples and D is the number of features.
            y_train: numpy array of shape (N,). Training labels corresponding to x_train.
            x_test: numpy array of shape (M, D) or None. Test feature matrix with
                   the same number of features D. Defaults to None.
        """
        x_tr, y_tr, x_te, y_te = self._split_data_arrays(x_train, y_train, ratio=0.8)
        self.x_train = x_tr
        self.y_train = y_tr
        self.x_test = x_te
        self.y_test = y_te
        self.num_cont_features = num_cont_features

    def __len__(self):
        """Returns the number of training samples in the dataset.

        Returns:
            int. The number of rows in x_train (N).
        """
        return len(self.x_train)

    def k_fold_generator(self, k: int) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.generic, np.generic], None, None]:
        """Generates k-fold cross-validation splits as a generator.

        Randomly shuffles the training data and divides it into k approximately
        equal-sized folds. For each fold, yields training and validation sets where
        the current fold becomes the validation set and the remaining k-1 folds
        become the training set.

        The data is shuffled once at the beginning using np.random.permutation,
        ensuring each sample appears exactly once in validation across all k iterations.

        Args:
            k: int. Number of folds to create. Must be >= 2.

        Yields:
            tuple of (x_train, y_train, x_val, y_val) for each fold:
                - x_train: numpy array of shape (N*(k-1)/k, D). Training features for this fold.
                - y_train: numpy array of shape (N*(k-1)/k,). Training labels for this fold.
                - x_val: numpy array of shape (N/k, D). Validation features for this fold.
                - y_val: numpy array of shape (N/k,). Validation labels for this fold.
        """
        # Calculate size of each fold (integer division)
        fold_size: int = int(len(self) / k)

        # Randomly shuffle all indices once
        indices = np.random.permutation(len(self))

        for i in range(k):
            # Define validation set range for current fold
            val_start = i * fold_size
            # Last fold takes any remaining samples if N is not divisible by k
            val_end = (i + 1) * fold_size if i < k else len(self)

            # Split indices: everything except current fold is training
            train_indices = np.concatenate((indices[:val_start], indices[val_end:]))
            val_indices = indices[val_start:val_end]

            # Extract training data (k-1 folds)
            x_train = self.x_train[train_indices].copy()
            y_train = self.y_train[train_indices]

            # Extract validation data (1 fold)
            x_val = self.x_train[val_indices].copy()
            y_val = self.y_train[val_indices]

            mean: np.generic = np.mean(x_train[:, 1:self.num_cont_features+1], axis=0)
            std: np.generic = np.std(x_train[:, 1:self.num_cont_features+1], axis=0)

            x_train[:, 1:self.num_cont_features+1] = (x_train[:, 1:self.num_cont_features+1] - mean) / std
            x_val[:, 1:self.num_cont_features+1] = (x_val[:, 1:self.num_cont_features+1] - mean) / std

            yield x_train, y_train, x_val, y_val, mean, std

    def split_data(self, ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.generic, np.generic]:
        """Randomly splits the training data into train and validation sets.

        Shuffles the entire training dataset and divides it into two subsets
        according to the specified ratio. This is useful for creating a single
        train/validation split for model development and hyperparameter tuning.

        Args:
            ratio: float in (0, 1). Proportion of data to use for training.
                  Default is 0.8 (80% training, 20% validation).

        Returns:
            tuple of (x_train, x_val, y_train, y_val):
                - x_train: numpy array of shape (N*ratio, D). Training features.
                - x_val: numpy array of shape (N*(1-ratio), D). Validation features.
                - y_train: numpy array of shape (N*ratio,). Training labels.
                - y_val: numpy array of shape (N*(1-ratio),). Validation labels.
                - mean: numpy generic. Mean of training features
                - std: numpy generic. Standard deviation of training features
        """
        x_train, y_train, x_val, y_val = self._split_data_arrays(self.x_train, self.y_train, ratio)

        mean: np.generic = np.mean(x_train[:, 1:self.num_cont_features+1], axis=0)
        std: np.generic = np.std(x_train[:, 1:self.num_cont_features+1], axis=0)

        x_train[:, 1:self.num_cont_features+1] = (x_train[:, 1:self.num_cont_features+1] - mean) / std
        x_val[:, 1:self.num_cont_features+1] = (x_val[:, 1:self.num_cont_features+1] - mean) / std

        return x_train, x_val, y_train, y_val, mean, std
    
    def get_test_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the test set features and labels.

        Returns:
            tuple of (x_test, y_test):
                - x_test: numpy array of shape (M, D). Test feature matrix.
                - y_test: numpy array of shape (M,). Test labels.
        """
        return self.x_test, self.y_test
    
    @staticmethod
    def _split_data_arrays(x_data, y_data, ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Randomly splits the training data into train and validation sets.

        Shuffles the entire training dataset and divides it into two subsets
        according to the specified ratio.

        Args:
            ratio: float in (0, 1). Proportion of data to use for training.
                  Default is 0.8 (80% training, 20% validation).

        Returns:
            tuple of (x_train, x_val, y_train, y_val):
                - x_train: numpy array of shape (N*ratio, D). Training features.
                - x_val: numpy array of shape (N*(1-ratio), D). Validation features.
                - y_train: numpy array of shape (N*ratio,). Training labels.
                - y_val: numpy array of shape (N*(1-ratio),). Validation labels.
        """
        N = np.shape(x_data)[0] # Number of datapoints
        
        # Calculate split point
        split_index = int(N * ratio)

        # Randomly shuffle all indices
        indices = np.random.permutation(N)

        # Split indices according to ratio
        train_indices = indices[:split_index]
        val_indices = indices[split_index:]

        # Extract training subset
        x_train = x_data[train_indices].copy()
        y_train = y_data[train_indices]

        # Extract validation subset
        x_rest = x_data[val_indices].copy()
        y_rest = y_data[val_indices]

        return x_train, y_train, x_rest, y_rest