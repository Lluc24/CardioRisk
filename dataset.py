"""
Dataset Management Module

This module provides a unified Dataset class for managing training, validation,
and test data with utilities for cross-validation, train/validation splitting,
and feature standardization.

The Dataset class encapsulates:
- Training data (features and labels)
- Validation/test data (features and labels)
- K-fold cross-validation generation with automatic standardization
- Train/validation split generation with randomization and standardization
- Feature standardization (zero mean, unit variance) applied to continuous features only

Key Features:
- Automatic 80/20 train/validation split during initialization
- Standardization applied only to continuous features (preserving one-hot encoded features)
- Reproducible randomization using configurable seed
- Support for both k-fold cross-validation and single train/val splits
"""
from typing import Generator
import numpy as np


class Dataset:
    """
    Container for machine learning datasets with cross-validation and standardization.

    This class manages training, validation, and test data, providing convenient methods
    for generating k-fold cross-validation splits and random train/validation splits.

    Upon initialization, the provided training data is automatically split into an 80/20
    train/validation split. The class tracks the number of continuous features to ensure
    that standardization (zero mean, unit variance) is applied only to continuous features,
    leaving one-hot encoded categorical features unchanged.

    All data shuffling uses NumPy's random permutation with a configurable seed for
    reproducibility across runs.

    Attributes:
        x_train: numpy array of shape (N_train, D). Training feature matrix after 80/20 split.
        y_train: numpy array of shape (N_train,). Training labels after 80/20 split.
        x_test: numpy array of shape (N_val, D). Validation/test feature matrix (20% of original data).
        y_test: numpy array of shape (N_val,). Validation/test labels (20% of original data).
        num_cont_features: int. Number of continuous features in the dataset.
                          Used to determine which columns to standardize.
                          Note: Column 0 is the intercept, columns 1 to num_cont_features
                          are continuous features, remaining columns are categorical.
        seed: int. Random seed for reproducible shuffling and splitting.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, num_cont_features: int, seed = 1) -> None:
        """Initializes the Dataset with automatic 80/20 train/validation split.

        Takes the full training dataset and automatically splits it into 80% training
        and 20% validation sets using random permutation. The split is reproducible
        based on the provided seed.

        Args:
            x_train: numpy array of shape (N, D). Full training feature matrix where
                    N is the number of samples and D is the number of features.
                    Expected structure: column 0 is intercept, columns 1 to num_cont_features
                    are continuous features, remaining columns are one-hot encoded categorical.
            y_train: numpy array of shape (N,). Training labels corresponding to x_train.
            num_cont_features: int. Number of continuous features (excluding intercept).
                              Used to determine which columns need standardization.
                              Columns [1:num_cont_features+1] will be standardized.
            seed: int. Random seed for reproducible data splitting. Default is 1.

        Side Effects:
            Populates x_train, y_train (80% of data), x_test, y_test (20% of data),
            num_cont_features, and seed attributes.
        """
        x_tr, y_tr, x_te, y_te = self._split_data_arrays(x_train, y_train, ratio=0.8)
        self.x_train = x_tr
        self.y_train = y_tr
        self.x_test = x_te
        self.y_test = y_te
        self.num_cont_features = num_cont_features
        self.seed = seed

    def __len__(self):
        """Returns the number of training samples in the dataset.

        Returns:
            int. The number of rows in x_train (N).
        """
        return len(self.x_train)

    def k_fold_generator(self, k: int = 5) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.generic, np.generic], None, None]:
        """Generates k-fold cross-validation splits with automatic standardization.

        Randomly shuffles the training data and divides it into k approximately
        equal-sized folds. For each fold, yields training and validation sets where
        the current fold becomes the validation set and the remaining k-1 folds
        become the training set.

        Standardization (zero mean, unit variance) is computed on the training set
        of each fold and applied to both training and validation sets. Only continuous
        features (columns 1 to num_cont_features) are standardized; column 0 (intercept)
        and categorical features (remaining columns) are left unchanged.

        The data is shuffled once at the beginning using np.random.permutation with
        the instance's seed, ensuring reproducibility and that each sample appears
        exactly once in validation across all k iterations.

        Args:
            k: int. Number of folds to create. Must be >= 2. Default is 5.

        Yields:
            tuple of (x_train, y_train, x_val, y_val, mean, std) for each fold:
                - x_train: numpy array of shape (N*(k-1)/k, D). Standardized training features.
                - y_train: numpy array of shape (N*(k-1)/k,). Training labels.
                - x_val: numpy array of shape (N/k, D). Standardized validation features.
                - y_val: numpy array of shape (N/k,). Validation labels.
                - mean: numpy generic of shape (num_cont_features,). Mean values computed
                       from training set continuous features.
                - std: numpy generic of shape (num_cont_features,). Standard deviation values
                      computed from training set continuous features.

        Example:
            >>> dataset = Dataset(x_train, y_train, num_cont_features=10, seed=42)
            >>> for x_tr, y_tr, x_val, y_val, mean, std in dataset.k_fold_generator(k=5):
            ...     # Train model on x_tr, y_tr
            ...     # Validate on x_val, y_val
            ...     # mean and std can be used later for test set standardization
        """
        # Calculate size of each fold (integer division)
        fold_size: int = int(len(self) / k)

        # Randomly shuffle all indices once
        np.random.seed(self.seed)
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
            # # balance classes in y_train for training portion of the fold only
            # np.random.seed()
            # pos_indices = np.where(y_train == 1)[0]
            # neg_indices = np.where(y_train == -1)[0]
            # n_pos = len(pos_indices)
            # n_neg = len(neg_indices)
            # if n_neg > n_pos:
            #     neg_indices = np.random.choice(neg_indices, n_pos, replace=False)
            # else:
            #     pos_indices = np.random.choice(pos_indices, n_neg, replace=False)
            # selected_indices = np.concatenate([pos_indices, neg_indices])
            # x_train= x_train[selected_indices]
            # y_train= y_train[selected_indices]

            # Extract validation data (1 fold)
            x_val = self.x_train[val_indices].copy()
            y_val = self.y_train[val_indices]

            # Compute standardization parameters from training set only (prevent data leakage)
            # Only compute for continuous features (columns 1 to num_cont_features)
            # Column 0 is intercept, remaining columns are one-hot encoded categorical
            mean: np.generic = np.mean(x_train[:, 1:self.num_cont_features+1], axis=0)
            std: np.generic = np.std(x_train[:, 1:self.num_cont_features+1], axis=0)

            # Apply standardization to continuous features in both train and validation sets
            x_train[:, 1:self.num_cont_features+1] = (x_train[:, 1:self.num_cont_features+1] - mean) / std
            x_val[:, 1:self.num_cont_features+1] = (x_val[:, 1:self.num_cont_features+1] - mean) / std

            yield x_train, y_train, x_val, y_val, mean, std

    def split_data(self, ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.generic, np.generic]:
        """Randomly splits the training data into train and validation sets with standardization.

        Shuffles the entire training dataset and divides it into two subsets
        according to the specified ratio. This is useful for creating a single
        train/validation split for model development and hyperparameter tuning.

        Standardization (zero mean, unit variance) is computed on the training set
        and applied to both training and validation sets. Only continuous features
        (columns 1 to num_cont_features) are standardized; column 0 (intercept)
        and categorical features (remaining columns) are left unchanged.

        The split uses random permutation with the instance's seed for reproducibility.

        Args:
            ratio: float in (0, 1). Proportion of data to use for training.
                  Default is 0.8 (80% training, 20% validation).

        Returns:
            tuple of (x_train, y_train, x_val, y_val, mean, std):
                - x_train: numpy array of shape (N*ratio, D). Standardized training features.
                - y_train: numpy array of shape (N*ratio,). Training labels.
                - x_val: numpy array of shape (N*(1-ratio), D). Standardized validation features.
                - y_val: numpy array of shape (N*(1-ratio),). Validation labels.
                - mean: numpy generic of shape (num_cont_features,). Mean values computed
                       from training set continuous features.
                - std: numpy generic of shape (num_cont_features,). Standard deviation values
                      computed from training set continuous features.

        Example:
            >>> dataset = Dataset(x_train, y_train, num_cont_features=10, seed=42)
            >>> x_tr, y_tr, x_val, y_val, mean, std = dataset.split_data(ratio=0.8)
            >>> # Train model on x_tr, y_tr; validate on x_val, y_val
            >>> # Use mean and std later for test set standardization
        """
        # Split data using random permutation
        x_train, y_train, x_val, y_val = self._split_data_arrays(self.x_train, self.y_train, ratio, seed = self.seed)

        # Compute standardization parameters from training set only (prevent data leakage)
        # Only compute for continuous features (columns 1 to num_cont_features)
        # Column 0 is intercept, remaining columns are one-hot encoded categorical
        mean: np.generic = np.mean(x_train[:, 1:self.num_cont_features+1], axis=0)
        std: np.generic = np.std(x_train[:, 1:self.num_cont_features+1], axis=0)

        # Apply standardization to continuous features in both train and validation sets
        x_train[:, 1:self.num_cont_features+1] = (x_train[:, 1:self.num_cont_features+1] - mean) / std
        x_val[:, 1:self.num_cont_features+1] = (x_val[:, 1:self.num_cont_features+1] - mean) / std

        return x_train, y_train, x_val, y_val, mean, std

    def get_test_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the test/validation set features and labels.

        This returns the 20% of data that was held out during initialization
        for use as a final test/validation set.

        Returns:
            tuple of (x_test, y_test):
                - x_test: numpy array of shape (N*0.2, D). Test feature matrix.
                - y_test: numpy array of shape (N*0.2,). Test labels.

        Example:
            >>> dataset = Dataset(x_train, y_train, num_cont_features=10, seed=42)
            >>> x_test, y_test = dataset.get_test_set()
            >>> # Evaluate final model on x_test, y_test
        """
        return self.x_test, self.y_test
    
    @staticmethod
    def _split_data_arrays(x_data, y_data, ratio: float = 0.8, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Randomly splits data arrays into two subsets based on a ratio.

        Internal utility method that performs the actual data splitting. Shuffles
        indices randomly using the provided seed, then partitions the data according
        to the specified ratio.

        This is a static method that can be used independently of the Dataset class
        for splitting arbitrary data arrays.

        Args:
            x_data: numpy array of shape (N, D). Feature matrix to split.
            y_data: numpy array of shape (N,). Labels to split.
            ratio: float in (0, 1). Proportion of data for the first subset.
                  Default is 0.8 (80% first subset, 20% second subset).
            seed: int. Random seed for reproducible shuffling. Default is 1.

        Returns:
            tuple of (x_train, y_train, x_rest, y_rest):
                - x_train: numpy array of shape (N*ratio, D). First subset features.
                - y_train: numpy array of shape (N*ratio,). First subset labels.
                - x_rest: numpy array of shape (N*(1-ratio), D). Second subset features.
                - y_rest: numpy array of shape (N*(1-ratio),). Second subset labels.

        Example:
            >>> x_tr, y_tr, x_val, y_val = Dataset._split_data_arrays(x, y, ratio=0.7, seed=42)
            >>> # x_tr contains 70% of data, x_val contains 30%
        """
        N = np.shape(x_data)[0] # Number of datapoints
        
        # Calculate split point
        split_index = int(N * ratio)

        # Randomly shuffle all indices
        np.random.seed(seed)
        indices = np.random.permutation(N)

        # Split indices according to ratio
        train_indices = indices[:split_index]
        val_indices = indices[split_index:]

        # Extract training subset (first ratio proportion)
        x_train = x_data[train_indices].copy()
        y_train = y_data[train_indices]

        # Extract validation/test subset (remaining 1-ratio proportion)
        x_rest = x_data[val_indices].copy()
        y_rest = y_data[val_indices]

        return x_train, y_train, x_rest, y_rest