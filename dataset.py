import numpy as np

class Dataset:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray = None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def __len__(self):
        return len(self.x_train)

    def k_fold_generator(self, k: int):
        fold_size: int = int(len(self) / k)
        indices = np.random.permutation(len(self))

        for i in range(k):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < k else len(self)

            train_indices = np.concatenate((indices[:val_start], indices[val_end:]))
            val_indices = indices[val_start:val_end]

            x_train = self.x_train[train_indices]
            y_train = self.y_train[train_indices]
            x_val = self.x_train[val_indices]
            y_val = self.y_train[val_indices]

            yield x_train, y_train, x_val, y_val