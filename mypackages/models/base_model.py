from abc import ABC, abstractmethod
import itertools
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold


class BaseModel(ABC):

    def __init__(self, x=None, y=None, split_ratio=[0.6, 0.2, 0.2], random_state=42):
        self.x = x
        self.y = y
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_val = None
        self.y_val = None


    @abstractmethod
    def set_model(self):
        pass


    @abstractmethod
    def train(self, x_train=None, y_train=None, x_val=None, y_val=None, verbose=True):
        pass


    @staticmethod
    def get_default_param_grid():
        pass


    def split_data(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y,
            test_size=self.split_ratio[1]/np.sum(self.split_ratio[:2]),
            random_state=self.random_state
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train,
            test_size=self.split_ratio[2]/np.sum(self.split_ratio[1:]),
            random_state=self.random_state
        )
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        return x_train, x_test, y_train, y_test, x_val, y_val


    def test(self, model, x_test=None, y_test=None, return_type='confusion_matrix'):
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test
        y_pred = model.predict(x_test)
        if return_type == 'confusion_matrix':
            return confusion_matrix(y_test, y_pred)
        elif return_type == 'accuracy':
            return accuracy_score(y_test, y_pred)
        elif return_type == 'predict':
            return y_pred
        else:
            raise ValueError(f"Unknown return_type: {return_type}")


    def cross_validate(self, cv=5, verbose=False):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = []
        for train_idx, test_idx in skf.split(self.x, self.y):
            x_train, x_test = self.x.iloc[train_idx], self.x.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            x_tr, x_val, y_tr, y_val = train_test_split(
                x_train, y_train, test_size=0.2, random_state=self.random_state)
            model = self.train(x_tr, y_tr, x_val, y_val, verbose=verbose)
            score = self.test(model, x_test, y_test, return_type='accuracy')
            scores.append(score)
        return scores, np.mean(scores)


    def grid_search(self, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = self.get_default_param_grid()
        best_score = -float('inf')
        best_params = None
        keys = list(param_grid.keys())
        all_combinations = list(itertools.product(*param_grid.values()))
        total = len(all_combinations)
        for i, values in enumerate(all_combinations, 1):
            print(f"\rGrid search progress: {i}/{total}", end='', flush=True)
            for k, v in zip(keys, values):
                setattr(self, k, v)
            scores, mean_score = self.cross_validate(cv=cv)
            if mean_score > best_score:
                best_score = mean_score
                best_params = dict(zip(keys, values))
        print()  # ループ後に改行
        return best_params, best_score