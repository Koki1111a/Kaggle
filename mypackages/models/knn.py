import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
from sklearn.neighbors import KNeighborsClassifier


class KNN(BaseModel):
    def __init__(
            self,
            x=None,  # 特徴量データ
            y=None,  # 目的変数データ
            split_ratio=[0.6,0.2,0.2],  # データ分割比率 [train, test, val]
            n_neighbors=5,  # 近傍数
            weights='uniform',  # 重み付け方法
            algorithm='auto',  # 近傍探索アルゴリズム
            leaf_size=30,  # 木の葉のサイズ
            p=2,  # 距離の定義（2:ユークリッド, 1:マンハッタン）
            metric='minkowski',  # 距離関数
            n_jobs=-1,  # 並列実行数
            random_state=42,  # 乱数シード（BaseModel用）
    ):
        super().__init__(x=x, y=y, split_ratio=split_ratio, random_state=random_state)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.n_jobs = n_jobs


    def set_model(self):
        model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            n_jobs=self.n_jobs
        )
        return model


    def train(self, x_train=None, y_train=None, x_val=None, y_val=None, verbose=True):
        if x_train is None:
            x_train = self.x_train
        if y_train is None:
            y_train = self.y_train
        model = self.set_model()
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def get_default_param_grid():
        return {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [20, 30, 40],
            'p': [1, 2],
        } 