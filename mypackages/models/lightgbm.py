import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation


class LightGBM(BaseModel):
    def __init__(
            self,
            x=None,  # 特徴量データ
            y=None,  # 目的変数データ
            split_ratio=[0.6,0.2,0.2],  # データ分割比率 [train, test, val]
            n_estimators=100,  # 木の数
            max_depth=-1,  # 木の最大深さ
            learning_rate=0.1,  # 学習率
            boosting_type='gbdt',  # ブースティングタイプ
            objective='binary',  # 目的関数
            metric='binary_logloss',  # 評価指標
            subsample=1.0,  # サブサンプル比率
            colsample_bytree=1.0,  # 各木の特徴量サブサンプル比率
            reg_alpha=0.0,  # L1正則化
            reg_lambda=0.0,  # L2正則化
            min_child_weight=1e-3,  # 葉の最小サンプル重み
            min_child_samples=20,  # 葉の最小サンプル数
            num_leaves=31,  # 葉の数
            random_state=42,  # 乱数シード
            n_jobs=-1,  # 並列実行数
            importance_type='split',  # feature_importances_の種類
            early_stopping_rounds=10,  # アーリーストッピング
            device='cpu',  # 使用デバイス
            verbose=True,  # ログ出力レベル
            verbosity=1,  # ログ出力レベル
    ):
        super().__init__(x=x, y=y, split_ratio=split_ratio, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.boosting_type = boosting_type
        self.objective = objective
        self.metric = metric
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.num_leaves = num_leaves
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.importance_type = importance_type
        self.early_stopping_rounds = early_stopping_rounds
        self.device = device
        self.verbose = verbose
        self.verbosity = verbosity


    def set_model(self):
        # verbose/verbosity are not always needed, so only set if verbose=True
        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            boosting_type=self.boosting_type,
            objective=self.objective,
            metric=self.metric,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            min_child_samples=self.min_child_samples,
            num_leaves=self.num_leaves,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            importance_type=self.importance_type,
            device=self.device
        )
        # Only set verbose/verbosity if verbose is True, otherwise set verbosity to -1
        if self.verbose:
            params['verbose'] = True
            params['verbosity'] = 1
        else:
            params['verbose'] = False
            params['verbosity'] = -1
        model = LGBMClassifier(**params)
        return model


    def train(self, x_train=None, y_train=None, x_val=None, y_val=None, verbose=True):
        if x_train is None:
            x_train = self.x_train
        if y_train is None:
            y_train = self.y_train
        if x_val is None:
            x_val = self.x_val
        if y_val is None:
            y_val = self.y_val
        if verbose:
            self.verbosity = 1
            self.verbose = True
        else:
            self.verbosity = -1
            self.verbose = False
        # yが2次元の場合は1次元に変換
        import numpy as np
        if hasattr(y_train, 'ndim') and y_train.ndim > 1:
            y_train = np.ravel(y_train)
        if hasattr(y_val, 'ndim') and y_val is not None and y_val.ndim > 1:
            y_val = np.ravel(y_val)
        model = self.set_model()
        callbacks = []
        callbacks.append(early_stopping(self.early_stopping_rounds, verbose=self.verbose))
        # log_evaluation(0) suppresses all eval logs
        if not self.verbose:
            callbacks.append(log_evaluation(0))
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            callbacks=callbacks
        )
        return model

    @staticmethod
    def get_default_param_grid():
        return {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [15, 31, 63],
            'max_depth': [-1, 5, 10],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0.0, 0.1, 1.0],
            'reg_lambda': [0.0, 0.1, 1.0],
        } 