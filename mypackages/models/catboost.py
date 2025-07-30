import sys
import os
import tempfile
import atexit
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
from catboost import CatBoostClassifier


class CatBoost(BaseModel):
    def __init__(
            self,
            x=None,  # 特徴量データ
            y=None,  # 目的変数データ
            split_ratio=[0.6,0.2,0.2],  # データ分割比率 [train, test, val]
            iterations=1000,  # 学習の繰り返し回数
            depth=6,  # 木の深さ
            learning_rate=0.03,  # 学習率
            loss_function='Logloss',  # 損失関数
            eval_metric='Accuracy',  # 評価指標
            random_seed=42,  # 乱数シード
            l2_leaf_reg=3.0,  # L2正則化
            border_count=254,  # 連続値特徴量の分割数
            thread_count=-1,  # 並列スレッド数
            early_stopping_rounds=10,  # アーリーストッピング
            verbose=1,  # ログ出力レベル
            cat_features=None,  # カテゴリ特徴量の指定
            task_type='CPU',  # 使用デバイス
            grow_policy='SymmetricTree',  # 木の成長方針
            boosting_type='Plain',  # ブースティングタイプ
            bagging_temperature=1.0,  # バギング温度
            scale_pos_weight=1.0,  # 正例と負例の重みバランス
            train_dir=None,  # 学習ログの出力ディレクトリ（Noneで一時ディレクトリを使用）
    ):
        super().__init__(x=x, y=y, split_ratio=split_ratio, random_state=random_seed)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.random_seed = random_seed
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.thread_count = thread_count
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.cat_features = cat_features
        self.task_type = task_type
        self.grow_policy = grow_policy
        self.boosting_type = boosting_type
        self.bagging_temperature = bagging_temperature
        self.scale_pos_weight = scale_pos_weight
        # train_dirがNoneの場合は一時ディレクトリを使用
        if train_dir is None:
            self.train_dir = tempfile.mkdtemp(prefix="catboost_")
            self._is_temp_dir = True
            # プログラム終了時にクリーンアップを登録
            atexit.register(self._cleanup_temp_dir)
        else:
            self.train_dir = train_dir
            self._is_temp_dir = False


    def set_model(self):
        model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            random_seed=self.random_seed,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            thread_count=self.thread_count,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
            cat_features=self.cat_features,
            task_type=self.task_type,
            grow_policy=self.grow_policy,
            boosting_type=self.boosting_type,
            bagging_temperature=self.bagging_temperature,
            scale_pos_weight=self.scale_pos_weight,
            train_dir=self.train_dir
        )
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
        model = self.set_model()
        model.fit(
            x_train,
            y_train,
            eval_set=(x_val, y_val),
            verbose=verbose,
            cat_features=self.cat_features
        )
        
        # 一時ディレクトリを使用した場合は学習後に削除
        if hasattr(self, '_is_temp_dir') and self._is_temp_dir:
            self._cleanup_temp_dir()
        
        return model

    def _cleanup_temp_dir(self):
        """一時ディレクトリを安全に削除"""
        if hasattr(self, '_is_temp_dir') and self._is_temp_dir and hasattr(self, 'train_dir'):
            if self.train_dir and os.path.exists(self.train_dir):
                try:
                    shutil.rmtree(self.train_dir)
                    self._is_temp_dir = False
                except Exception:
                    pass  # 削除に失敗してもエラーにしない

    def __del__(self):
        """オブジェクト破棄時にクリーンアップ"""
        self._cleanup_temp_dir()

    @staticmethod
    def get_default_param_grid():
        return {
            'iterations': [500, 1000, 1500],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.03, 0.1],
            'l2_leaf_reg': [1, 3, 5],
            'bagging_temperature': [0.5, 1.0, 2.0],
            'border_count': [32, 128, 254],
        } 