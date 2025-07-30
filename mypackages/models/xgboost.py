import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
from xgboost import XGBClassifier


class XGBoost(BaseModel):
    def __init__(
            self,
            x=None,  # 特徴量データ
            y=None,  # 目的変数データ
            split_ratio=[0.6,0.2,0.2],  # データ分割比率 [train, test, val]
            n_estimators=100,  # 木の数（弱学習器の数）
            max_depth=3,  # 木の最大深さ
            learning_rate=0.01,  # 学習率
            booster='gbtree',  # ブースタータイプ
            objective='reg:squarederror',  # 目的関数
            eval_metric='mlogloss',  # 評価指標
            subsample=0.8,  # 各木の学習に使うサンプルの割合
            colsample_bytree=0.8,  # 各木の学習に使う特徴量の割合
            colsample_bylevel=1,  # 各レベルでの特徴量サブサンプリング比率
            colsample_bynode=1,  # 各ノードでの特徴量サブサンプリング比率
            gamma=0,  # 分岐のために必要な最小損失減少
            min_child_weight=1,  # 葉に必要な最小のサンプル重みの合計
            max_delta_step=0,  # 各木の重み推定の最大ステップ
            reg_alpha=0,  # L1正則化項
            reg_lambda=1,  # L2正則化項
            scale_pos_weight=1,  # 正例と負例の重みバランス
            early_stopping_rounds=10,  # アーリーストッピングのラウンド数
            tree_method='auto',  # 木構築アルゴリズム
            enable_categorical=False,  # カテゴリ変数のサポート有無
            num_parallel_tree=1,  # ランダムフォレスト時の並列木数
            monotone_constraints=None,  # 単調制約
            interaction_constraints=None,  # 相互作用制約
            base_score=0.5,  # 初期予測値
            random_state=42,  # 乱数シード
            n_jobs=-1,  # 並列実行数
            verbosity=1,  # ログ出力レベル
            device='cpu',  # 使用するデバイス (cpu, cuda, cuda:0 など)
            missing=None,  # 欠損値の指定
            importance_type='gain',  # feature_importances_の種類
            validate_parameters=1,  # パラメータ検証の有無
    ):
        super().__init__(x=x, y=y, split_ratio=split_ratio, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.booster = booster
        self.objective = objective
        self.eval_metric = eval_metric
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.tree_method = tree_method
        self.enable_categorical = enable_categorical
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.base_score = base_score
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.device = device
        self.missing = missing
        self.importance_type = importance_type
        self.validate_parameters = validate_parameters


    def set_model(self):
        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            booster=self.booster,
            objective=self.objective,
            eval_metric=self.eval_metric,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            early_stopping_rounds=self.early_stopping_rounds,
            tree_method=self.tree_method,
            enable_categorical=self.enable_categorical,
            num_parallel_tree=self.num_parallel_tree,
            monotone_constraints=self.monotone_constraints,
            interaction_constraints=self.interaction_constraints,
            base_score=self.base_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            device=self.device,
            missing=self.missing,
            importance_type=self.importance_type,
            validate_parameters=self.validate_parameters
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
            sample_weight=None,
            eval_set=[(x_val, y_val)],
            verbose=verbose,
            xgb_model=None,
            sample_weight_eval_set=None,
            feature_weights=None
        )
        return model 

    @staticmethod
    def get_default_param_grid():
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 1, 5],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2],
        } 