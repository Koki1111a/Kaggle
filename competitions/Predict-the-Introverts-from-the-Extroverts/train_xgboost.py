import os
import sys

import numpy as np
import pandas as pd

# パス設定
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(DIR_PATH))
sys.path.append(ROOT_PATH)

from mypackages.models.xgboost import XGBoost
from mypackages.utils.process_data import ProcessData
from mypackages.utils.edit_file import update_json_file

def main():
    # データ読み込み
    train_path = os.path.join(DIR_PATH, "data", "train.csv")
    df = pd.read_csv(train_path)

    # データ前処理
    processor = ProcessData()
    df = processor.drop_features(df, ["id"])
    df = processor.encode_features(df)
    X = processor.drop_features(df, ["Personality"])
    y = processor.get_features(df, ["Personality"])

    # モデル初期化
    xgb = XGBoost(
        x=X,
        y=y,
        objective='binary:logistic',
        eval_metric='logloss',
        missing=np.nan
    )

    # グリッドサーチ
    best_params, best_score = xgb.grid_search(cv=5)
    print(f"Best params: {best_params}")
    print(f"Best CV score: {best_score:.4f}")

    # best_paramsをファイルに保存（複数モデル対応）
    params_path = os.path.join(DIR_PATH, "best_params.json")

    # update_json_file関数を使って保存
    update_json_file(params_path, "xgboost", best_params)


if __name__ == "__main__":
    main()
