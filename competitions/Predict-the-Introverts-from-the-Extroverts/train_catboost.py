import os
import sys

import pandas as pd

# パス設定
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(DIR_PATH))
sys.path.append(ROOT_PATH)

from mypackages.models.catboost import CatBoost
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
    cat = CatBoost(
        x=X,
        y=y,
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42
    )

    # グリッドサーチ
    best_params, best_score = cat.grid_search(cv=5)
    print(f"Best params: {best_params}")
    print(f"Best CV score: {best_score:.4f}")

    # best_paramsをファイルに保存（複数モデル対応）
    params_path = os.path.join(DIR_PATH, "best_params.json")

    # update_json_file関数を使って保存
    update_json_file(params_path, "catboost", best_params)


if __name__ == "__main__":
    main() 