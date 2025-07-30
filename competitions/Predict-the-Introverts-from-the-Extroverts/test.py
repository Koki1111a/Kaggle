import os
import sys
import numpy as np
import pandas as pd

# パス設定
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(DIR_PATH))
OUTPUT_DIR_PATH = os.path.join(DIR_PATH, "outputs")
sys.path.append(ROOT_PATH)

from mypackages.models.xgboost import XGBoost
from mypackages.models.catboost import CatBoost
from mypackages.models.lightgbm import LightGBM
from mypackages.utils.process_data import ProcessData
from mypackages.utils.edit_file import get_value_from_json_file

def main():
    # データ読み込み
    train_path = os.path.join(DIR_PATH, "data", "train.csv")
    test_path = os.path.join(DIR_PATH, "data", "test.csv")
    sample_submission_path = os.path.join(DIR_PATH, "data", "sample_submission.csv")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_sample = pd.read_csv(sample_submission_path)

    # データ前処理
    processor = ProcessData()
    
    # 訓練データの前処理
    df_train = processor.drop_features(df_train, ["id"])
    df_train = processor.encode_features(df_train)
    X_train_full = processor.drop_features(df_train, ["Personality"])
    y_train_full = processor.get_features(df_train, ["Personality"])
    y_train_full = y_train_full.squeeze()  # Series化

    # テストデータの前処理
    df_test = processor.drop_features(df_test, ["id"])
    X_test = processor.encode_features(df_test)

    # best_paramsの読込
    params_path = os.path.join(DIR_PATH, "best_params.json")
    best_params_xgb = get_value_from_json_file(params_path, "xgboost", default={})
    best_params_cat = get_value_from_json_file(params_path, "catboost", default={})
    best_params_lgb = get_value_from_json_file(params_path, "lightgbm", default={})
    if not (best_params_xgb and best_params_cat and best_params_lgb):
        print("best_params.jsonに必要なパラメータがありません")
        return

    # Personality: 文字列ラベルを数値化（0/1）
    if y_train_full.dtype == object or y_train_full.dtype.name == 'category':
        y_train_full = y_train_full.map({"Extrovert": 0, "Introvert": 1})

    # データ分割
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import mode
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # 各フォールドでのテスト予測を保存するリスト
    test_predictions_xgb = []
    test_predictions_cat = []
    test_predictions_lgb = []
    
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        # 各モデルの初期化
        xgb = XGBoost(
            x=X_train,
            y=y_train,
            objective='binary:logistic',
            eval_metric='logloss',
            missing=np.nan,
            **best_params_xgb
        )
        cat = CatBoost(
            x=X_train,
            y=y_train,
            loss_function='Logloss',
            eval_metric='Accuracy',
            random_seed=42,
            **best_params_cat
        )
        lgb = LightGBM(
            x=X_train,
            y=y_train,
            objective='binary',
            metric='binary_logloss',
            random_state=42,
            **best_params_lgb
        )

        # 各モデルの学習
        model_xgb = xgb.train(X_train, y_train, X_val, y_val, verbose=False)
        model_cat = cat.train(X_train, y_train, X_val, y_val, verbose=False)
        model_lgb = lgb.train(X_train, y_train, X_val, y_val, verbose=False)

        # バリデーション予測（平均）
        label_xgb = model_xgb.predict(X_val)
        label_cat = model_cat.predict(X_val)
        label_lgb = model_lgb.predict(X_val)
        # 平均
        preds = np.stack([label_xgb, label_cat, label_lgb], axis=1)
        pred_label = (np.mean(preds, axis=1) > 0.5).astype(int)
        acc = (pred_label == y_val.values).mean()
        scores.append(acc)
        print(f"Fold {fold+1} accuracy: {acc:.4f}")

        # テストデータの予測
        test_pred_xgb = model_xgb.predict(X_test)
        test_pred_cat = model_cat.predict(X_test)
        test_pred_lgb = model_lgb.predict(X_test)
        
        test_predictions_xgb.append(test_pred_xgb)
        test_predictions_cat.append(test_pred_cat)
        test_predictions_lgb.append(test_pred_lgb)
    
    print(f"Mean CV accuracy (ensemble): {np.mean(scores):.4f}")

    # 全フォールドでのテスト予測を平均
    test_predictions_xgb = np.array(test_predictions_xgb)
    test_predictions_cat = np.array(test_predictions_cat)
    test_predictions_lgb = np.array(test_predictions_lgb)
    
    # 各モデルの平均予測（確率の平均）
    avg_pred_xgb = np.mean(test_predictions_xgb, axis=0)
    avg_pred_cat = np.mean(test_predictions_cat, axis=0)
    avg_pred_lgb = np.mean(test_predictions_lgb, axis=0)
    
    # 最終的な平均
    final_predictions = np.stack([avg_pred_xgb, avg_pred_cat, avg_pred_lgb], axis=1)
    final_pred_label = (np.mean(final_predictions, axis=1) > 0.5).astype(int)
    
    # 予測結果を文字列に変換
    final_pred_string = np.where(final_pred_label == 0, "Extrovert", "Introvert")
    
    # サブミッションファイルの作成
    submission_df = df_sample.copy()
    submission_df['Personality'] = final_pred_string
    
    # 出力
    output_path = os.path.join(OUTPUT_DIR_PATH, "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")
    
    # # 予測結果の統計
    # extrovert_count = np.sum(final_pred_label == 0)
    # introvert_count = np.sum(final_pred_label == 1)
    # print(f"Extrovert predictions: {extrovert_count}")
    # print(f"Introvert predictions: {introvert_count}")

if __name__ == "__main__":
    main()
