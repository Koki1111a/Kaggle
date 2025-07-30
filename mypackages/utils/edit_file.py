import os
import json


def update_json_file(json_path, key, value):
    """
    指定したJSONファイルに対して、keyでvalueを保存・更新する汎用関数。
    既存ファイルがあれば内容を更新、なければ新規作成。

    Args:
        json_path (str): JSONファイルのパス
        key (str): 保存・更新するキー
        value (Any): 保存する値（シリアライズ可能な型）
    """
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[key] = value

    with open(json_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_value_from_json_file(json_path, key, default=None):
    """
    指定したJSONファイルからkeyの値を取得する関数。
    ファイルやkeyが存在しない場合はdefaultを返す。

    Args:
        json_path (str): JSONファイルのパス
        key (str): 取得するキー
        default (Any): 見つからなかった場合に返す値（デフォルト: None）

    Returns:
        Any: 取得した値、またはdefault
    """
    if not os.path.exists(json_path):
        return default
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get(key, default) 