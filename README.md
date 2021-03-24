# amplify-hackathon

## 実行方法

### アクセストークンについて

トークン流出が怖いのでローカルファイルから呼び出すようにしています。

`~/.amplify/token.json`に以下のjson形式で保存してください。

```~/.amplify/token.json
{
    "AMPLIFY_TOKEN": "YOUR TOKEN"
}
```

### 動作確認

全てDocker上で動かしています。
ターミナルで以下のコードを打ってトークンが存在するか確認してください。
```bash
docker-compose up check
```
```
# OK
Your token file exists.

# Bad (トークンファイルが見つからない)
Your token file doesn't exist.
```

### データセットについて
#### MNIST

[Digit Recognizer | Kaggle](https://www.kaggle.com/c/digit-recognizer) のデータセットを使っています。  
これはMNISTのデータを扱いやすくcsvに変換する処理を施したものです。

- `train.csv`: そのままのデータ  
- `train_400.csv`: 28x28(pixels)から20x20(pixels)に変換した後のデータ

`train_400.csv`の作成は`src/resize.py`を用いました。

これらのデータは既に作成し、GitHubにあるのでcloneした時点で利用可能となっています。
