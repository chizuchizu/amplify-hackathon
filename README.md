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
```bash
# OK
Your token file exists.

# Bad (トークンファイルが見つからない)
Your token file doesn't exist.
```
