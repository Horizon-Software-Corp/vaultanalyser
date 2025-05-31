# HyperLiquid User Analysis Extension

このプロジェクトは、HyperLiquid Vault Analyserを拡張して、個別ユーザーアドレスの分析も可能にしたものです。

## 機能

### 1. Vault Analysis (既存機能)
- **実行方法**: `streamlit run main.py`
- **機能**: HyperLiquidのボルト分析
- **対象**: ボルトアドレスのみ

### 2. User Analysis (新機能)
- **実行方法**: `streamlit run user_analysis.py`
- **機能**: HyperLiquidのユーザーアドレス分析
- **対象**: リーダーボードのユーザーアドレス

## セットアップ

### 必要な依存関係
```bash
pip install -r requirements.txt
```

### ディレクトリ構造
```
vaultanalyser/
├── main.py                    # Vault分析メインページ
├── user_analysis.py           # User分析メインページ
├── hyperliquid/
│   ├── vaults.py             # Vault用API処理
│   └── users.py              # User用API処理（新規）
├── metrics/
│   ├── drawdown.py           # 共通メトリクス計算
│   └── sharpe_reliability.py # Sharpe Ratio信頼性分析
├── cache/                    # Vaultデータキャッシュ
└── user_cache/               # Userデータキャッシュ（新規）
    ├── leaderboard.json      # リーダーボードデータ
    ├── processed_addresses.txt # 処理済みアドレス
    ├── failed_addresses.txt   # 失敗したアドレス
    └── user_data/            # 個別ユーザーデータ
```

## 使用方法

### Vault Analysis
```bash
streamlit run main.py
```
- 既存のボルト分析機能
- ボルトのパフォーマンス指標を表示
- フィルタリング・ソート機能

### User Analysis
```bash
streamlit run user_analysis.py
```
- 新しいユーザー分析機能
- リーダーボードからユーザーデータを取得・分析
- ボルトと同じメトリクスで評価

## User Analysis の詳細

### 初回実行
1. リーダーボードデータを自動取得（17,155ユーザー）
2. 上位60ユーザーのポートフォリオデータを取得
3. 各ユーザーのメトリクスを計算・表示

### データ取得設定
- **処理対象**: 初回60アドレス（設定変更可能）
- **API制限対策**: 各リクエスト間に1秒sleep
- **進捗管理**: 処理済みアドレスを記録、次回はスキップ
- **エラーハンドリング**: 失敗したアドレスを記録

### 分析指標
ボルト分析と同じ指標を使用：
- **Max DD %**: 最大ドローダウン
- **Rekt**: 破産回数
- **Current Account Value**: 現在の口座価値
- **Sharpe Ratio**: シャープレシオ
- **Sortino Ratio**: ソルティノレシオ
- **Av. Daily Gain %**: 平均日次利益率
- **Gain %**: 総利益率
- **Days Since**: 取引開始からの日数
- **Sharpe Reliability**: シャープレシオ信頼性
- **JKM Test P-value**: JKM検定p値
- **Lo Test P-value**: Lo検定p値
- **Fisher Score**: フィッシャースコア

### フィルタリング機能
- アドレス検索
- 各指標での範囲指定
- リアルタイムフィルタリング

## API仕様

### リーダーボード取得
```python
POST https://api.hyperliquid.xyz/info
{
  "type": "leaderboard"
}
```

### ユーザーポートフォリオ取得
```python
POST https://api.hyperliquid.xyz/info
{
  "type": "portfolio",
  "user": "0x..."
}
```

## キャッシュシステム

### Vault用キャッシュ
- `./cache/vaults_cache.json` - ボルト一覧（24時間有効）
- `./cache/vault_detail/{leader}_{vault}/` - 個別ボルト詳細

### User用キャッシュ
- `./user_cache/leaderboard.json` - リーダーボード（永続）
- `./user_cache/user_data/{address}.json` - 個別ユーザーデータ
- `./user_cache/processed_addresses.txt` - 処理済みアドレス一覧
- `./user_cache/user_dataframe.pkl` - 分析結果キャッシュ

## 設定変更

### 処理対象アドレス数の変更
`hyperliquid/users.py`の`MAX_ADDRESSES_TO_PROCESS`を変更：
```python
MAX_ADDRESSES_TO_PROCESS = 100  # デフォルト: 60
```

### API制限の調整
`hyperliquid/users.py`の`API_SLEEP_SECONDS`を変更：
```python
API_SLEEP_SECONDS = 2  # デフォルト: 1秒
```

## トラブルシューティング

### よくある問題

1. **API制限エラー**
   - `API_SLEEP_SECONDS`を増やす
   - 処理対象アドレス数を減らす

2. **データが表示されない**
   - `user_cache/`ディレクトリの権限を確認
   - キャッシュファイルを削除して再実行

3. **メモリ不足**
   - 処理対象アドレス数を減らす
   - キャッシュファイルを定期的に削除

### ログ確認
コンソール出力でAPI呼び出し状況を確認：
```
Leaderboard cache found, skipping download
Total addresses in leaderboard: 17155
Already processed: 60
Will process: 60
Successfully processed 0x123...
```

## 今後の拡張予定

1. **バッチ処理機能**: 大量アドレスの効率的処理
2. **比較機能**: ボルトリーダー vs 一般ユーザー
3. **統計分析**: ユーザー群の傾向分析
4. **アラート機能**: 優秀なユーザーの自動検出

## 注意事項

- API制限を考慮し、大量データ取得時は時間がかかります
- リーダーボードデータは手動更新が必要です
- ユーザーデータは投資助言ではありません
