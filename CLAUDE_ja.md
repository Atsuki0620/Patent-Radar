# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

注目特許仕分けくん（Patent-Radar）は、液体分離設備分野における特許スクリーニングシステムです。発明アイデアに対して先行特許をLLMベースで二値分類（hit/miss）し、抵触可能性のある特許を効率的に抽出します。

## 開発コマンド

```bash
# 仮想環境のアクティベート
venv/Scripts/activate  # Windows
source venv/bin/activate  # macOS/Linux

# テスト実行
pytest tests/                    # 全テスト実行
pytest tests/unit/              # ユニットテストのみ
pytest tests/integration/       # 統合テストのみ
pytest -v tests/unit/test_normalizer.py  # 特定ファイルのテスト
pytest -k "test_normalize"      # 特定の名前を含むテストのみ

# カバレッジ測定
pytest --cov=src --cov-report=html

# 型チェック（将来追加予定）
mypy src/

# コードフォーマット（将来追加予定）
black src/ tests/
isort src/ tests/

# メイン処理実行
python -m src.main --invention data/invention.json --patents data/patents.jsonl
python -m src.main --config custom_config.yaml  # カスタム設定で実行
```

## アーキテクチャ概要

### 処理パイプライン

```
1. データ入力
   ├── 発明アイデア (JSON/テキスト)
   └── 先行特許群 (JSONL)
        ↓
2. 前処理 (src/core/normalizer.py)
   ├── JSON正規化（文字エンコーディング、全半角統一）
   └── データ検証（必須フィールドチェック）
        ↓
3. 特徴抽出 (src/core/extractor.py)
   ├── 独立請求項（claim 1）の抽出
   └── 抄録（abstract）の切り出し
        ↓
4. LLM処理 (src/llm/)
   ├── 発明要旨化 (summarizer.py) - 1回のみ実行
   ├── 特許要旨化 (summarizer.py) - 各特許ごと
   └── 二値判定 (classifier.py) - hit/miss判定＋理由抽出
        ↓
5. ランキング (src/ranking/ranker.py)
   └── LLM_confidence降順ソート（同点時はpub_number）
        ↓
6. 出力 (src/core/exporter.py)
   ├── CSV形式（一覧表示用）
   └── JSONL形式（詳細データ）
```

### モジュール間の依存関係

```
src/
├── main.py              # エントリーポイント、全体の流れを制御
├── core/
│   ├── normalizer.py    # 入力データの正規化
│   ├── extractor.py     # 特許文書から必要部分を抽出
│   ├── exporter.py      # 結果をCSV/JSONLに出力
│   └── models.py        # Pydanticモデル定義
├── llm/
│   ├── client.py        # OpenAI API クライアント
│   ├── prompts.py       # プロンプトテンプレート管理
│   ├── summarizer.py    # 発明・特許の要旨化
│   └── classifier.py    # hit/miss判定とスコアリング
├── ranking/
│   └── ranker.py        # 信頼度に基づくランキング
└── utils/
    ├── config.py        # 設定ファイル読み込み
    └── logger.py        # ログ設定
```

## 重要な設計判断

### MVPスコープ
- **実装済み**: LLM_confidenceのみでのスコアリング
- **未実装**: TopK絞り込み、retrieval_score、エビデンス検証
- **理由**: まず基本機能で精度を検証し、段階的に機能追加

### LLM処理の最適化
- 発明要旨化は1回のみ実行してキャッシュ
- 特許処理はバッチ処理で並列化（max_workers設定）
- temperature=0.0で決定論的な出力を確保

### エラーハンドリング
- LLM APIエラー時は3回リトライ（指数バックオフ）
- 個別特許の処理失敗は全体を止めない（エラーログ記録）
- 必須フィールド欠損時は明確なエラーメッセージ

## データフォーマット詳細

### 発明アイデア入力
```json
{
  "title": "新規液体分離システム",
  "problem": "従来技術の課題",
  "solution": "解決手段の説明",
  "effects": "期待される効果",
  "key_elements": ["要素1", "要素2"],  // 重要な技術要素
  "constraints": "制約条件"             // オプション
}
```

### 特許JSON必須フィールド
```json
{
  "publication_number": "JP2025-123456A",
  "title": "特許タイトル",
  "assignee": "出願人名",
  "pub_date": "2025-01-01",
  "claims": [
    {
      "no": 1,
      "text": "請求項1の内容...",
      "is_independent": true
    }
  ],
  "abstract": "要約文..."
}
```

### 判定結果の構造
```json
{
  "pub_number": "JP2025-123456A",
  "decision": "hit",           // "hit" or "miss"
  "LLM_confidence": 0.82,      // 0.0-1.0
  "reasons": [
    {
      "quote": "引用テキスト（最大12語）",
      "source": {
        "field": "claim",       // "claim" or "abstract"
        "locator": "claim 1"    // 位置情報
      }
    }
  ],
  "flags": {
    "verified": false,          // MVP: 常にfalse
    "used_retrieval": false,    // MVP: 常にfalse
    "used_topk": false         // MVP: 常にfalse
  }
}
```

## 設定管理

### 環境変数（.env）
- `OPENAI_API_KEY`: 必須、APIキー
- `LOG_LEVEL`: INFO/DEBUG/ERROR
- `ENVIRONMENT`: development/staging/production

### config.yaml
- LLM設定の変更時は`llm`セクションを編集
- 出力先変更は`io`セクションで指定
- デバッグ時は`debug.verbose: true`を設定

## テスト戦略

### ユニットテスト重点項目
- 正規化処理の各種パターン（全半角、大小文字）
- 請求項抽出の境界値（独立/従属の判定）
- スコアリングの安定性（同一入力で同一出力）

### 統合テスト
- ゴールドセット（tests/data/）を使用した精度評価
- Precision@TopN、Recall@TopNの自動計算
- 処理時間のベンチマーク

## トラブルシューティング

### よくある問題
1. **LLM APIエラー**: API キーとクォータを確認
2. **文字化け**: 入力JSONLのエンコーディングを確認（UTF-8推奨）
3. **メモリ不足**: batch_sizeを小さく調整
4. **出力の不安定性**: temperature=0.0を確認

## 将来の拡張ポイント

### Phase 2（計画中）
- TopK絞り込み: 大量特許の事前フィルタリング
- Retrieval score: 意味的類似度の追加指標
- エビデンス検証: 引用箇所の実在性チェック

### Phase 3（構想）
- マルチモーダル対応（図面解析）
- 進歩性評価の詳細化
- WebUIの追加