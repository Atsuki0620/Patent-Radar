# Patent-Radar 包括的評価システム 使用方法ガイド

## 1. システム概要

### 1.1 目的
Patent-Radar分類器の二値分類性能を包括的に評価し、ビジネス判断に必要な詳細レポートを自動生成する。

### 1.2 主要機能
- **厳密評価**: 1000分割閾値探索によるF1最適化
- **網羅的分析**: ROC・分布・混同行列・ビジネス指標の統合評価
- **詳細レポート**: 目的・結果・考察付きHTMLレポート自動生成
- **履歴管理**: 評価ID付きの体系的管理・比較分析

## 2. 基本的な使用方法

### 2.1 ワンコマンド実行（推奨）
```bash
python analysis/run_comprehensive_evaluation.py \
    --model-name "PatentRadar_v2.0" \
    --description "閾値最適化改良版" \
    --labels-path "testing/data/labels.jsonl" \
    --predictions-path "archive/outputs/goldset_results.jsonl"
```

**パラメータ説明**:
- `--model-name`: 分類器識別名（評価IDに使用）
- `--description`: 評価の説明文（省略可）
- `--labels-path`: ゴールドラベルファイルパス
- `--predictions-path`: 予測結果ファイルパス

### 2.2 段階的実行
```bash
# 1. データ統合・前処理
python analysis/evaluation_system/data_processor.py \
    --labels "testing/data/labels.jsonl" \
    --predictions "archive/outputs/goldset_results.jsonl" \
    --output "analysis/temp/evaluation_data.csv"

# 2. 評価実行
python analysis/evaluation_system/strict_evaluator.py \
    --data "analysis/temp/evaluation_data.csv" \
    --eval-id "PE_20250901_143052_PatentRadar_v2.0"

# 3. レポート生成
python analysis/evaluation_system/report_builder.py \
    --eval-id "PE_20250901_143052_PatentRadar_v2.0"
```

## 3. 入力データ形式

### 3.1 ゴールドラベルファイル（labels.jsonl）
```json
{"publication_number": "JP2025-100001A", "gold_label": "hit", "brief_rationale": "..."}
{"publication_number": "JP2025-100002A", "gold_label": "miss", "brief_rationale": "..."}
{"publication_number": "EP2025-300401A1", "gold_label": "borderline", "brief_rationale": "..."}
```

**必須フィールド**:
- `publication_number`: 特許番号（一意識別子）
- `gold_label`: 正解ラベル（"hit", "miss", "borderline"）
- `brief_rationale`: 判定根拠（省略可）

### 3.2 予測結果ファイル（goldset_results.jsonl）
```json
{"publication_number": "JP2025-100001A", "decision": "hit", "confidence": 0.95, "rank": 1, ...}
```

**必須フィールド**:
- `publication_number`: 特許番号（ゴールドラベルとの結合キー）
- `decision`: 予測結果（"hit", "miss"）
- `confidence`: 予測信頼度（0.0-1.0）

## 4. 出力物の詳細

### 4.1 ディレクトリ構造
```
analysis/evaluations/PE_20250901_143052_PatentRadar_v2.0/
├── report.html                         # 統合HTMLレポート
├── data/
│   ├── evaluation_dataset_PE_*.csv     # 評価用統合データ
│   ├── comprehensive_metrics_PE_*.json # 全評価メトリクス
│   └── final_predictions_PE_*.csv      # 最適閾値での予測結果
├── visualizations/
│   ├── roc_analysis_PE_*.png           # ROC曲線・AUC分析
│   ├── prediction_distribution_PE_*.png # 予測確率分布
│   ├── f1_optimization_PE_*.png        # F1-閾値最適化
│   ├── confusion_matrix_comparison_PE_*.png # 混同行列比較
│   └── performance_dashboard_PE_*.png  # 性能ダッシュボード
└── logs/
    └── detailed_evaluation_log_PE_*.txt
```

### 4.2 HTMLレポート内容
1. **評価サマリー**: KPI・総合判定・推奨アクション
2. **データセット分析**: 分布・品質・制約事項
3. **ROC分析**: AUC・識別能力評価
4. **予測分布分析**: 確信度・キャリブレーション品質
5. **閾値最適化**: F1最大化・トレードオフ分析
6. **性能比較**: 標準閾値vs最適閾値
7. **混同行列分析**: 誤分類パターン・対策提案
8. **ビジネス影響評価**: 機会損失・ROI・運用提言
9. **技術仕様**: 再現情報・環境詳細

## 5. 評価履歴管理

### 5.1 履歴一覧確認
```bash
python analysis/evaluation_system/list_evaluations.py
```

出力例:
```
評価履歴一覧:
PE_20250901_143052_PatentRadar_v1.0  AUC: 0.815  F1: 0.800  [2025-09-01 14:30:52]
PE_20250901_203015_PatentRadar_v2.0  AUC: 0.845  F1: 0.825  [2025-09-01 20:30:15]
PE_20250902_091230_PatentRadar_v3.0  AUC: 0.863  F1: 0.840  [2025-09-02 09:12:30]
```

### 5.2 評価比較
```bash
python analysis/evaluation_system/comparison_analyzer.py \
    --eval-ids "PE_20250901_143052_PatentRadar_v1.0,PE_20250901_203015_PatentRadar_v2.0"
```

## 6. カスタマイズ・設定

### 6.1 評価パラメータ調整
`analysis/evaluation_system/config.yaml`:
```yaml
evaluation:
  threshold_resolution: 1000    # 閾値分割数
  random_seed: 42              # 再現性確保
  confidence_intervals: true   # 信頼区間計算
  
visualization:
  dpi: 300                     # 画像解像度
  figure_size: [10, 8]        # 図のサイズ
  color_palette: "colorblind"  # カラーパレット

business_metrics:
  fn_cost: 1000000            # False Negative 1件あたりのコスト（円）
  fp_cost: 7200               # False Positive 1件あたりのコスト（円、2時間×3600円/時）
```

### 6.2 カスタムレポートテンプレート
`analysis/evaluation_system/templates/` 配下のHTMLテンプレートを編集可能

## 7. トラブルシューティング

### 7.1 よくあるエラー

**データ読み込みエラー**:
```
FileNotFoundError: testing/data/labels.jsonl not found
```
→ ファイルパスを確認、相対パスは実行ディレクトリから

**データ結合エラー**:
```
KeyError: No matching publication_numbers found
```
→ ゴールドラベルと予測結果の特許番号一致を確認

**メモリ不足**:
```
MemoryError: Unable to allocate array
```
→ 大規模データの場合はバッチ処理オプションを使用

### 7.2 パフォーマンス最適化

**高速化オプション**:
```bash
python analysis/run_comprehensive_evaluation.py \
    --fast-mode \                    # 閾値分割数を100に削減
    --skip-confidence-intervals \    # 信頼区間計算をスキップ
    --parallel-processing           # 並列処理を有効化
```

**大規模データ対応**:
```bash
python analysis/run_comprehensive_evaluation.py \
    --batch-size 1000 \             # バッチ処理サイズ
    --memory-efficient             # メモリ効率モード
```

## 8. 実行例

### 8.1 初回評価
```bash
# ベースライン評価
python analysis/run_comprehensive_evaluation.py \
    --model-name "PatentRadar_baseline" \
    --description "初期実装版"
```

### 8.2 改良版評価
```bash
# 改良版評価
python analysis/run_comprehensive_evaluation.py \
    --model-name "PatentRadar_v2.0" \
    --description "特徴量エンジニアリング改良版"
```

### 8.3 比較分析
```bash
# 複数版の比較
python analysis/evaluation_system/comparison_analyzer.py \
    --eval-ids "PE_*_PatentRadar_baseline,PE_*_PatentRadar_v2.0" \
    --output "analysis/evaluations/comparison_reports/baseline_vs_v2.html"
```

## 9. ベストプラクティス

### 9.1 評価実行前の確認事項
- [ ] 入力データの形式・完整性確認
- [ ] 分類器名・バージョンの適切な命名
- [ ] 十分なディスク容量の確保
- [ ] 実行環境の依存関係インストール

### 9.2 結果解釈時の注意点
- データセット分析結果を踏まえた解釈を行う
- ビジネス観点での指標の重み付けを考慮する
- 前回評価との差異要因を分析する
- False Negativeの業務影響を特に重視する

### 9.3 継続的改善
- 定期的な評価実行（月次・四半期）
- 評価結果に基づく優先改良項目の特定
- 改良効果の定量的測定・比較
- 評価ルール・基準の見直し・更新

---

**サポート**: 本システムに関する質問・バグレポートは analysis/docs/ISSUES.md を参照してください。