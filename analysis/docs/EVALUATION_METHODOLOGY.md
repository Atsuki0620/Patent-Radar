# Patent-Radar 評価手法説明書

## 1. 評価手法概要

### 1.1 評価目的
特許分類システム「Patent-Radar」の二値分類性能（HIT/MISS判定）を、ビジネス要件に基づいて多角的に評価し、実運用における最適設定を決定する。

### 1.2 評価原則
- **厳密性**: 1000分割閾値探索による精密な最適化
- **網羅性**: ROC・分布・混同行列・ビジネス指標の統合分析
- **実用性**: 特許業務の特性を考慮したビジネス観点での評価
- **再現性**: 固定乱数シード・詳細ログによる完全再現性確保

## 2. データ準備・前処理

### 2.1 データソース
- **ゴールドスタンダード**: 専門家アノテーション付き特許データ（testing/data/labels.jsonl）
- **予測結果**: 分類器による予測結果（archive/outputs/goldset_results.jsonl）
- **結合キー**: publication_number による完全内結合

### 2.2 前処理手順
1. **データ統合**: ゴールドラベルと予測結果の結合
2. **品質チェック**: 欠損値・重複・異常値の検出・処理
3. **クラス変換**: hit=1, miss=0 の数値変換（borderlineは除外）
4. **確率正規化**: confidence値の[0,1]範囲正規化・検証

### 2.3 データ品質評価
- **完整性**: 欠損値・NULL値の割合・処理方法
- **一致性**: publication_number の一致率・不一致理由
- **代表性**: サンプルの時系列・技術分野・難易度分布
- **均衡性**: クラス不均衡度・統計的偏りの評価

## 3. 評価メトリクス体系

### 3.1 基本分類指標
```
Accuracy = (TP + TN) / (TP + FP + TN + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
Specificity = TN / (TN + FP)
```

### 3.2 確率的評価指標
- **ROC-AUC**: 閾値によらない識別能力（0.5=ランダム, 1.0=完全分離）
- **PR-AUC**: 不均衡データに頑健なPrecision-Recall AUC
- **Calibration**: 予測確率の実際確率との一致度
- **Distribution Overlap**: HIT/MISS予測分布の重なり度

### 3.3 特許業務特化指標
- **False Negative Rate**: 見逃し率（機会損失直結）
- **False Positive Rate**: 誤検出率（工数増加要因）
- **Hit Detection Efficiency**: HIT特許の上位集中度
- **Review Workload**: 人的確認が必要な件数推定

### 3.4 ビジネス指標
```
機会損失コスト = FN件数 × 1件あたり見逃しコスト（100万円）
工数増加コスト = FP件数 × 1件あたり確認工数（2時間×時給）
システム価値 = 削減工数価値 - 機会損失コスト - システム運用コスト
```

## 4. 閾値最適化手法

### 4.1 最適化アルゴリズム
```python
# F1スコア最大化による最適閾値決定
thresholds = np.linspace(0.000, 1.000, 1001)  # 1000分割
f1_scores = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_scores.append(f1)

optimal_threshold = thresholds[np.argmax(f1_scores)]
max_f1_score = max(f1_scores)
```

### 4.2 閾値設定戦略
- **Conservative（保守的）**: Precision優先、誤検出最小化
- **Balanced（均衡）**: F1スコア最大化、バランス重視
- **Aggressive（積極的）**: Recall優先、見逃し最小化

### 4.3 感度分析
各閾値での性能指標変化を分析し、閾値変更の影響範囲・安定性を評価

## 5. ROC分析手法

### 5.1 ROC曲線生成
```python
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
auc_score = auc(fpr, tpr)

# 最適動作点（左上に最も近い点）
optimal_idx = np.argmax(tpr - fpr)
optimal_fpr = fpr[optimal_idx]
optimal_tpr = tpr[optimal_idx]
```

### 5.2 AUC解釈基準
- **0.95-1.00**: 優秀（実用化強く推奨）
- **0.90-0.95**: 良好（実用化推奨）
- **0.80-0.90**: 改善要（追加開発後に実用化）
- **0.70-0.80**: 要再設計（大幅改良必要）
- **0.50-0.70**: 実用化困難

### 5.3 信頼区間推定
ブートストラップ法によるAUC信頼区間推定（95%信頼区間）

## 6. 予測分布分析

### 6.1 分布特性評価
- **中心傾向**: HIT/MISS別の平均・中央値・モード
- **散らばり**: 標準偏差・四分位範囲・分布形状
- **分離度**: 分布間距離・重なり領域の割合
- **偏り**: 分布の歪度・尖度・極端値の影響

### 6.2 キャリブレーション評価
```python
# 信頼性図（Reliability Diagram）
bin_boundaries = np.linspace(0, 1, 11)
bin_lowers = bin_boundaries[:-1]
bin_uppers = bin_boundaries[1:]

for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
    in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
    prop_in_bin = in_bin.mean()
    
    if prop_in_bin > 0:
        accuracy_in_bin = y_true[in_bin].mean()
        avg_confidence_in_bin = y_proba[in_bin].mean()
```

## 7. 混同行列分析

### 7.1 混同行列生成
```python
# 標準閾値（0.5）と最適閾値での比較
cm_standard = confusion_matrix(y_true, y_proba >= 0.5)
cm_optimal = confusion_matrix(y_true, y_proba >= optimal_threshold)

# 正規化版も生成（行和で正規化）
cm_normalized = cm_optimal.astype('float') / cm_optimal.sum(axis=1)[:, np.newaxis]
```

### 7.2 誤分類分析
- **False Negative分析**: 見逃された特許の特徴・パターン抽出
- **False Positive分析**: 誤検出された特許の特徴・共通点分析
- **境界事例分析**: 予測確率0.4-0.6の困難事例の詳細検証

## 8. 統計的有意性検定

### 8.1 改善効果の統計検定
```python
# McNemar検定（2つの分類器の比較）
from statsmodels.stats.contingency_tables import mcnemar

# 分割表作成
table = [[n_both_wrong, n_model1_wrong], 
         [n_model2_wrong, n_both_correct]]

result = mcnemar(table, exact=True)
p_value = result.pvalue
```

### 8.2 信頼区間
- **AUC信頼区間**: ブートストラップ法（n=1000）
- **精度信頼区間**: 二項分布に基づく信頼区間
- **改善効果**: 差の信頼区間・効果量推定

## 9. ビジネス影響評価手法

### 9.1 コスト・ベネフィット分析
```python
# 機会損失コスト
opportunity_cost = fn_count × COST_PER_MISSED_PATENT

# 工数増加コスト
labor_cost = fp_count × HOURS_PER_REVIEW × HOURLY_RATE

# システム価値
total_patents = len(y_true)
manual_review_cost = total_patents × HOURS_PER_MANUAL_REVIEW × HOURLY_RATE
automated_review_cost = opportunity_cost + labor_cost + SYSTEM_OPERATION_COST

system_value = manual_review_cost - automated_review_cost
roi = (system_value - SYSTEM_DEVELOPMENT_COST) / SYSTEM_DEVELOPMENT_COST
```

### 9.2 リスク評価
- **見逃しリスク**: False Negative による機会損失・競争劣位
- **誤検出リスク**: False Positive による工数増・効率悪化
- **システムリスク**: 過信・誤用による判断ミス

## 10. 継続改善手法

### 10.1 性能劣化検出
- **データドリフト**: 入力データ分布の変化検出
- **概念ドリフト**: ラベル定義・基準の変化検出
- **性能ドリフト**: 時系列での性能指標変化監視

### 10.2 改善効果測定
- **A/Bテスト**: 新旧システムの並行運用・比較
- **段階導入**: 部分的導入による効果測定・リスク管理
- **フィードバック収集**: ユーザー評価・業務影響の定性評価

## 11. 品質保証手法

### 11.1 交差検証
```python
from sklearn.model_selection import StratifiedKFold

# 5-fold交差検証による頑健性評価
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in skf.split(X, y):
    # 各フォールドでの評価
    auc_fold = roc_auc_score(y[val_idx], y_proba[val_idx])
    cv_scores.append(auc_fold)

mean_auc = np.mean(cv_scores)
std_auc = np.std(cv_scores)
```

### 11.2 外部検証
- **時系列分割**: 過去データでの学習・未来データでの検証
- **技術分野別**: 分野ごとの性能差・汎化性能の確認
- **難易度別**: 明確・境界・困難事例での性能分析

---

**更新履歴**:
- v1.0 (2025-09-01): 初版作成、基本手法確立
- 今後の更新予定: 評価結果蓄積に基づく手法改良・新指標追加