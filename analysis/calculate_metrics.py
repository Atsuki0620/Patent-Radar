#!/usr/bin/env python3
"""
注目特許仕分けくん 性能指標計算スクリプト
軽量版 - 基本ライブラリのみで性能指標を計算
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict, Counter


def load_jsonl(file_path: str) -> List[Dict]:
    """JSONLファイルを読み込み"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json(file_path: str) -> Dict:
    """JSONファイルを読み込み"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """二値分類の基本指標を計算"""
    
    # 混同行列の要素を計算
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    total = tp + fp + tn + fn
    
    # 基本指標
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'false_negative_rate': fn / (tp + fn) if (tp + fn) > 0 else 0.0
    }


def calculate_roc_auc_simple(y_true: List[int], y_scores: List[float]) -> float:
    """簡易ROC AUCの計算（台形公式）"""
    
    # スコア順でソート
    paired_data = list(zip(y_scores, y_true))
    paired_data.sort(reverse=True, key=lambda x: x[0])  # スコア降順
    
    # 異なる閾値での TPR, FPR を計算
    thresholds = sorted(set(y_scores), reverse=True)
    
    roc_points = []
    
    for threshold in thresholds:
        y_pred_threshold = [1 if score >= threshold else 0 for score in y_scores]
        
        tp = sum(1 for t, p in zip(y_true, y_pred_threshold) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred_threshold) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred_threshold) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred_threshold) if t == 1 and p == 0)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        roc_points.append((fpr, tpr))
    
    # (0, 0) と (1, 1) を追加
    roc_points = [(0.0, 0.0)] + roc_points + [(1.0, 1.0)]
    roc_points = sorted(set(roc_points))  # 重複除去とソート
    
    # 台形公式でAUCを計算
    auc = 0.0
    for i in range(1, len(roc_points)):
        x1, y1 = roc_points[i-1]
        x2, y2 = roc_points[i]
        auc += (x2 - x1) * (y1 + y2) / 2.0
    
    return auc


def calculate_precision_at_k(rankings: List[Tuple[int, int]], k_values: List[int]) -> Dict[str, float]:
    """Precision@K を計算
    rankings: [(rank, is_hit), ...] の形式
    """
    precision_at_k = {}
    
    # ランクでソート
    rankings.sort(key=lambda x: x[0])
    
    for k in k_values:
        if k <= len(rankings):
            top_k = rankings[:k]
            hits_in_top_k = sum(1 for _, is_hit in top_k if is_hit == 1)
            precision_at_k[f'precision_at_{k}'] = hits_in_top_k / k
        else:
            precision_at_k[f'precision_at_{k}'] = 0.0
    
    return precision_at_k


def calculate_mean_average_precision(rankings: List[Tuple[int, int]]) -> float:
    """Mean Average Precision (MAP) を計算"""
    
    # ランクでソート
    rankings.sort(key=lambda x: x[0])
    
    average_precisions = []
    hits_found = 0
    
    for i, (rank, is_hit) in enumerate(rankings):
        if is_hit == 1:
            hits_found += 1
            precision_at_i = hits_found / (i + 1)
            average_precisions.append(precision_at_i)
    
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0


def main():
    """メイン計算実行"""
    
    print("=== 注目特許仕分けくん 性能指標計算 ===")
    
    base_path = Path(".")
    
    # データ読み込み
    print("データ読み込み中...")
    results_data = load_jsonl(base_path / "archive" / "outputs" / "testing_results.jsonl")
    
    # ゴールドラベル読み込み
    gold_labels = {}
    gold_data = load_jsonl(base_path / "testing" / "data" / "labels.jsonl")
    for item in gold_data:
        gold_labels[item['publication_number']] = item['gold_label']
    
    # 発明データ読み込み
    invention_data = load_json(base_path / "testing" / "data" / "invention_sample.json")
    
    print(f"テスト結果: {len(results_data)}件")
    print(f"ゴールドラベル: {len(gold_labels)}件")
    
    # データ準備
    evaluation_data = []
    for result in results_data:
        pub_num = result['publication_number']
        gold_label = gold_labels.get(pub_num, 'unknown')
        
        evaluation_data.append({
            'pub_number': pub_num,
            'predicted_decision': result.get('decision', 'miss'),
            'confidence': result.get('confidence', 0.0),
            'rank': result.get('rank', 999),
            'gold_label': gold_label,
            'title': result.get('title', ''),
            'hit_reason_1': result.get('hit_reason_1', ''),
            'gold_rationale': result.get('brief_rationale', '')
        })
    
    # ランクでソート
    evaluation_data.sort(key=lambda x: x['rank'])
    
    print("\\n=== 基本統計 ===")
    gold_counter = Counter(item['gold_label'] for item in evaluation_data)
    pred_counter = Counter(item['predicted_decision'] for item in evaluation_data)
    
    print("ゴールドラベル分布:")
    for label, count in gold_counter.items():
        print(f"  {label}: {count}件")
    
    print("予測分布:")
    for label, count in pred_counter.items():
        print(f"  {label}: {count}件")
    
    # 二値分類データ（borderline除外）
    binary_data = [item for item in evaluation_data if item['gold_label'] in ['hit', 'miss']]
    
    print(f"\\n二値分類対象: {len(binary_data)}件")
    
    # ラベル変換
    y_true = [1 if item['gold_label'] == 'hit' else 0 for item in binary_data]
    y_pred = [1 if item['predicted_decision'] == 'hit' else 0 for item in binary_data]
    y_scores = [item['confidence'] for item in binary_data]
    
    # 二値分類指標
    print("\\n=== 二値分類性能 ===")
    binary_metrics = calculate_binary_metrics(y_true, y_pred)
    
    print(f"総合精度: {binary_metrics['accuracy']:.3f}")
    print(f"HIT検出精度: {binary_metrics['precision']:.3f}")
    print(f"HIT検出再現率: {binary_metrics['recall']:.3f}")
    print(f"F1スコア: {binary_metrics['f1_score']:.3f}")
    print(f"特異度: {binary_metrics['specificity']:.3f}")
    
    cm = binary_metrics['confusion_matrix']
    print("\\n混同行列:")
    print(f"  True Positive:  {cm['tp']}")
    print(f"  False Positive: {cm['fp']}")
    print(f"  True Negative:  {cm['tn']}")
    print(f"  False Negative: {cm['fn']}")
    
    # ROC AUC
    roc_auc = calculate_roc_auc_simple(y_true, y_scores)
    print(f"\\nROC AUC: {roc_auc:.3f}")
    
    # ランキング指標
    print("\\n=== ランキング性能 ===")
    
    # HIT特許のランキング情報
    hit_rankings = [(item['rank'], 1 if item['gold_label'] == 'hit' else 0) for item in evaluation_data]
    
    # Precision@K
    precision_at_k = calculate_precision_at_k(hit_rankings, [5, 10, 20, 30])
    for k_name, precision in precision_at_k.items():
        print(f"{k_name.replace('_', '@').title()}: {precision:.3f}")
    
    # MAP
    map_score = calculate_mean_average_precision(hit_rankings)
    print(f"Mean Average Precision: {map_score:.3f}")
    
    # HIT特許の統計
    hit_data = [item for item in evaluation_data if item['gold_label'] == 'hit']
    if hit_data:
        hit_ranks = [item['rank'] for item in hit_data]
        avg_hit_rank = sum(hit_ranks) / len(hit_ranks)
        best_hit_rank = min(hit_ranks)
        print(f"\\nHIT特許統計:")
        print(f"  総HIT数: {len(hit_data)}")
        print(f"  平均ランク: {avg_hit_rank:.1f}")
        print(f"  最高ランク: {best_hit_rank}")
    
    # トップ10, 20の統計
    top_10 = evaluation_data[:10]
    top_20 = evaluation_data[:20]
    
    hits_in_top_10 = sum(1 for item in top_10 if item['gold_label'] == 'hit')
    hits_in_top_20 = sum(1 for item in top_20 if item['gold_label'] == 'hit')
    
    print(f"  上位10件中のHIT: {hits_in_top_10}")
    print(f"  上位20件中のHIT: {hits_in_top_20}")
    
    # Borderlineケース分析
    print("\\n=== Borderlineケース分析 ===")
    borderline_data = [item for item in evaluation_data if item['gold_label'] == 'borderline']
    
    if borderline_data:
        print(f"Borderline総数: {len(borderline_data)}")
        
        hit_predicted = sum(1 for item in borderline_data if item['predicted_decision'] == 'hit')
        miss_predicted = len(borderline_data) - hit_predicted
        
        print(f"HIT予測: {hit_predicted}件 ({hit_predicted/len(borderline_data)*100:.1f}%)")
        print(f"MISS予測: {miss_predicted}件")
        
        if hit_predicted > 0:
            hit_confidences = [item['confidence'] for item in borderline_data if item['predicted_decision'] == 'hit']
            avg_hit_conf = sum(hit_confidences) / len(hit_confidences)
            print(f"HIT予測時の平均信頼度: {avg_hit_conf:.3f}")
        
        if miss_predicted > 0:
            miss_confidences = [item['confidence'] for item in borderline_data if item['predicted_decision'] == 'miss']
            avg_miss_conf = sum(miss_confidences) / len(miss_confidences)
            print(f"MISS予測時の平均信頼度: {avg_miss_conf:.3f}")
        
        print("\\nBorderlineケース詳細:")
        for item in borderline_data:
            print(f"  {item['pub_number']}: {item['predicted_decision']} (信頼度: {item['confidence']:.3f})")
    else:
        print("Borderlineケースなし")
    
    # エラー分析
    print("\\n=== エラー分析 ===")
    
    # 偽陽性（システム予測HIT、実際MISS）
    false_positives = [item for item in binary_data if item['gold_label'] == 'miss' and item['predicted_decision'] == 'hit']
    
    # 偽陰性（システム予測MISS、実際HIT）
    false_negatives = [item for item in binary_data if item['gold_label'] == 'hit' and item['predicted_decision'] == 'miss']
    
    print(f"偽陽性（False Positive）: {len(false_positives)}件")
    if false_positives:
        fp_avg_conf = sum(item['confidence'] for item in false_positives) / len(false_positives)
        fp_avg_rank = sum(item['rank'] for item in false_positives) / len(false_positives)
        print(f"  平均信頼度: {fp_avg_conf:.3f}")
        print(f"  平均ランク: {fp_avg_rank:.1f}")
        
        print("  主要ケース:")
        for item in false_positives[:3]:
            print(f"    {item['pub_number']}: {item['title'][:50]}... (信頼度: {item['confidence']:.3f})")
    
    print(f"\\n偽陰性（False Negative）: {len(false_negatives)}件")
    if false_negatives:
        fn_avg_conf = sum(item['confidence'] for item in false_negatives) / len(false_negatives)
        fn_avg_rank = sum(item['rank'] for item in false_negatives) / len(false_negatives)
        print(f"  平均信頼度: {fn_avg_conf:.3f}")
        print(f"  平均ランク: {fn_avg_rank:.1f}")
        
        print("  主要ケース:")
        for item in false_negatives[:3]:
            print(f"    {item['pub_number']}: {item['title'][:50]}... (信頼度: {item['confidence']:.3f})")
    
    # 信頼度分析
    print("\\n=== 信頼度分析 ===")
    
    all_confidences = [item['confidence'] for item in binary_data]
    hit_confidences = [item['confidence'] for item in binary_data if item['gold_label'] == 'hit']
    miss_confidences = [item['confidence'] for item in binary_data if item['gold_label'] == 'miss']
    
    print(f"全体信頼度: 平均 {sum(all_confidences)/len(all_confidences):.3f}")
    print(f"HIT信頼度: 平均 {sum(hit_confidences)/len(hit_confidences):.3f}") if hit_confidences else None
    print(f"MISS信頼度: 平均 {sum(miss_confidences)/len(miss_confidences):.3f}") if miss_confidences else None
    
    # 成功基準評価
    print("\\n=== 成功基準評価 ===")
    
    success_criteria = {
        'target_accuracy': 0.8,
        'target_recall': 0.8,
        'acceptable_precision': 0.7
    }
    
    print("成功基準:")
    print(f"  目標総合精度: ≥{success_criteria['target_accuracy']:.1%}")
    print(f"  目標再現率: ≥{success_criteria['target_recall']:.1%}")
    print(f"  許容精度: ≥{success_criteria['acceptable_precision']:.1%}")
    
    print("\\n評価結果:")
    accuracy_ok = binary_metrics['accuracy'] >= success_criteria['target_accuracy']
    recall_ok = binary_metrics['recall'] >= success_criteria['target_recall']
    precision_ok = binary_metrics['precision'] >= success_criteria['acceptable_precision']
    
    print(f"  総合精度: {binary_metrics['accuracy']:.3f} {'✓' if accuracy_ok else '✗'}")
    print(f"  HIT再現率: {binary_metrics['recall']:.3f} {'✓' if recall_ok else '✗'}")
    print(f"  HIT精度: {binary_metrics['precision']:.3f} {'✓' if precision_ok else '✗'}")
    
    overall_success = accuracy_ok and recall_ok and precision_ok
    print(f"\\n総合評価: {'成功基準クリア' if overall_success else '改善が必要'}")
    
    # 運用関連指標
    print("\\n=== 運用指標 ===")
    
    print(f"対象発明: {invention_data.get('title', '')}")
    print(f"処理特許数: {len(results_data)}件")
    print(f"HIT予測: {sum(1 for item in evaluation_data if item['predicted_decision'] == 'hit')}件")
    print(f"MISS予測: {sum(1 for item in evaluation_data if item['predicted_decision'] == 'miss')}件")
    
    # 簡易結果保存
    results_summary = {
        'evaluation_date': datetime.now().isoformat(),
        'total_patents': len(evaluation_data),
        'binary_classification': len(binary_data),
        'borderline_cases': len(borderline_data),
        'metrics': {
            'accuracy': binary_metrics['accuracy'],
            'precision': binary_metrics['precision'],
            'recall': binary_metrics['recall'],
            'f1_score': binary_metrics['f1_score'],
            'roc_auc': roc_auc,
            'map_score': map_score
        },
        'success_criteria_met': {
            'accuracy': accuracy_ok,
            'recall': recall_ok,
            'precision': precision_ok,
            'overall': overall_success
        },
        'ranking_performance': precision_at_k,
        'error_counts': {
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives)
        }
    }
    
    # 結果保存
    output_path = base_path / "analysis" / "metrics_summary.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\\n詳細結果を保存: {output_path}")
    
    return results_summary


if __name__ == "__main__":
    main()