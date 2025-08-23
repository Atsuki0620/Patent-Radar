#!/usr/bin/env python3
"""
注目特許仕分けくん 最終分析スクリプト
実データに基づいた正確な性能指標計算と最終レポート生成
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter


def calculate_actual_metrics():
    """実データから正確な性能指標を計算"""
    
    print("=== 注目特許仕分けくん 最終性能分析 ===")
    
    # データ読み込み
    base_path = Path(".")
    
    # テスト結果読み込み
    results_data = []
    with open(base_path / "archive" / "outputs" / "testing_results.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results_data.append(json.loads(line))
    
    # ゴールドラベル読み込み
    gold_labels = {}
    with open(base_path / "testing" / "data" / "labels.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_labels[item['publication_number']] = item
    
    print(f"テスト結果: {len(results_data)}件")
    print(f"ゴールドラベル: {len(gold_labels)}件")
    
    # 評価データ準備
    evaluation_data = []
    for result in results_data:
        pub_num = result['publication_number'] 
        gold_info = gold_labels.get(pub_num, {})
        
        evaluation_data.append({
            'pub_number': pub_num,
            'predicted_decision': result.get('decision', 'miss'),
            'confidence': result.get('confidence', 0.0),
            'rank': result.get('rank', 999),
            'gold_label': gold_info.get('gold_label', 'unknown'),
            'gold_rationale': gold_info.get('brief_rationale', ''),
            'title': result.get('title', ''),
            'hit_reason_1': result.get('hit_reason_1', ''),
            'brief_rationale': result.get('brief_rationale', '')
        })
    
    # ランクでソート
    evaluation_data.sort(key=lambda x: x['rank'])
    
    # 基本統計
    print("\\n=== 基本統計 ===")
    gold_dist = Counter(item['gold_label'] for item in evaluation_data)
    pred_dist = Counter(item['predicted_decision'] for item in evaluation_data)
    
    print("ゴールドラベル分布:")
    for label, count in gold_dist.items():
        print(f"  {label}: {count}件")
    
    print("\\n予測分布:")
    for label, count in pred_dist.items():
        print(f"  {label}: {count}件")
    
    # 二値分類データ（borderline除外）
    binary_data = [item for item in evaluation_data if item['gold_label'] in ['hit', 'miss']]
    borderline_data = [item for item in evaluation_data if item['gold_label'] == 'borderline']
    
    print(f"\\n二値分類対象: {len(binary_data)}件")
    print(f"Borderlineケース: {len(borderline_data)}件")
    
    # 混同行列計算
    tp = len([item for item in binary_data if item['gold_label'] == 'hit' and item['predicted_decision'] == 'hit'])
    fp = len([item for item in binary_data if item['gold_label'] == 'miss' and item['predicted_decision'] == 'hit'])
    tn = len([item for item in binary_data if item['gold_label'] == 'miss' and item['predicted_decision'] == 'miss'])
    fn = len([item for item in binary_data if item['gold_label'] == 'hit' and item['predicted_decision'] == 'miss'])
    
    total_binary = tp + fp + tn + fn
    
    print("\\n=== 混同行列 ===")
    print(f"True Positive:  {tp}")
    print(f"False Positive: {fp}")
    print(f"True Negative:  {tn}")
    print(f"False Negative: {fn}")
    print(f"総計: {total_binary}")
    
    # 基本指標
    accuracy = (tp + tn) / total_binary if total_binary > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\\n=== 二値分類性能 ===")
    print(f"総合精度: {accuracy:.3f} ({accuracy:.1%})")
    print(f"HIT検出精度: {precision:.3f} ({precision:.1%})")
    print(f"HIT検出再現率: {recall:.3f} ({recall:.1%})")
    print(f"特異度: {specificity:.3f} ({specificity:.1%})")
    print(f"F1スコア: {f1_score:.3f}")
    
    # ROC AUC簡易計算
    y_true = [1 if item['gold_label'] == 'hit' else 0 for item in binary_data]
    y_scores = [item['confidence'] for item in binary_data]
    
    # 簡易AUC計算（Wilcoxon-Mann-Whitney統計量ベース）
    pos_scores = [score for score, true_label in zip(y_scores, y_true) if true_label == 1]
    neg_scores = [score for score, true_label in zip(y_scores, y_true) if true_label == 0]
    
    if pos_scores and neg_scores:
        # すべての正例・負例ペアの比較
        comparisons = 0
        wins = 0
        for pos_score in pos_scores:
            for neg_score in neg_scores:
                comparisons += 1
                if pos_score > neg_score:
                    wins += 1
                elif pos_score == neg_score:
                    wins += 0.5
        
        roc_auc = wins / comparisons if comparisons > 0 else 0.5
    else:
        roc_auc = 0.5
    
    print(f"ROC AUC: {roc_auc:.3f}")
    
    # ランキング分析
    print("\\n=== ランキング分析 ===")
    
    # HIT特許の統計
    hit_data = [item for item in evaluation_data if item['gold_label'] == 'hit']
    if hit_data:
        hit_ranks = [item['rank'] for item in hit_data]
        avg_hit_rank = sum(hit_ranks) / len(hit_ranks)
        best_hit_rank = min(hit_ranks)
        worst_hit_rank = max(hit_ranks)
        
        print(f"HIT特許統計:")
        print(f"  総数: {len(hit_data)}")
        print(f"  平均ランク: {avg_hit_rank:.1f}")
        print(f"  最高ランク: {best_hit_rank}")
        print(f"  最低ランク: {worst_hit_rank}")
    
    # Precision@K
    precision_at_k = {}
    for k in [5, 10, 20, 30]:
        if k <= len(evaluation_data):
            top_k = evaluation_data[:k]
            hits_in_top_k = len([item for item in top_k if item['gold_label'] == 'hit'])
            precision_at_k[f'precision_at_{k}'] = hits_in_top_k / k
            print(f"  Precision@{k}: {hits_in_top_k / k:.3f} ({hits_in_top_k}/{k})")
    
    # MAP計算
    average_precisions = []
    hits_found = 0
    for i, item in enumerate(evaluation_data):
        if item['gold_label'] == 'hit':
            hits_found += 1
            precision_at_i = hits_found / (i + 1)
            average_precisions.append(precision_at_i)
    
    map_score = sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    print(f"  Mean Average Precision: {map_score:.3f}")
    
    # Borderlineケース分析
    print("\\n=== Borderlineケース分析 ===")
    if borderline_data:
        hit_predicted = len([item for item in borderline_data if item['predicted_decision'] == 'hit'])
        miss_predicted = len(borderline_data) - hit_predicted
        
        print(f"Borderline総数: {len(borderline_data)}")
        print(f"HIT予測: {hit_predicted}件 ({hit_predicted/len(borderline_data)*100:.1f}%)")
        print(f"MISS予測: {miss_predicted}件")
        
        if hit_predicted > 0:
            hit_conf = [item['confidence'] for item in borderline_data if item['predicted_decision'] == 'hit']
            avg_hit_conf = sum(hit_conf) / len(hit_conf)
            print(f"HIT予測時の平均信頼度: {avg_hit_conf:.3f}")
        
        if miss_predicted > 0:
            miss_conf = [item['confidence'] for item in borderline_data if item['predicted_decision'] == 'miss']
            avg_miss_conf = sum(miss_conf) / len(miss_conf)
            print(f"MISS予測時の平均信頼度: {avg_miss_conf:.3f}")
    
    # エラー分析
    print("\\n=== エラー分析 ===")
    
    false_positives = [item for item in binary_data if item['gold_label'] == 'miss' and item['predicted_decision'] == 'hit']
    false_negatives = [item for item in binary_data if item['gold_label'] == 'hit' and item['predicted_decision'] == 'miss']
    
    print(f"偽陽性（False Positive）: {len(false_positives)}件")
    if false_positives:
        fp_avg_conf = sum(item['confidence'] for item in false_positives) / len(false_positives)
        fp_avg_rank = sum(item['rank'] for item in false_positives) / len(false_positives)
        print(f"  平均信頼度: {fp_avg_conf:.3f}")
        print(f"  平均ランク: {fp_avg_rank:.1f}")
    
    print(f"\\n偽陰性（False Negative）: {len(false_negatives)}件")
    if false_negatives:
        fn_avg_conf = sum(item['confidence'] for item in false_negatives) / len(false_negatives)
        fn_avg_rank = sum(item['rank'] for item in false_negatives) / len(false_negatives)
        print(f"  平均信頼度: {fn_avg_conf:.3f}")
        print(f"  平均ランク: {fn_avg_rank:.1f}")
        
        print("  詳細:")
        for item in false_negatives:
            print(f"    {item['pub_number']}: {item['title'][:50]}... (信頼度: {item['confidence']:.3f}, ランク: {item['rank']})")
    
    # 成功基準評価
    print("\\n=== 成功基準評価 ===")
    
    success_criteria = {
        'target_accuracy': 0.8,
        'target_recall': 0.8, 
        'acceptable_precision': 0.7
    }
    
    accuracy_ok = accuracy >= success_criteria['target_accuracy']
    recall_ok = recall >= success_criteria['target_recall']
    precision_ok = precision >= success_criteria['acceptable_precision']
    overall_success = accuracy_ok and recall_ok and precision_ok
    
    print(f"目標値:")
    print(f"  総合精度: ≥{success_criteria['target_accuracy']:.1%}")
    print(f"  HIT再現率: ≥{success_criteria['target_recall']:.1%}")
    print(f"  HIT精度: ≥{success_criteria['acceptable_precision']:.1%}")
    
    print(f"\\n評価結果:")
    print(f"  総合精度: {accuracy:.1%} {'✓' if accuracy_ok else '✗'}")
    print(f"  HIT再現率: {recall:.1%} {'✓' if recall_ok else '✗'}")
    print(f"  HIT精度: {precision:.1%} {'✓' if precision_ok else '✗'}")
    print(f"  総合判定: {'成功基準クリア' if overall_success else '改善が必要'}")
    
    # 結果まとめ
    final_results = {
        'evaluation_metadata': {
            'total_patents': len(evaluation_data),
            'binary_classification_count': len(binary_data),
            'borderline_count': len(borderline_data),
            'evaluation_date': datetime.now().isoformat()
        },
        'confusion_matrix': {
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn
        },
        'performance_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'map_score': map_score
        },
        'ranking_metrics': precision_at_k,
        'success_criteria_assessment': {
            'target_accuracy': success_criteria['target_accuracy'],
            'target_recall': success_criteria['target_recall'],
            'acceptable_precision': success_criteria['acceptable_precision'],
            'accuracy_met': accuracy_ok,
            'recall_met': recall_ok,
            'precision_met': precision_ok,
            'overall_success': overall_success
        },
        'error_analysis': {
            'false_positives': {
                'count': len(false_positives),
                'avg_confidence': sum(item['confidence'] for item in false_positives) / len(false_positives) if false_positives else 0.0,
                'avg_rank': sum(item['rank'] for item in false_positives) / len(false_positives) if false_positives else 0.0
            },
            'false_negatives': {
                'count': len(false_negatives),
                'avg_confidence': sum(item['confidence'] for item in false_negatives) / len(false_negatives) if false_negatives else 0.0,
                'avg_rank': sum(item['rank'] for item in false_negatives) / len(false_negatives) if false_negatives else 0.0,
                'cases': [{'pub_number': item['pub_number'], 'title': item['title'], 'confidence': item['confidence'], 'rank': item['rank']} for item in false_negatives]
            }
        },
        'borderline_analysis': {
            'total_count': len(borderline_data),
            'hit_predicted': len([item for item in borderline_data if item['predicted_decision'] == 'hit']),
            'miss_predicted': len([item for item in borderline_data if item['predicted_decision'] == 'miss']),
            'hit_prediction_rate': len([item for item in borderline_data if item['predicted_decision'] == 'hit']) / len(borderline_data) * 100 if borderline_data else 0.0
        }
    }
    
    # 結果保存
    output_path = base_path / "analysis" / "final_evaluation_results.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n詳細結果を保存: {output_path}")
    
    return final_results


if __name__ == "__main__":
    results = calculate_actual_metrics()