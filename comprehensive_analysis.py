#!/usr/bin/env python3
"""
包括的精度分析スクリプト
ゴールドラベルデータを用いた詳細な性能評価
- 混同行列
- 精度指標（Precision, Recall, F1-score）
- 閾値最適化
- 誤分類分析
- 実用性評価
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, cohen_kappa_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ComprehensiveAnalysis:
    """包括的分析クラス"""
    
    def __init__(self):
        self.predictions = []
        self.gold_labels = {}
        self.analysis_results = {}
        
    def load_data(self):
        """データ読み込み"""
        print("データ読み込み中...")
        
        # 予測結果
        predictions_path = Path("archive/outputs/goldset_results.jsonl")
        with open(predictions_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.predictions.append(json.loads(line))
        
        # ゴールドラベル
        labels_path = Path("tests/data/labels.jsonl")
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    label_data = json.loads(line)
                    self.gold_labels[label_data['publication_number']] = label_data
        
        print(f"予測結果: {len(self.predictions)}件")
        print(f"ゴールドラベル: {len(self.gold_labels)}件")
    
    def prepare_classification_data(self):
        """分類用データ準備"""
        y_true = []
        y_pred = []
        y_score = []
        matched_pairs = []
        
        for pred in self.predictions:
            pub_num = pred['publication_number']
            if pub_num in self.gold_labels:
                gold = self.gold_labels[pub_num]['gold_label']
                pred_decision = pred.get('decision', 'miss')
                confidence = pred.get('confidence', 0.0)
                
                y_true.append(gold)
                y_pred.append(pred_decision)
                y_score.append(confidence)
                
                matched_pairs.append({
                    'pub_number': pub_num,
                    'true_label': gold,
                    'pred_label': pred_decision,
                    'confidence': confidence,
                    'title': pred.get('title', ''),
                    'correct': gold == pred_decision,
                    'pred_data': pred,
                    'gold_data': self.gold_labels[pub_num]
                })
        
        return y_true, y_pred, y_score, matched_pairs
    
    def calculate_confusion_matrix(self, y_true, y_pred):
        """混同行列の計算"""
        print("\n混同行列計算中...")
        
        labels = ['hit', 'borderline', 'miss']
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # 正規化版も計算
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        self.analysis_results['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'matrix_normalized': cm_normalized.tolist(),
            'labels': labels,
            'accuracy': np.trace(cm) / np.sum(cm)
        }
        
        print("混同行列:")
        print(f"          予測")
        print(f"        hit  bord miss")
        for i, true_label in enumerate(labels):
            print(f"実際 {true_label:4s} {cm[i][0]:3d}   {cm[i][1]:3d}  {cm[i][2]:3d}")
        
        return cm, labels
    
    def calculate_classification_metrics(self, y_true, y_pred):
        """分類性能指標の計算"""
        print("\n分類指標計算中...")
        
        # 基本指標
        labels = ['hit', 'borderline', 'miss']
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        
        # マクロ平均
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # マイクロ平均（全体精度）
        micro_precision = micro_recall = micro_f1 = sum([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # クラス別詳細
        class_metrics = {}
        for i, label in enumerate(labels):
            class_metrics[label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        self.analysis_results['classification_metrics'] = {
            'class_metrics': class_metrics,
            'macro_avg': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1_score': float(macro_f1)
            },
            'micro_avg': {
                'precision': float(micro_precision),
                'recall': float(micro_recall),
                'f1_score': float(micro_f1)
            },
            'overall_accuracy': float(micro_precision),
            'cohens_kappa': float(kappa)
        }
        
        print("クラス別性能:")
        for label, metrics in class_metrics.items():
            print(f"{label:10s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}, Support={metrics['support']}")
        print(f"Overall Accuracy: {micro_precision:.3f}")
        print(f"Cohen's Kappa: {kappa:.3f}")
    
    def threshold_optimization(self, y_true, y_score, matched_pairs):
        """閾値最適化分析"""
        print("\n閾値最適化分析中...")
        
        # 二値分類用データ準備（hit vs non-hit）
        y_binary = [1 if label == 'hit' else 0 for label in y_true]
        
        # ROC曲線
        fpr, tpr, roc_thresholds = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        # PR曲線
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_binary, y_score)
        avg_precision = average_precision_score(y_binary, y_score)
        
        # 異なる閾値での性能評価
        threshold_analysis = []
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for threshold in thresholds:
            # 閾値適用
            y_pred_threshold = []
            for i, pair in enumerate(matched_pairs):
                confidence = pair['confidence']
                if confidence >= threshold:
                    y_pred_threshold.append('hit')
                elif confidence >= threshold * 0.6:  # borderline範囲
                    y_pred_threshold.append('borderline') 
                else:
                    y_pred_threshold.append('miss')
            
            # 性能計算
            accuracy = sum([1 for t, p in zip(y_true, y_pred_threshold) if t == p]) / len(y_true)
            
            # hit検出性能
            hit_precision = 0
            hit_recall = 0
            hit_f1 = 0
            
            hit_tp = sum([1 for t, p in zip(y_true, y_pred_threshold) if t == 'hit' and p == 'hit'])
            hit_fp = sum([1 for t, p in zip(y_true, y_pred_threshold) if t != 'hit' and p == 'hit'])
            hit_fn = sum([1 for t, p in zip(y_true, y_pred_threshold) if t == 'hit' and p != 'hit'])
            
            if hit_tp + hit_fp > 0:
                hit_precision = hit_tp / (hit_tp + hit_fp)
            if hit_tp + hit_fn > 0:
                hit_recall = hit_tp / (hit_tp + hit_fn)
            if hit_precision + hit_recall > 0:
                hit_f1 = 2 * hit_precision * hit_recall / (hit_precision + hit_recall)
            
            threshold_analysis.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'hit_precision': hit_precision,
                'hit_recall': hit_recall,
                'hit_f1': hit_f1
            })
        
        # 最適閾値選択（F1スコア最大）
        best_threshold_data = max(threshold_analysis, key=lambda x: x['hit_f1'])
        
        self.analysis_results['threshold_optimization'] = {
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'auc': float(roc_auc)
            },
            'pr_curve': {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(), 
                'thresholds': pr_thresholds.tolist(),
                'avg_precision': float(avg_precision)
            },
            'threshold_analysis': threshold_analysis,
            'optimal_threshold': best_threshold_data
        }
        
        print(f"ROC AUC: {roc_auc:.3f}")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"最適閾値: {best_threshold_data['threshold']} (F1={best_threshold_data['hit_f1']:.3f})")
    
    def misclassification_analysis(self, matched_pairs):
        """誤分類分析"""
        print("\n誤分類分析中...")
        
        # 誤分類事例の抽出
        false_positives = []  # 誤ってHITと判定
        false_negatives = []  # HITを見逃し
        borderline_errors = []  # Borderline関連エラー
        
        for pair in matched_pairs:
            true_label = pair['true_label']
            pred_label = pair['pred_label']
            
            if not pair['correct']:
                if pred_label == 'hit' and true_label != 'hit':
                    false_positives.append(pair)
                elif true_label == 'hit' and pred_label != 'hit':
                    false_negatives.append(pair)
                elif 'borderline' in [true_label, pred_label]:
                    borderline_errors.append(pair)
        
        # エラーパターン分析
        error_patterns = Counter()
        for pair in matched_pairs:
            if not pair['correct']:
                pattern = f"{pair['true_label']} -> {pair['pred_label']}"
                error_patterns[pattern] += 1
        
        self.analysis_results['misclassification_analysis'] = {
            'false_positives': [self._simplify_pair(p) for p in false_positives[:10]],
            'false_negatives': [self._simplify_pair(p) for p in false_negatives[:10]],
            'borderline_errors': [self._simplify_pair(p) for p in borderline_errors[:10]],
            'error_patterns': dict(error_patterns),
            'summary': {
                'total_errors': len([p for p in matched_pairs if not p['correct']]),
                'false_positive_count': len(false_positives),
                'false_negative_count': len(false_negatives),
                'borderline_error_count': len(borderline_errors)
            }
        }
        
        print(f"誤分類総数: {len([p for p in matched_pairs if not p['correct']])}")
        print(f"False Positives: {len(false_positives)}")
        print(f"False Negatives: {len(false_negatives)}")
        print(f"Borderline エラー: {len(borderline_errors)}")
        print("エラーパターン:")
        for pattern, count in error_patterns.most_common():
            print(f"  {pattern}: {count}件")
    
    def business_impact_analysis(self, matched_pairs):
        """実用性・ビジネス効果分析"""
        print("\n実用性分析中...")
        
        # 基本統計
        total_patents = len(matched_pairs)
        correct_classifications = len([p for p in matched_pairs if p['correct']])
        
        # HIT検出性能
        true_hits = len([p for p in matched_pairs if p['true_label'] == 'hit'])
        detected_hits = len([p for p in matched_pairs if p['pred_label'] == 'hit'])
        correct_hits = len([p for p in matched_pairs if p['true_label'] == 'hit' and p['pred_label'] == 'hit'])
        
        # 作業削減効果の試算
        # 前提: 手動確認なしなら全件確認が必要
        manual_review_all = total_patents * 15  # 15分/件
        
        # システム利用時: HITのみ詳細確認 + 境界線ケース確認
        hit_reviews = detected_hits * 10  # 10分/件（HIT詳細確認）
        borderline_reviews = len([p for p in matched_pairs if p['pred_label'] == 'borderline']) * 15  # 15分/件
        system_review_time = hit_reviews + borderline_reviews
        
        time_saved = manual_review_all - system_review_time
        efficiency_gain = time_saved / manual_review_all * 100
        
        # リスク分析
        missed_hits = len([p for p in matched_pairs if p['true_label'] == 'hit' and p['pred_label'] == 'miss'])
        risk_score = missed_hits / true_hits * 100 if true_hits > 0 else 0
        
        # コスト試算（1時間あたり5000円と仮定）
        hourly_rate = 5000
        cost_saved = (time_saved / 60) * hourly_rate
        
        self.analysis_results['business_impact'] = {
            'efficiency_analysis': {
                'total_patents': total_patents,
                'manual_review_time_minutes': manual_review_all,
                'system_review_time_minutes': system_review_time,
                'time_saved_minutes': time_saved,
                'efficiency_gain_percent': efficiency_gain,
                'cost_saved_yen': cost_saved
            },
            'risk_analysis': {
                'true_hit_count': true_hits,
                'detected_hit_count': detected_hits,
                'correct_hit_count': correct_hits,
                'missed_hit_count': missed_hits,
                'hit_detection_rate': correct_hits / true_hits * 100 if true_hits > 0 else 0,
                'miss_risk_percent': risk_score
            },
            'quality_metrics': {
                'overall_accuracy': correct_classifications / total_patents * 100,
                'precision_hit': correct_hits / detected_hits * 100 if detected_hits > 0 else 0,
                'recall_hit': correct_hits / true_hits * 100 if true_hits > 0 else 0
            }
        }
        
        print(f"作業時間削減: {time_saved:.0f}分 ({efficiency_gain:.1f}%削減)")
        print(f"コスト削減: {cost_saved:,.0f}円")
        print(f"HIT検出率: {correct_hits}/{true_hits} ({correct_hits/true_hits*100:.1f}%)")
        print(f"見逃しリスク: {risk_score:.1f}%")
    
    def _simplify_pair(self, pair):
        """表示用にペアデータを簡素化"""
        return {
            'pub_number': pair['pub_number'],
            'title': pair['title'][:50] + '...' if len(pair['title']) > 50 else pair['title'],
            'true_label': pair['true_label'],
            'pred_label': pair['pred_label'],
            'confidence': round(pair['confidence'], 3),
            'rationale': pair['gold_data'].get('brief_rationale', '')[:100] + '...' if len(pair['gold_data'].get('brief_rationale', '')) > 100 else pair['gold_data'].get('brief_rationale', '')
        }
    
    def save_analysis_results(self):
        """分析結果の保存"""
        output_dir = Path("archive/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # メイン分析結果
        results_path = output_dir / "comprehensive_analysis_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'data_source': 'goldset (60 patents)',
                    'analysis_version': '1.0'
                },
                'results': self.analysis_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n分析結果保存: {results_path}")
        return results_path
    
    def run_comprehensive_analysis(self):
        """包括分析の実行"""
        print("="*80)
        print("包括的精度分析開始")
        print("="*80)
        
        # データ読み込み
        self.load_data()
        
        # 分類用データ準備
        y_true, y_pred, y_score, matched_pairs = self.prepare_classification_data()
        
        # 各種分析実行
        self.calculate_confusion_matrix(y_true, y_pred)
        self.calculate_classification_metrics(y_true, y_pred)
        self.threshold_optimization(y_true, y_score, matched_pairs)
        self.misclassification_analysis(matched_pairs)
        self.business_impact_analysis(matched_pairs)
        
        # 結果保存
        results_path = self.save_analysis_results()
        
        print("="*80)
        print("包括分析完了")
        print("="*80)
        
        return results_path

def main():
    analyzer = ComprehensiveAnalysis()
    results_path = analyzer.run_comprehensive_analysis()
    
    print(f"\n全分析完了: {results_path}")
    return results_path

if __name__ == "__main__":
    main()