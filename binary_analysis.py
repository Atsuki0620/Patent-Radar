#!/usr/bin/env python3
"""
二値分類システム用の精度分析
HIT/MISSの二値判定システムに特化した分析
BORDERLINEケースの詳細分析を含む
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

class BinaryClassificationAnalysis:
    """二値分類システム専用分析クラス"""
    
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
    
    def prepare_binary_analysis(self):
        """二値分類用データ準備"""
        matched_pairs = []
        
        for pred in self.predictions:
            pub_num = pred['publication_number']
            if pub_num in self.gold_labels:
                gold_label = self.gold_labels[pub_num]['gold_label']
                pred_decision = pred.get('decision', 'miss')
                confidence = pred.get('confidence', 0.0)
                
                matched_pairs.append({
                    'pub_number': pub_num,
                    'gold_label': gold_label,
                    'pred_label': pred_decision,
                    'confidence': confidence,
                    'title': pred.get('title', ''),
                    'pred_data': pred,
                    'gold_data': self.gold_labels[pub_num]
                })
        
        return matched_pairs
    
    def calculate_binary_confusion_matrix(self, matched_pairs):
        """二値分類用混同行列計算"""
        print("\n二値分類混同行列計算中...")
        
        # TP, TN, FP, FN の計算
        tp = 0  # True Positive: 正解HIT → 予測HIT
        tn = 0  # True Negative: 正解MISS → 予測MISS
        fp = 0  # False Positive: 正解MISS → 予測HIT
        fn = 0  # False Negative: 正解HIT → 予測MISS
        
        # BORDERLINEケースの詳細
        borderline_cases = []
        borderline_hit_pred = 0
        borderline_miss_pred = 0
        
        for pair in matched_pairs:
            gold = pair['gold_label']
            pred = pair['pred_label']
            
            if gold == 'borderline':
                borderline_cases.append(pair)
                if pred == 'hit':
                    borderline_hit_pred += 1
                else:
                    borderline_miss_pred += 1
            elif gold == 'hit':
                if pred == 'hit':
                    tp += 1
                else:
                    fn += 1
            elif gold == 'miss':
                if pred == 'hit':
                    fp += 1
                else:
                    tn += 1
        
        # 精度指標計算
        total_clear = tp + tn + fp + fn  # BORDERLINE除く
        accuracy = (tp + tn) / total_clear if total_clear > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        self.analysis_results['binary_confusion_matrix'] = {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'total_clear_cases': total_clear,
            'borderline_count': len(borderline_cases),
            'borderline_hit_pred': borderline_hit_pred,
            'borderline_miss_pred': borderline_miss_pred
        }
        
        self.analysis_results['borderline_analysis'] = {
            'cases': [self._simplify_borderline_case(case) for case in borderline_cases],
            'total_count': len(borderline_cases),
            'hit_predictions': borderline_hit_pred,
            'miss_predictions': borderline_miss_pred,
            'hit_ratio': borderline_hit_pred / len(borderline_cases) if borderline_cases else 0
        }
        
        print("二値分類混同行列:")
        print(f"            予測")
        print(f"         HIT  MISS")
        print(f"実際 HIT  {tp:3d}   {fn:3d}")
        print(f"     MISS {fp:3d}   {tn:3d}")
        print(f"\nBORDERLINEケース: {len(borderline_cases)}件")
        print(f"  → HIT予測: {borderline_hit_pred}件")
        print(f"  → MISS予測: {borderline_miss_pred}件")
        print(f"精度: {accuracy:.3f}, 精密度: {precision:.3f}, 再現率: {recall:.3f}, F1: {f1_score:.3f}")
    
    def analyze_borderline_cases(self, matched_pairs):
        """BORDERLINE症例の詳細分析"""
        print("\nBORDERLINEケース詳細分析中...")
        
        borderline_cases = [p for p in matched_pairs if p['gold_label'] == 'borderline']
        
        if not borderline_cases:
            print("BORDERLINEケースが見つかりません")
            return
        
        # 信頼度による分析
        hit_predicted = [case for case in borderline_cases if case['pred_label'] == 'hit']
        miss_predicted = [case for case in borderline_cases if case['pred_label'] == 'miss']
        
        # 信頼度統計
        hit_confidences = [case['confidence'] for case in hit_predicted]
        miss_confidences = [case['confidence'] for case in miss_predicted]
        
        print(f"BORDERLINE → HIT予測: {len(hit_predicted)}件")
        if hit_confidences:
            print(f"  平均信頼度: {np.mean(hit_confidences):.3f}")
            print(f"  信頼度範囲: {min(hit_confidences):.3f} - {max(hit_confidences):.3f}")
        
        print(f"BORDERLINE → MISS予測: {len(miss_predicted)}件")
        if miss_confidences:
            print(f"  平均信頼度: {np.mean(miss_confidences):.3f}")
            print(f"  信頼度範囲: {min(miss_confidences):.3f} - {max(miss_confidences):.3f}")
        
        # 理由分析
        hit_reasons = [case['gold_data'].get('brief_rationale', '') for case in hit_predicted]
        miss_reasons = [case['gold_data'].get('brief_rationale', '') for case in miss_predicted]
        
        self.analysis_results['borderline_detailed_analysis'] = {
            'hit_predicted_cases': [self._simplify_borderline_case(case) for case in hit_predicted],
            'miss_predicted_cases': [self._simplify_borderline_case(case) for case in miss_predicted],
            'hit_confidence_stats': {
                'mean': np.mean(hit_confidences) if hit_confidences else 0,
                'min': min(hit_confidences) if hit_confidences else 0,
                'max': max(hit_confidences) if hit_confidences else 0,
                'count': len(hit_confidences)
            },
            'miss_confidence_stats': {
                'mean': np.mean(miss_confidences) if miss_confidences else 0,
                'min': min(miss_confidences) if miss_confidences else 0,
                'max': max(miss_confidences) if miss_confidences else 0,
                'count': len(miss_confidences)
            }
        }
    
    def calculate_business_metrics(self, matched_pairs):
        """実用性指標の計算"""
        print("\n実用性指標計算中...")
        
        cm = self.analysis_results['binary_confusion_matrix']
        
        # 基本統計
        total_patents = len(matched_pairs)
        clear_cases = cm['total_clear_cases']  # BORDERLINE除く
        borderline_count = cm['borderline_count']
        
        # 実際のHIT/MISSケース
        actual_hits = cm['tp'] + cm['fn']  # 実際のHIT数
        actual_misses = cm['tn'] + cm['fp']  # 実際のMISS数
        
        # 予測されたHIT/MISSケース  
        predicted_hits = cm['tp'] + cm['fp'] + cm['borderline_hit_pred']
        predicted_misses = cm['tn'] + cm['fn'] + cm['borderline_miss_pred']
        
        # 作業効率化計算
        manual_time_all = total_patents * 15  # 全件手動確認
        system_time = predicted_hits * 10 + predicted_misses * 1  # HIT詳細確認10分、MISS確認1分
        time_saved = manual_time_all - system_time
        efficiency_gain = time_saved / manual_time_all * 100
        
        # リスク評価
        miss_risk = cm['fn'] / actual_hits * 100 if actual_hits > 0 else 0
        false_alarm_rate = cm['fp'] / predicted_hits * 100 if predicted_hits > 0 else 0
        
        # コスト計算
        hourly_rate = 5000
        cost_saved = (time_saved / 60) * hourly_rate
        
        self.analysis_results['business_metrics'] = {
            'efficiency': {
                'manual_time_minutes': manual_time_all,
                'system_time_minutes': system_time,
                'time_saved_minutes': time_saved,
                'efficiency_gain_percent': efficiency_gain,
                'cost_saved_yen': cost_saved
            },
            'risk_assessment': {
                'hit_miss_risk_percent': miss_risk,
                'false_alarm_rate_percent': false_alarm_rate,
                'borderline_handling': 'Manual review required'
            },
            'workload_distribution': {
                'predicted_hits_review': predicted_hits,
                'predicted_miss_check': predicted_misses,
                'borderline_manual_review': borderline_count,
                'total_manual_effort_minutes': predicted_hits * 10 + borderline_count * 20
            }
        }
        
        print(f"作業効率化: {efficiency_gain:.1f}% ({time_saved:.0f}分削減)")
        print(f"コスト削減: {cost_saved:,.0f}円")
        print(f"HIT見逃しリスク: {miss_risk:.1f}%")
        print(f"False Alarm率: {false_alarm_rate:.1f}%")
    
    def _simplify_borderline_case(self, case):
        """BORDERLINE症例の簡素化"""
        return {
            'pub_number': case['pub_number'],
            'title': case['title'][:60] + '...' if len(case['title']) > 60 else case['title'],
            'pred_label': case['pred_label'],
            'confidence': round(case['confidence'], 3),
            'rationale': case['gold_data'].get('brief_rationale', '')[:120] + '...' if len(case['gold_data'].get('brief_rationale', '')) > 120 else case['gold_data'].get('brief_rationale', '')
        }
    
    def save_binary_analysis(self):
        """二値分析結果の保存"""
        output_path = Path("archive/outputs/binary_analysis_results.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'system_type': 'Binary Classification (HIT/MISS)',
                    'data_source': 'goldset (60 patents)',
                    'analysis_version': '2.0'
                },
                'results': self.analysis_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n二値分析結果保存: {output_path}")
        return output_path
    
    def run_binary_analysis(self):
        """二値分類システム用分析実行"""
        print("="*80)
        print("二値分類システム専用分析開始")
        print("="*80)
        
        self.load_data()
        matched_pairs = self.prepare_binary_analysis()
        
        self.calculate_binary_confusion_matrix(matched_pairs)
        self.analyze_borderline_cases(matched_pairs)
        self.calculate_business_metrics(matched_pairs)
        
        result_path = self.save_binary_analysis()
        
        print("="*80)
        print("二値分析完了")
        print("="*80)
        
        return result_path

def main():
    analyzer = BinaryClassificationAnalysis()
    result_path = analyzer.run_binary_analysis()
    print(f"\n二値分析完了: {result_path}")
    return result_path

if __name__ == "__main__":
    main()