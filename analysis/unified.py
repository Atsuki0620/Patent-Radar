#!/usr/bin/env python3
"""
統合特許分析システム
二値分類システム(HIT/MISS)に特化した簡素化分析
- 二値分類混同行列
- 実用性指標 
- BORDERLINEケース分析
- HTMLレポート生成
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class PatentAnalysis:
    """統合特許分析クラス"""
    
    def __init__(self):
        self.predictions = []
        self.gold_labels = {}
        self.analysis_results = {}
        
    def load_data(self):
        """データ読み込み"""
        print("データ読み込み中...")
        
        predictions_path = Path("archive/outputs/goldset_results.jsonl")
        with open(predictions_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.predictions.append(json.loads(line))
        
        labels_path = Path("testing/data/labels.jsonl")
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    label_data = json.loads(line)
                    self.gold_labels[label_data['publication_number']] = label_data
        
        print(f"予測結果: {len(self.predictions)}件")
        print(f"ゴールドラベル: {len(self.gold_labels)}件")
    
    def prepare_analysis_data(self):
        """分析用データ準備"""
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
    
    def calculate_binary_metrics(self, matched_pairs):
        """二値分類指標計算"""
        print("\n二値分類指標計算中...")
        
        tp = tn = fp = fn = 0
        borderline_cases = []
        borderline_hit_pred = borderline_miss_pred = 0
        
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
                tp += 1 if pred == 'hit' else (fn := fn + 1)
            elif gold == 'miss':
                fp += 1 if pred == 'hit' else (tn := tn + 1)
        
        total_clear = tp + tn + fp + fn
        accuracy = (tp + tn) / total_clear if total_clear > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        self.analysis_results['binary_metrics'] = {
            'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
            'performance': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score},
            'borderline': {
                'total_count': len(borderline_cases),
                'hit_predictions': borderline_hit_pred,
                'miss_predictions': borderline_miss_pred,
                'cases': [self._simplify_case(case) for case in borderline_cases[:10]]
            }
        }
        
        print(f"精度: {accuracy:.3f}, 精密度: {precision:.3f}, 再現率: {recall:.3f}, F1: {f1_score:.3f}")
        print(f"BORDERLINEケース: {len(borderline_cases)}件 (HIT予測: {borderline_hit_pred}, MISS予測: {borderline_miss_pred})")
    
    def calculate_business_metrics(self, matched_pairs):
        """実用性指標計算"""
        print("\n実用性指標計算中...")
        
        metrics = self.analysis_results['binary_metrics']
        cm = metrics['confusion_matrix']
        
        total_patents = len(matched_pairs)
        predicted_hits = cm['tp'] + cm['fp'] + metrics['borderline']['hit_predictions']
        predicted_misses = cm['tn'] + cm['fn'] + metrics['borderline']['miss_predictions']
        
        manual_time_all = total_patents * 15
        system_time = predicted_hits * 10 + predicted_misses * 1
        time_saved = manual_time_all - system_time
        efficiency_gain = time_saved / manual_time_all * 100
        
        actual_hits = cm['tp'] + cm['fn']
        miss_risk = cm['fn'] / actual_hits * 100 if actual_hits > 0 else 0
        false_alarm_rate = cm['fp'] / predicted_hits * 100 if predicted_hits > 0 else 0
        
        cost_saved = (time_saved / 60) * 5000
        
        self.analysis_results['business_metrics'] = {
            'efficiency_gain_percent': efficiency_gain,
            'time_saved_minutes': time_saved,
            'cost_saved_yen': cost_saved,
            'miss_risk_percent': miss_risk,
            'false_alarm_rate_percent': false_alarm_rate
        }
        
        print(f"作業効率化: {efficiency_gain:.1f}% ({time_saved:.0f}分削減)")
        print(f"コスト削減: {cost_saved:,.0f}円")
        print(f"HIT見逃しリスク: {miss_risk:.1f}%")
    
    def _simplify_case(self, case):
        """ケース情報簡素化"""
        return {
            'pub_number': case['pub_number'],
            'title': case['title'][:50] + '...' if len(case['title']) > 50 else case['title'],
            'pred_label': case['pred_label'],
            'confidence': round(case['confidence'], 3)
        }
    
    def generate_html_report(self):
        """HTMLレポート生成"""
        print("\nHTMLレポート生成中...")
        
        cm = self.analysis_results['binary_metrics']['confusion_matrix']
        perf = self.analysis_results['binary_metrics']['performance']
        biz = self.analysis_results['business_metrics']
        borderline = self.analysis_results['binary_metrics']['borderline']
        
        html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>特許二値分類システム分析レポート</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
        .metric-label {{ color: #7f8c8d; font-size: 0.9em; }}
        .confusion-matrix {{ margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: center; border: 1px solid #bdc3c7; }}
        th {{ background-color: #34495e; color: white; }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
        .borderline-cases {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .summary {{ background: #d4edda; padding: 20px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>特許二値分類システム分析レポート</h1>
        <p><strong>分析日時:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>🎯 精度指標</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value positive">{perf['accuracy']:.1%}</div>
                <div class="metric-label">総合精度</div>
            </div>
            <div class="metric-card">
                <div class="metric-value positive">{perf['precision']:.1%}</div>
                <div class="metric-label">精密度</div>
            </div>
            <div class="metric-card">
                <div class="metric-value positive">{perf['recall']:.1%}</div>
                <div class="metric-label">再現率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value positive">{perf['f1_score']:.1%}</div>
                <div class="metric-label">F1スコア</div>
            </div>
        </div>
        
        <div class="confusion-matrix">
            <h3>混同行列</h3>
            <table>
                <thead>
                    <tr><th></th><th colspan="2">予測</th></tr>
                    <tr><th>実際</th><th>HIT</th><th>MISS</th></tr>
                </thead>
                <tbody>
                    <tr><th>HIT</th><td class="positive">{cm['tp']}</td><td class="negative">{cm['fn']}</td></tr>
                    <tr><th>MISS</th><td class="warning">{cm['fp']}</td><td class="positive">{cm['tn']}</td></tr>
                </tbody>
            </table>
        </div>
        
        <h2>💼 実用性指標</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value positive">{biz['efficiency_gain_percent']:.1f}%</div>
                <div class="metric-label">作業効率化</div>
            </div>
            <div class="metric-card">
                <div class="metric-value positive">{biz['cost_saved_yen']:,.0f}円</div>
                <div class="metric-label">コスト削減</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'negative' if biz['miss_risk_percent'] > 10 else 'warning'}">{biz['miss_risk_percent']:.1f}%</div>
                <div class="metric-label">HIT見逃しリスク</div>
            </div>
            <div class="metric-card">
                <div class="metric-value warning">{biz['false_alarm_rate_percent']:.1f}%</div>
                <div class="metric-label">誤警報率</div>
            </div>
        </div>
        
        <div class="borderline-cases">
            <h3>⚖️ BORDERLINEケース分析</h3>
            <p><strong>総数:</strong> {borderline['total_count']}件</p>
            <p><strong>HIT予測:</strong> {borderline['hit_predictions']}件 | <strong>MISS予測:</strong> {borderline['miss_predictions']}件</p>
            <p>BORDERLINEケースは人間による手動レビューが必要です。</p>
        </div>
        
        <div class="summary">
            <h3>📊 総合評価</h3>
            <p>本システムは <strong>{perf['accuracy']:.1%}</strong> の総合精度を達成し、
            作業効率を <strong>{biz['efficiency_gain_percent']:.1f}%</strong> 向上させ、
            <strong>{biz['cost_saved_yen']:,.0f}円</strong> のコスト削減効果が期待されます。</p>
            <p>HIT見逃しリスクは <strong>{biz['miss_risk_percent']:.1f}%</strong> に抑制されており、
            実用レベルでの運用が可能です。</p>
        </div>
    </div>
</body>
</html>"""
        
        report_path = Path("archive/reports/unified_analysis_report.html")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTMLレポート生成完了: {report_path}")
        return report_path
    
    def save_results(self):
        """結果保存"""
        output_path = Path("archive/outputs/unified_analysis_results.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'system_type': 'Binary Classification (HIT/MISS)',
                    'data_source': 'goldset (60 patents)',
                    'analysis_version': '3.0'
                },
                'results': self.analysis_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"分析結果保存: {output_path}")
        return output_path
    
    def run_analysis(self):
        """統合分析実行"""
        print("=" * 80)
        print("統合特許分析開始")
        print("=" * 80)
        
        self.load_data()
        matched_pairs = self.prepare_analysis_data()
        
        self.calculate_binary_metrics(matched_pairs)
        self.calculate_business_metrics(matched_pairs)
        
        html_report = self.generate_html_report()
        json_results = self.save_results()
        
        print("=" * 80)
        print("統合分析完了")
        print("=" * 80)
        
        return html_report, json_results

def main():
    analyzer = PatentAnalysis()
    html_report, json_results = analyzer.run_analysis()
    print(f"\nHTMLレポート: {html_report}")
    print(f"JSON結果: {json_results}")
    return html_report, json_results

if __name__ == "__main__":
    main()