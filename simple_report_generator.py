#!/usr/bin/env python3
"""
Simplified report generator for testing evaluation
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def calculate_binary_metrics(tp, fp, tn, fn):
    """Calculate basic binary classification metrics"""
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    }

def load_and_analyze():
    """Load data and perform analysis"""
    
    # Load results
    results_path = Path("archive/outputs/testing_results.jsonl")
    labels_path = Path("testing/data/labels.jsonl")
    
    predictions = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line.strip()))
    
    labels = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            label_data = json.loads(line.strip())
            labels[label_data['publication_number']] = label_data
    
    # Analyze results
    total_patents = len(predictions)
    hit_predictions = [p for p in predictions if p['decision'] == 'hit']
    miss_predictions = [p for p in predictions if p['decision'] == 'miss']
    
    # Calculate confusion matrix (excluding borderline from binary evaluation)
    tp = fp = tn = fn = 0
    borderline_count = 0
    
    gold_distribution = {'hit': 0, 'miss': 0, 'borderline': 0}
    pred_distribution = {'hit': 0, 'miss': 0, 'borderline': 0}
    
    for pred in predictions:
        pub_num = pred['publication_number']
        gold_label = labels.get(pub_num, {}).get('gold_label', 'miss')
        pred_decision = pred['decision']
        
        gold_distribution[gold_label] += 1
        pred_distribution[pred_decision] += 1
        
        if gold_label == 'borderline':
            borderline_count += 1
            continue
            
        # Binary classification metrics (hit vs miss only)
        if gold_label == 'hit' and pred_decision == 'hit':
            tp += 1
        elif gold_label == 'miss' and pred_decision == 'hit':
            fp += 1
        elif gold_label == 'miss' and pred_decision == 'miss':
            tn += 1
        elif gold_label == 'hit' and pred_decision == 'miss':
            fn += 1
    
    binary_total = tp + fp + tn + fn
    metrics = calculate_binary_metrics(tp, fp, tn, fn)
    
    # Confidence analysis
    hit_confidences = [p['confidence'] for p in predictions if p['decision'] == 'hit']
    miss_confidences = [p['confidence'] for p in predictions if p['decision'] == 'miss']
    
    avg_hit_conf = sum(hit_confidences) / len(hit_confidences) if hit_confidences else 0
    avg_miss_conf = sum(miss_confidences) / len(miss_confidences) if miss_confidences else 0
    
    # Ranking analysis - how well do hits rank?
    predictions_sorted = sorted(predictions, key=lambda x: (-x['confidence'], x['pub_number']))
    
    hit_ranks = []
    for i, pred in enumerate(predictions_sorted):
        pub_num = pred['publication_number']
        gold_label = labels.get(pub_num, {}).get('gold_label', 'miss')
        if gold_label == 'hit':
            hit_ranks.append(i + 1)
    
    avg_hit_rank = sum(hit_ranks) / len(hit_ranks) if hit_ranks else 0
    
    return {
        'total_patents': total_patents,
        'gold_distribution': gold_distribution,
        'pred_distribution': pred_distribution,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        'binary_total': binary_total,
        'borderline_count': borderline_count,
        'metrics': metrics,
        'confidence_analysis': {
            'avg_hit_confidence': avg_hit_conf,
            'avg_miss_confidence': avg_miss_conf,
            'hit_count': len(hit_confidences),
            'miss_count': len(miss_confidences)
        },
        'ranking_analysis': {
            'total_hits': len(hit_ranks),
            'avg_hit_rank': avg_hit_rank,
            'hit_ranks': hit_ranks[:10]  # Top 10 for display
        }
    }

def generate_html_report(analysis_data):
    """Generate HTML report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>注目特許仕分けくん - 性能評価レポート</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        .status-good {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-poor {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .summary-box {{ background: #e8f4f8; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #3498db; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 注目特許仕分けくん - 性能評価レポート</h1>
        
        <div class="summary-box">
            <h3>📊 実行概要</h3>
            <p><strong>評価日時:</strong> {timestamp}</p>
            <p><strong>テストデータ:</strong> {analysis_data['total_patents']}件の特許</p>
            <p><strong>システム構成:</strong> GPT-4o-mini / LLM_confidenceベースランキング</p>
            <p><strong>対象発明:</strong> 液体分離設備の効率改善システム</p>
        </div>

        <h2>🎯 総合性能指標</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value {'status-good' if analysis_data['metrics']['accuracy'] >= 0.8 else 'status-warning' if analysis_data['metrics']['accuracy'] >= 0.7 else 'status-poor'}">
                    {analysis_data['metrics']['accuracy']:.1%}
                </div>
                <div class="metric-label">総合精度</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'status-good' if analysis_data['metrics']['precision'] >= 0.8 else 'status-warning' if analysis_data['metrics']['precision'] >= 0.7 else 'status-poor'}">
                    {analysis_data['metrics']['precision']:.1%}
                </div>
                <div class="metric-label">HIT適合率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'status-good' if analysis_data['metrics']['recall'] >= 0.8 else 'status-warning' if analysis_data['metrics']['recall'] >= 0.7 else 'status-poor'}">
                    {analysis_data['metrics']['recall']:.1%}
                </div>
                <div class="metric-label">HIT再現率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'status-good' if analysis_data['metrics']['f1'] >= 0.8 else 'status-warning' if analysis_data['metrics']['f1'] >= 0.7 else 'status-poor'}">
                    {analysis_data['metrics']['f1']:.1%}
                </div>
                <div class="metric-label">F1スコア</div>
            </div>
        </div>

        <h2>📈 分類結果詳細</h2>
        <table>
            <tr><th>指標</th><th>値</th><th>評価</th></tr>
            <tr><td>総特許数</td><td>{analysis_data['total_patents']}</td><td>-</td></tr>
            <tr><td>二値分類対象</td><td>{analysis_data['binary_total']}</td><td>-</td></tr>
            <tr><td>Borderlineケース</td><td>{analysis_data['borderline_count']}</td><td>別途分析</td></tr>
            <tr><td>True Positive (正解HIT)</td><td>{analysis_data['confusion_matrix']['tp']}</td><td>-</td></tr>
            <tr><td>False Positive (誤判定HIT)</td><td>{analysis_data['confusion_matrix']['fp']}</td><td>-</td></tr>
            <tr><td>True Negative (正解MISS)</td><td>{analysis_data['confusion_matrix']['tn']}</td><td>-</td></tr>
            <tr><td>False Negative (見逃しHIT)</td><td>{analysis_data['confusion_matrix']['fn']}</td><td>-</td></tr>
        </table>

        <h2>🎲 信頼度分析</h2>
        <table>
            <tr><th>分類</th><th>平均信頼度</th><th>件数</th></tr>
            <tr><td>HIT予測</td><td>{analysis_data['confidence_analysis']['avg_hit_confidence']:.3f}</td><td>{analysis_data['confidence_analysis']['hit_count']}</td></tr>
            <tr><td>MISS予測</td><td>{analysis_data['confidence_analysis']['avg_miss_confidence']:.3f}</td><td>{analysis_data['confidence_analysis']['miss_count']}</td></tr>
        </table>

        <h2>📊 ランキング品質</h2>
        <div class="summary-box">
            <p><strong>HIT特許の平均ランク:</strong> {analysis_data['ranking_analysis']['avg_hit_rank']:.1f}位 (総{analysis_data['total_patents']}件中)</p>
            <p><strong>HIT特許総数:</strong> {analysis_data['ranking_analysis']['total_hits']}件</p>
            <p><strong>上位HIT特許ランク:</strong> {', '.join(map(str, analysis_data['ranking_analysis']['hit_ranks']))}</p>
        </div>

        <h2>✅ 成功基準評価</h2>
        <table>
            <tr><th>基準</th><th>目標</th><th>実績</th><th>評価</th></tr>
            <tr><td>総合精度</td><td>≥80%</td><td>{analysis_data['metrics']['accuracy']:.1%}</td><td>{'✅ 達成' if analysis_data['metrics']['accuracy'] >= 0.8 else '❌ 未達成'}</td></tr>
            <tr><td>HIT検出精度</td><td>高再現率</td><td>{analysis_data['metrics']['recall']:.1%}</td><td>{'✅ 良好' if analysis_data['metrics']['recall'] >= 0.8 else '⚠️ 要改善' if analysis_data['metrics']['recall'] >= 0.7 else '❌ 不十分'}</td></tr>
            <tr><td>ランキング品質</td><td>HITが上位</td><td>平均{analysis_data['ranking_analysis']['avg_hit_rank']:.1f}位</td><td>{'✅ 良好' if analysis_data['ranking_analysis']['avg_hit_rank'] <= analysis_data['total_patents'] * 0.3 else '⚠️ 要改善'}</td></tr>
        </table>

        <h2>🚀 改善提案</h2>
        <div class="summary-box">
            <h4>優先度：高</h4>
            <ul>
                {'<li>HIT再現率の向上 - False Negativeを' + str(analysis_data['confusion_matrix']['fn']) + '件から削減</li>' if analysis_data['metrics']['recall'] < 0.8 else ''}
                {'<li>精度向上 - False Positiveを' + str(analysis_data['confusion_matrix']['fp']) + '件から削減</li>' if analysis_data['metrics']['precision'] < 0.8 else ''}
                {'<li>Borderlineケース' + str(analysis_data['borderline_count']) + '件の処理方針策定</li>' if analysis_data['borderline_count'] > 0 else ''}
            </ul>
            
            <h4>優先度：中</h4>
            <ul>
                <li>信頼度スコアのキャリブレーション調整</li>
                <li>ランキングアルゴリズムの最適化</li>
                <li>処理効率の向上</li>
            </ul>
        </div>

        <footer style="margin-top: 40px; text-align: center; color: #7f8c8d; border-top: 1px solid #ddd; padding-top: 20px;">
            注目特許仕分けくん - 性能評価レポート | Generated on {timestamp}
        </footer>
    </div>
</body>
</html>
"""
    
    return html_content

def generate_markdown_report(analysis_data):
    """Generate Markdown report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"""# 注目特許仕分けくん - 性能評価レポート

## 📊 実行概要

- **評価日時:** {timestamp}
- **テストデータ:** {analysis_data['total_patents']}件の特許
- **システム構成:** GPT-4o-mini / LLM_confidenceベースランキング
- **対象発明:** 液体分離設備の効率改善システム

## 🎯 総合性能指標

| 指標 | 値 | 評価 |
|------|-----|------|
| 総合精度 | {analysis_data['metrics']['accuracy']:.1%} | {'✅ 良好' if analysis_data['metrics']['accuracy'] >= 0.8 else '⚠️ 要改善' if analysis_data['metrics']['accuracy'] >= 0.7 else '❌ 不十分'} |
| HIT適合率 | {analysis_data['metrics']['precision']:.1%} | {'✅ 良好' if analysis_data['metrics']['precision'] >= 0.8 else '⚠️ 要改善' if analysis_data['metrics']['precision'] >= 0.7 else '❌ 不十分'} |
| HIT再現率 | {analysis_data['metrics']['recall']:.1%} | {'✅ 良好' if analysis_data['metrics']['recall'] >= 0.8 else '⚠️ 要改善' if analysis_data['metrics']['recall'] >= 0.7 else '❌ 不十分'} |
| F1スコア | {analysis_data['metrics']['f1']:.1%} | {'✅ 良好' if analysis_data['metrics']['f1'] >= 0.8 else '⚠️ 要改善' if analysis_data['metrics']['f1'] >= 0.7 else '❌ 不十分'} |

## 📈 混同行列

```
                実際
予測      HIT    MISS
HIT       {analysis_data['confusion_matrix']['tp']}     {analysis_data['confusion_matrix']['fp']}
MISS      {analysis_data['confusion_matrix']['fn']}     {analysis_data['confusion_matrix']['tn']}
```

- **True Positive (正解HIT):** {analysis_data['confusion_matrix']['tp']}件
- **False Positive (誤判定HIT):** {analysis_data['confusion_matrix']['fp']}件  
- **True Negative (正解MISS):** {analysis_data['confusion_matrix']['tn']}件
- **False Negative (見逃しHIT):** {analysis_data['confusion_matrix']['fn']}件

## 🎲 信頼度分析

- **HIT予測の平均信頼度:** {analysis_data['confidence_analysis']['avg_hit_confidence']:.3f} ({analysis_data['confidence_analysis']['hit_count']}件)
- **MISS予測の平均信頼度:** {analysis_data['confidence_analysis']['avg_miss_confidence']:.3f} ({analysis_data['confidence_analysis']['miss_count']}件)

## 📊 ランキング品質

- **HIT特許の平均ランク:** {analysis_data['ranking_analysis']['avg_hit_rank']:.1f}位 (総{analysis_data['total_patents']}件中)
- **HIT特許総数:** {analysis_data['ranking_analysis']['total_hits']}件
- **上位HIT特許ランク:** {', '.join(map(str, analysis_data['ranking_analysis']['hit_ranks']))}

## ✅ 成功基準評価

| 基準 | 目標 | 実績 | 評価 |
|------|------|------|------|
| 総合精度 | ≥80% | {analysis_data['metrics']['accuracy']:.1%} | {'✅ 達成' if analysis_data['metrics']['accuracy'] >= 0.8 else '❌ 未達成'} |
| HIT検出精度 | 高再現率 | {analysis_data['metrics']['recall']:.1%} | {'✅ 良好' if analysis_data['metrics']['recall'] >= 0.8 else '⚠️ 要改善' if analysis_data['metrics']['recall'] >= 0.7 else '❌ 不十分'} |
| ランキング品質 | HITが上位 | 平均{analysis_data['ranking_analysis']['avg_hit_rank']:.1f}位 | {'✅ 良好' if analysis_data['ranking_analysis']['avg_hit_rank'] <= analysis_data['total_patents'] * 0.3 else '⚠️ 要改善'} |

## 🚀 改善提案

### 優先度：高

{'- HIT再現率の向上 - False Negativeを' + str(analysis_data['confusion_matrix']['fn']) + '件から削減' if analysis_data['metrics']['recall'] < 0.8 else ''}
{'- 精度向上 - False Positiveを' + str(analysis_data['confusion_matrix']['fp']) + '件から削減' if analysis_data['metrics']['precision'] < 0.8 else ''}
{'- Borderlineケース' + str(analysis_data['borderline_count']) + '件の処理方針策定' if analysis_data['borderline_count'] > 0 else ''}

### 優先度：中

- 信頼度スコアのキャリブレーション調整
- ランキングアルゴリズムの最適化  
- 処理効率の向上

## 📋 技術的詳細

- **二値分類対象:** {analysis_data['binary_total']}件 (Borderline {analysis_data['borderline_count']}件は除外)
- **ゴールド標準分布:** HIT {analysis_data['gold_distribution']['hit']}件, MISS {analysis_data['gold_distribution']['miss']}件, Borderline {analysis_data['gold_distribution']['borderline']}件
- **予測分布:** HIT {analysis_data['pred_distribution']['hit']}件, MISS {analysis_data['pred_distribution']['miss']}件

---
*Generated on {timestamp} - 注目特許仕分けくん Performance Evaluation*
"""
    
    return md_content

def main():
    """Main execution function"""
    
    print("=== 注目特許仕分けくん 性能評価レポート生成 ===")
    
    # Analyze data
    print("1. データ分析実行中...")
    analysis_data = load_and_analyze()
    
    # Generate reports
    print("2. HTMLレポート生成中...")
    html_content = generate_html_report(analysis_data)
    
    print("3. Markdownレポート生成中...")
    md_content = generate_markdown_report(analysis_data)
    
    # Save reports
    reports_dir = Path("archive/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    
    html_path = reports_dir / f"patent_screening_evaluation_report_{timestamp_str}.html"
    md_path = reports_dir / f"patent_screening_evaluation_report_{timestamp_str}.md"
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"4. レポート生成完了!")
    print(f"   HTML: {html_path}")
    print(f"   Markdown: {md_path}")
    
    # Summary
    print(f"\n=== 評価サマリー ===")
    print(f"総合精度: {analysis_data['metrics']['accuracy']:.1%}")
    print(f"HIT適合率: {analysis_data['metrics']['precision']:.1%}")  
    print(f"HIT再現率: {analysis_data['metrics']['recall']:.1%}")
    print(f"F1スコア: {analysis_data['metrics']['f1']:.1%}")
    
    success_criteria_met = (
        analysis_data['metrics']['accuracy'] >= 0.8 and
        analysis_data['metrics']['recall'] >= 0.8 and
        analysis_data['ranking_analysis']['avg_hit_rank'] <= analysis_data['total_patents'] * 0.3
    )
    
    print(f"\n成功基準達成: {'✅ YES' if success_criteria_met else '❌ NO'}")
    
    return {
        'html_path': str(html_path),
        'md_path': str(md_path),
        'metrics': analysis_data['metrics'],
        'success_criteria_met': success_criteria_met
    }

if __name__ == "__main__":
    result = main()