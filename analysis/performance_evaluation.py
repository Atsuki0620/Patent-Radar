#!/usr/bin/env python3
"""
注目特許仕分けくん 性能評価分析スクリプト
テスト結果データとゴールドセットを比較して包括的な性能指標を算出

実行データ要件:
- testing_results.jsonl: システム予測結果（decision, confidence, hit_reason等）
- labels.jsonl: エキスパートアノテーション（gold_label）
- patents.jsonl: 特許データ
- invention_sample.json: 分類対象発明
- config.yaml: システム設定
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from datetime import datetime
import yaml


class PatentScreeningEvaluator:
    """注目特許仕分けくんの性能評価クラス"""
    
    def __init__(self, base_path: str = "."):
        """初期化"""
        self.base_path = Path(base_path)
        self.results_data = None
        self.gold_labels = None
        self.patents_data = None
        self.invention_data = None
        self.config_data = None
        
    def load_data(self):
        """すべての必要データをロード"""
        
        # テスト結果データ
        results_path = self.base_path / "archive" / "outputs" / "testing_results.jsonl"
        self.results_data = []
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.results_data.append(json.loads(line))
        
        # ゴールドラベル
        labels_path = self.base_path / "testing" / "data" / "labels.jsonl"
        self.gold_labels = {}
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.gold_labels[item['publication_number']] = item
        
        # 特許データ
        patents_path = self.base_path / "testing" / "data" / "patents.jsonl"
        self.patents_data = {}
        with open(patents_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.patents_data[item['publication_number']] = item
        
        # 発明サンプル
        invention_path = self.base_path / "testing" / "data" / "invention_sample.json"
        with open(invention_path, 'r', encoding='utf-8') as f:
            self.invention_data = json.load(f)
        
        # 設定データ
        config_path = self.base_path / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config_data = yaml.safe_load(f)
            
        print(f"データロード完了:")
        print(f"- テスト結果: {len(self.results_data)}件")
        print(f"- ゴールドラベル: {len(self.gold_labels)}件")
        print(f"- 特許データ: {len(self.patents_data)}件")

    def prepare_evaluation_data(self) -> pd.DataFrame:
        """評価用のデータフレームを準備"""
        
        evaluation_records = []
        
        for result in self.results_data:
            pub_num = result['publication_number']
            
            # ゴールドラベル取得
            gold_data = self.gold_labels.get(pub_num, {})
            gold_label = gold_data.get('gold_label', 'unknown')
            gold_rationale = gold_data.get('brief_rationale', '')
            
            # 特許データ取得
            patent_data = self.patents_data.get(pub_num, {})
            
            record = {
                'pub_number': pub_num,
                'title': result.get('title', ''),
                'assignee': result.get('assignee', ''),
                'pub_date': result.get('pub_date', ''),
                'predicted_decision': result.get('decision', 'miss'),
                'confidence': result.get('confidence', 0.0),
                'hit_reason_1': result.get('hit_reason_1', ''),
                'hit_src_1': result.get('hit_src_1', ''),
                'hit_reason_2': result.get('hit_reason_2', ''),
                'hit_src_2': result.get('hit_src_2', ''),
                'rank': result.get('rank', 999),
                'gold_label': gold_label,
                'gold_rationale': gold_rationale,
                'url_hint': result.get('url_hint', '')
            }
            
            evaluation_records.append(record)
        
        df = pd.DataFrame(evaluation_records)
        df = df.sort_values('rank').reset_index(drop=True)
        
        print(f"評価データフレーム作成完了: {len(df)}件")
        return df

    def calculate_binary_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """二値分類の性能指標を計算（hit/missのみ、borderlineは除外）"""
        
        # borderlineを除外した二値分類用データ
        binary_df = df[df['gold_label'].isin(['hit', 'miss'])].copy()
        
        # 予測とゴールドラベルをバイナリに変換
        y_true = (binary_df['gold_label'] == 'hit').astype(int)
        y_pred = (binary_df['predicted_decision'] == 'hit').astype(int)
        y_scores = binary_df['confidence'].values
        
        # 混同行列
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 基本指標
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # ROC分析
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        metrics = {
            'total_patents': len(binary_df),
            'confusion_matrix': {
                'true_positive': int(tp),
                'false_positive': int(fp),
                'true_negative': int(tn),
                'false_negative': int(fn)
            },
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'specificity': float(specificity),
            'roc_auc': float(roc_auc),
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'false_negative_rate': float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
        }
        
        return metrics

    def analyze_borderline_cases(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Borderlineケースの分析"""
        
        borderline_df = df[df['gold_label'] == 'borderline'].copy()
        
        if len(borderline_df) == 0:
            return {'total_borderline': 0, 'analysis': 'No borderline cases found'}
        
        # Borderlineケースのシステム予測分布
        hit_predicted = len(borderline_df[borderline_df['predicted_decision'] == 'hit'])
        miss_predicted = len(borderline_df[borderline_df['predicted_decision'] == 'miss'])
        
        # 信頼度分析
        hit_confidences = borderline_df[borderline_df['predicted_decision'] == 'hit']['confidence'].values
        miss_confidences = borderline_df[borderline_df['predicted_decision'] == 'miss']['confidence'].values
        
        analysis = {
            'total_borderline': len(borderline_df),
            'predicted_hit': int(hit_predicted),
            'predicted_miss': int(miss_predicted),
            'hit_percentage': float(hit_predicted / len(borderline_df) * 100),
            'avg_confidence_when_hit': float(np.mean(hit_confidences)) if len(hit_confidences) > 0 else 0.0,
            'avg_confidence_when_miss': float(np.mean(miss_confidences)) if len(miss_confidences) > 0 else 0.0,
            'borderline_cases': borderline_df[['pub_number', 'title', 'predicted_decision', 'confidence', 'gold_rationale']].to_dict('records')
        }
        
        return analysis

    def calculate_ranking_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """ランキング品質の指標を計算"""
        
        # HIT特許のランキング
        hit_patents = df[df['gold_label'] == 'hit'].copy()
        total_hits = len(hit_patents)
        
        if total_hits == 0:
            return {'total_hits': 0, 'analysis': 'No hit patents found'}
        
        # Precision@K計算
        precision_at_k = {}
        for k in [5, 10, 20, 30]:
            if k <= len(df):
                top_k = df.head(k)
                hits_in_top_k = len(top_k[top_k['gold_label'] == 'hit'])
                precision_at_k[f'precision_at_{k}'] = float(hits_in_top_k / k)
        
        # Mean Average Precision (MAP)
        average_precisions = []
        hits_found = 0
        for i, row in df.iterrows():
            if row['gold_label'] == 'hit':
                hits_found += 1
                precision_at_i = hits_found / (i + 1)
                average_precisions.append(precision_at_i)
        
        map_score = float(np.mean(average_precisions)) if average_precisions else 0.0
        
        # HIT特許の平均ランキング
        avg_hit_rank = float(hit_patents['rank'].mean())
        
        # 最高信頼度のHIT特許ランキング
        top_hit_rank = int(hit_patents['rank'].min()) if len(hit_patents) > 0 else 999
        
        ranking_metrics = {
            'total_hits': int(total_hits),
            'precision_at_k': precision_at_k,
            'mean_average_precision': map_score,
            'average_hit_rank': avg_hit_rank,
            'best_hit_rank': top_hit_rank,
            'hits_in_top_10': int(len(df.head(10)[df.head(10)['gold_label'] == 'hit'])),
            'hits_in_top_20': int(len(df.head(20)[df.head(20)['gold_label'] == 'hit']))
        }
        
        return ranking_metrics

    def analyze_confidence_calibration(self, df: pd.DataFrame) -> Dict[str, Any]:
        """信頼度キャリブレーションの分析"""
        
        # 二値分類データのみ
        binary_df = df[df['gold_label'].isin(['hit', 'miss'])].copy()
        
        # 信頼度区間別の分析
        confidence_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        bin_analysis = []
        
        for i in range(len(confidence_bins) - 1):
            lower = confidence_bins[i]
            upper = confidence_bins[i + 1]
            
            bin_data = binary_df[(binary_df['confidence'] >= lower) & (binary_df['confidence'] < upper)]
            if i == len(confidence_bins) - 2:  # 最後の区間は上限を含む
                bin_data = binary_df[(binary_df['confidence'] >= lower) & (binary_df['confidence'] <= upper)]
            
            if len(bin_data) > 0:
                actual_hit_rate = len(bin_data[bin_data['gold_label'] == 'hit']) / len(bin_data)
                predicted_hit_rate = len(bin_data[bin_data['predicted_decision'] == 'hit']) / len(bin_data)
                avg_confidence = bin_data['confidence'].mean()
                
                bin_analysis.append({
                    'confidence_range': f'{lower:.1f}-{upper:.1f}',
                    'count': int(len(bin_data)),
                    'actual_hit_rate': float(actual_hit_rate),
                    'predicted_hit_rate': float(predicted_hit_rate),
                    'avg_confidence': float(avg_confidence)
                })
        
        # ラベル別の信頼度分布
        hit_confidences = binary_df[binary_df['gold_label'] == 'hit']['confidence']
        miss_confidences = binary_df[binary_df['gold_label'] == 'miss']['confidence']
        
        calibration_metrics = {
            'confidence_bins': bin_analysis,
            'hit_confidence_stats': {
                'mean': float(hit_confidences.mean()),
                'std': float(hit_confidences.std()),
                'min': float(hit_confidences.min()),
                'max': float(hit_confidences.max())
            },
            'miss_confidence_stats': {
                'mean': float(miss_confidences.mean()),
                'std': float(miss_confidences.std()),
                'min': float(miss_confidences.min()),
                'max': float(miss_confidences.max())
            }
        }
        
        return calibration_metrics

    def analyze_error_cases(self, df: pd.DataFrame) -> Dict[str, Any]:
        """エラーケース（偽陽性・偽陰性）の詳細分析"""
        
        # 二値分類データのみ
        binary_df = df[df['gold_label'].isin(['hit', 'miss'])].copy()
        
        # 偽陽性: システムがHITと予測したが実際はMISS
        false_positives = binary_df[
            (binary_df['gold_label'] == 'miss') & 
            (binary_df['predicted_decision'] == 'hit')
        ].copy()
        
        # 偽陰性: システムがMISSと予測したが実際はHIT
        false_negatives = binary_df[
            (binary_df['gold_label'] == 'hit') & 
            (binary_df['predicted_decision'] == 'miss')
        ].copy()
        
        # 偽陽性の詳細分析
        fp_analysis = []
        for _, row in false_positives.iterrows():
            fp_analysis.append({
                'pub_number': row['pub_number'],
                'title': row['title'],
                'confidence': float(row['confidence']),
                'rank': int(row['rank']),
                'hit_reason_1': row['hit_reason_1'],
                'gold_rationale': row['gold_rationale']
            })
        
        # 偽陰性の詳細分析
        fn_analysis = []
        for _, row in false_negatives.iterrows():
            fn_analysis.append({
                'pub_number': row['pub_number'],
                'title': row['title'],
                'confidence': float(row['confidence']),
                'rank': int(row['rank']),
                'gold_rationale': row['gold_rationale']
            })
        
        error_analysis = {
            'false_positives': {
                'count': len(false_positives),
                'avg_confidence': float(false_positives['confidence'].mean()) if len(false_positives) > 0 else 0.0,
                'avg_rank': float(false_positives['rank'].mean()) if len(false_positives) > 0 else 0.0,
                'cases': fp_analysis
            },
            'false_negatives': {
                'count': len(false_negatives),
                'avg_confidence': float(false_negatives['confidence'].mean()) if len(false_negatives) > 0 else 0.0,
                'avg_rank': float(false_negatives['rank'].mean()) if len(false_negatives) > 0 else 0.0,
                'cases': fn_analysis
            }
        }
        
        return error_analysis

    def calculate_operational_metrics(self) -> Dict[str, Any]:
        """運用指標の計算"""
        
        # システム設定から情報を抽出
        config = self.config_data
        model_name = config.get('llm', {}).get('model', 'unknown')
        temperature = config.get('llm', {}).get('temperature', 0.0)
        max_tokens = config.get('llm', {}).get('max_tokens', 320)
        
        # 処理済み特許数
        total_processed = len(self.results_data)
        
        # 信頼度分布
        confidences = [r.get('confidence', 0.0) for r in self.results_data]
        
        operational_metrics = {
            'system_config': {
                'model': model_name,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'ranking_method': config.get('ranking', {}).get('method', 'llm_only')
            },
            'processing_stats': {
                'total_patents_processed': total_processed,
                'hit_predictions': len([r for r in self.results_data if r.get('decision') == 'hit']),
                'miss_predictions': len([r for r in self.results_data if r.get('decision') == 'miss']),
                'avg_confidence': float(np.mean(confidences)),
                'confidence_std': float(np.std(confidences))
            },
            'target_invention': {
                'title': self.invention_data.get('title', ''),
                'problem': self.invention_data.get('problem', ''),
                'solution': self.invention_data.get('solution', ''),
                'key_elements': self.invention_data.get('key_elements', [])
            }
        }
        
        return operational_metrics

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """包括的な性能評価レポートを生成"""
        
        print("包括的性能評価レポートを生成中...")
        
        # データ準備
        df = self.prepare_evaluation_data()
        
        # 各種分析実行
        binary_metrics = self.calculate_binary_metrics(df)
        borderline_analysis = self.analyze_borderline_cases(df)
        ranking_metrics = self.calculate_ranking_metrics(df)
        confidence_analysis = self.analyze_confidence_calibration(df)
        error_analysis = self.analyze_error_cases(df)
        operational_metrics = self.calculate_operational_metrics()
        
        # 成功基準との比較
        success_criteria = {
            'target_accuracy': 0.8,  # 80%以上
            'target_recall': 0.8,    # HIT検出の高い再現率
            'acceptable_precision': 0.7  # 精度70%以上
        }
        
        criteria_met = {
            'accuracy_met': binary_metrics['accuracy'] >= success_criteria['target_accuracy'],
            'recall_met': binary_metrics['recall'] >= success_criteria['target_recall'],
            'precision_met': binary_metrics['precision'] >= success_criteria['acceptable_precision']
        }
        
        # 統合レポート
        comprehensive_report = {
            'evaluation_metadata': {
                'report_generated': datetime.now().isoformat(),
                'system_name': '注目特許仕分けくん',
                'evaluation_dataset': 'testing dataset (59 patents)',
                'gold_standard': 'Expert annotations with hit/miss/borderline labels'
            },
            'success_criteria': success_criteria,
            'criteria_assessment': criteria_met,
            'binary_classification_performance': binary_metrics,
            'borderline_case_analysis': borderline_analysis,
            'ranking_quality': ranking_metrics,
            'confidence_calibration': confidence_analysis,
            'error_analysis': error_analysis,
            'operational_metrics': operational_metrics,
            'recommendations': self.generate_recommendations(binary_metrics, error_analysis, ranking_metrics)
        }
        
        return comprehensive_report

    def generate_recommendations(self, binary_metrics: Dict, error_analysis: Dict, ranking_metrics: Dict) -> List[str]:
        """分析結果に基づく改善提案を生成"""
        
        recommendations = []
        
        # 精度に基づく提案
        if binary_metrics['accuracy'] < 0.8:
            recommendations.append("総合精度が80%を下回っています。プロンプトエンジニアリングの改善を推奨します。")
        
        # 再現率に基づく提案
        if binary_metrics['recall'] < 0.8:
            recommendations.append("HIT検出の再現率が低下しています。見逃しリスク軽減のため、判定閾値の調整を検討してください。")
        
        # 偽陽性に基づく提案
        if error_analysis['false_positives']['count'] > len(self.results_data) * 0.2:
            recommendations.append("偽陽性率が高い状態です。MISS判定の精度向上のため、除外条件の強化を検討してください。")
        
        # ランキングに基づく提案
        if ranking_metrics['precision_at_k'].get('precision_at_10', 0) < 0.5:
            recommendations.append("上位10件におけるHIT精度が低下しています。ランキング手法の改善を推奨します。")
        
        # 成功基準をクリアしている場合
        if (binary_metrics['accuracy'] >= 0.8 and 
            binary_metrics['recall'] >= 0.8 and 
            binary_metrics['precision'] >= 0.7):
            recommendations.append("システムは実用水準に到達しています。本格運用の開始を推奨します。")
        
        return recommendations

    def save_report(self, report: Dict[str, Any], output_path: str = None):
        """レポートをJSONファイルに保存"""
        
        if output_path is None:
            output_path = self.base_path / "analysis" / "comprehensive_evaluation_report.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"包括的性能評価レポートを保存しました: {output_path}")
        return str(output_path)


def main():
    """メイン実行関数"""
    
    print("=== 注目特許仕分けくん 性能評価分析 ===")
    
    # 評価実行
    evaluator = PatentScreeningEvaluator()
    evaluator.load_data()
    
    # 包括的レポート生成
    report = evaluator.generate_comprehensive_report()
    
    # 結果保存
    report_path = evaluator.save_report(report)
    
    # サマリ表示
    print("\n=== 評価結果サマリ ===")
    binary_metrics = report['binary_classification_performance']
    print(f"総合精度: {binary_metrics['accuracy']:.3f}")
    print(f"HIT検出精度: {binary_metrics['precision']:.3f}")
    print(f"HIT検出再現率: {binary_metrics['recall']:.3f}")
    print(f"F1スコア: {binary_metrics['f1_score']:.3f}")
    
    ranking_metrics = report['ranking_quality']
    print(f"上位10件のHIT数: {ranking_metrics['hits_in_top_10']}")
    print(f"MAP スコア: {ranking_metrics['mean_average_precision']:.3f}")
    
    borderline_analysis = report['borderline_case_analysis']
    if borderline_analysis['total_borderline'] > 0:
        print(f"Borderlineケース: {borderline_analysis['total_borderline']}件 (HIT予測: {borderline_analysis['hit_percentage']:.1f}%)")
    
    print(f"\n詳細レポート: {report_path}")
    
    return report


if __name__ == "__main__":
    main()