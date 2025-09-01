#!/usr/bin/env python3
"""
Patent-Radar 厳密評価エンジン
1000分割閾値探索、ROC分析、混同行列分析による包括的性能評価
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score
)
import json


class StrictBinaryEvaluator:
    """厳密二値分類評価クラス"""
    
    def __init__(self, random_seed: int = 42, threshold_resolution: int = 1000):
        """
        初期化
        
        Args:
            random_seed: 再現性確保用乱数シード
            threshold_resolution: 閾値分割数（デフォルト: 1000）
        """
        self.random_seed = random_seed
        self.threshold_resolution = threshold_resolution
        np.random.seed(random_seed)
        
        # 評価結果格納用
        self.evaluation_results = {}
        
        # データ保持用
        self.y_true = None
        self.y_proba = None
        self.dataset = None
        
        # ロギング設定
        self._setup_logging()
        
    def _setup_logging(self):
        """ロギング設定"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def load_evaluation_data(self, data_path: str) -> bool:
        """
        評価用データの読み込み
        
        Args:
            data_path: CSVデータファイルパス
            
        Returns:
            読み込み成功フラグ
        """
        try:
            self.logger.info(f"Loading evaluation data from: {data_path}")
            
            self.dataset = pd.read_csv(data_path, encoding='utf-8')
            
            # 必須カラムの確認
            required_columns = ['y_true', 'y_proba']
            missing_columns = [col for col in required_columns if col not in self.dataset.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            self.y_true = self.dataset['y_true'].values
            self.y_proba = self.dataset['y_proba'].values
            
            # データ品質チェック
            self._validate_input_data()
            
            self.logger.info(f"Loaded {len(self.dataset)} samples for evaluation")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load evaluation data: {e}")
            return False
            
    def _validate_input_data(self):
        """入力データの品質チェック"""
        # 形状チェック
        if len(self.y_true) != len(self.y_proba):
            raise ValueError("Length mismatch between y_true and y_proba")
            
        # y_true の値チェック
        unique_true = set(self.y_true)
        if not unique_true.issubset({0, 1}):
            raise ValueError(f"y_true must contain only 0 and 1, found: {unique_true}")
            
        # y_proba の範囲チェック
        if np.any(self.y_proba < 0) or np.any(self.y_proba > 1):
            self.logger.warning("y_proba contains values outside [0,1] range")
            # [0,1] にクリップ
            self.y_proba = np.clip(self.y_proba, 0, 1)
            
        # 欠損値チェック
        if np.any(np.isnan(self.y_true)) or np.any(np.isnan(self.y_proba)):
            raise ValueError("Data contains NaN values")
            
        # クラス分布チェック
        class_counts = np.bincount(self.y_true)
        if len(class_counts) < 2 or np.any(class_counts == 0):
            raise ValueError("Data must contain both positive and negative samples")
            
        self.logger.info(f"Data validation passed: {len(self.y_true)} samples, "
                        f"class distribution: {dict(enumerate(class_counts))}")
                        
    def compute_roc_analysis(self) -> Dict[str, Any]:
        """
        ROC分析・AUC計算
        
        Returns:
            ROC分析結果辞書
        """
        self.logger.info("Computing ROC analysis...")
        
        # ROC曲線計算
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba)
        roc_auc = auc(fpr, tpr)
        
        # 最適動作点（左上に最も近い点）
        optimal_idx = np.argmax(tpr - fpr)
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        optimal_threshold_roc = thresholds[optimal_idx]
        
        # Youden's J statistic
        youdens_j = tpr - fpr
        youdens_idx = np.argmax(youdens_j)
        youdens_threshold = thresholds[youdens_idx]
        youdens_score = youdens_j[youdens_idx]
        
        roc_results = {
            'auc': float(roc_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(), 
            'thresholds': thresholds.tolist(),
            'optimal_point': {
                'fpr': float(optimal_fpr),
                'tpr': float(optimal_tpr),
                'threshold': float(optimal_threshold_roc),
                'index': int(optimal_idx)
            },
            'youdens_index': {
                'threshold': float(youdens_threshold),
                'j_score': float(youdens_score),
                'index': int(youdens_idx)
            }
        }
        
        self.evaluation_results['roc_analysis'] = roc_results
        self.logger.info(f"ROC analysis completed: AUC = {roc_auc:.3f}")
        
        return roc_results
        
    def analyze_prediction_distribution(self) -> Dict[str, Any]:
        """
        予測確率分布の詳細分析
        
        Returns:
            分布分析結果辞書
        """
        self.logger.info("Analyzing prediction distribution...")
        
        # クラス別の予測確率分布
        pos_scores = self.y_proba[self.y_true == 1]
        neg_scores = self.y_proba[self.y_true == 0]
        
        # 統計量計算
        def compute_stats(scores):
            return {
                'count': len(scores),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores)),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75))
            }
            
        pos_stats = compute_stats(pos_scores)
        neg_stats = compute_stats(neg_scores)
        
        # 分離度測定
        separation_metrics = self._compute_separation_metrics(pos_scores, neg_scores)
        
        # ヒストグラム用データ（20bins）
        bins = np.linspace(0, 1, 21)
        pos_hist, _ = np.histogram(pos_scores, bins=bins)
        neg_hist, _ = np.histogram(neg_scores, bins=bins)
        
        distribution_results = {
            'positive_class': pos_stats,
            'negative_class': neg_stats,
            'separation_metrics': separation_metrics,
            'histogram_data': {
                'bins': bins.tolist(),
                'positive_hist': pos_hist.tolist(),
                'negative_hist': neg_hist.tolist()
            }
        }
        
        self.evaluation_results['distribution_analysis'] = distribution_results
        self.logger.info("Distribution analysis completed")
        
        return distribution_results
        
    def _compute_separation_metrics(self, pos_scores: np.ndarray, neg_scores: np.ndarray) -> Dict[str, float]:
        """分布の分離度メトリクス計算"""
        # Cohen's d (効果量)
        pooled_std = np.sqrt(((len(pos_scores) - 1) * np.var(pos_scores) + 
                             (len(neg_scores) - 1) * np.var(neg_scores)) / 
                            (len(pos_scores) + len(neg_scores) - 2))
        cohens_d = (np.mean(pos_scores) - np.mean(neg_scores)) / pooled_std if pooled_std > 0 else 0
        
        # 重複度（重複する範囲の割合）
        pos_min, pos_max = np.min(pos_scores), np.max(pos_scores)
        neg_min, neg_max = np.min(neg_scores), np.max(neg_scores)
        
        overlap_start = max(pos_min, neg_min)
        overlap_end = min(pos_max, neg_max)
        overlap_range = max(0, overlap_end - overlap_start)
        total_range = max(pos_max, neg_max) - min(pos_min, neg_min)
        
        overlap_ratio = overlap_range / total_range if total_range > 0 else 0
        
        # Kolmogorov-Smirnov統計量（簡易版）
        from scipy import stats
        ks_statistic, ks_pvalue = stats.ks_2samp(pos_scores, neg_scores)
        
        return {
            'cohens_d': float(cohens_d),
            'overlap_ratio': float(overlap_ratio),
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue)
        }
        
    def optimize_threshold_f1(self) -> Dict[str, Any]:
        """
        F1スコア最大化による閾値最適化（1000分割）
        
        Returns:
            閾値最適化結果辞書
        """
        self.logger.info(f"Optimizing threshold with {self.threshold_resolution} divisions...")
        
        # 閾値候補生成
        thresholds = np.linspace(0.000, 1.000, self.threshold_resolution + 1)
        
        # 各閾値での性能計算
        metrics_by_threshold = []
        
        for threshold in thresholds:
            y_pred = (self.y_proba >= threshold).astype(int)
            
            # 基本メトリクス計算
            accuracy = accuracy_score(self.y_true, y_pred)
            precision = precision_score(self.y_true, y_pred, zero_division=0)
            recall = recall_score(self.y_true, y_pred, zero_division=0)
            f1 = f1_score(self.y_true, y_pred, zero_division=0)
            
            # 混同行列
            cm = confusion_matrix(self.y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            metrics_by_threshold.append({
                'threshold': float(threshold),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            })
            
        # F1スコア最大の閾値を特定
        f1_scores = [m['f1_score'] for m in metrics_by_threshold]
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        # 最適閾値での詳細メトリクス
        optimal_metrics = metrics_by_threshold[optimal_idx]
        
        # デフォルト閾値(0.5)での性能も記録
        default_idx = np.argmin(np.abs(thresholds - 0.5))
        default_metrics = metrics_by_threshold[default_idx]
        
        optimization_results = {
            'threshold_resolution': self.threshold_resolution,
            'optimal_threshold': float(optimal_threshold),
            'optimal_f1_score': float(optimal_f1),
            'optimal_metrics': optimal_metrics,
            'default_threshold': 0.5,
            'default_metrics': default_metrics,
            'all_thresholds': thresholds.tolist(),
            'all_f1_scores': f1_scores,
            'all_precision_scores': [m['precision'] for m in metrics_by_threshold],
            'all_recall_scores': [m['recall'] for m in metrics_by_threshold],
            'improvement_over_default': float(optimal_f1 - default_metrics['f1_score'])
        }
        
        self.evaluation_results['threshold_optimization'] = optimization_results
        
        self.logger.info(f"Threshold optimization completed: "
                        f"optimal = {optimal_threshold:.3f}, F1 = {optimal_f1:.3f}")
                        
        return optimization_results
        
    def compute_precision_recall_analysis(self) -> Dict[str, Any]:
        """
        Precision-Recall分析
        
        Returns:
            PR分析結果辞書  
        """
        self.logger.info("Computing Precision-Recall analysis...")
        
        # PR曲線計算
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_proba)
        pr_auc = auc(recall, precision)
        
        # F1スコア曲線（PR曲線の各点でのF1）
        f1_scores = []
        for p, r in zip(precision, recall):
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_scores.append(f1)
            
        pr_results = {
            'pr_auc': float(pr_auc),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': np.concatenate([[1.0], thresholds]).tolist(),  # 最初の点の閾値を1.0とする
            'f1_scores': f1_scores
        }
        
        self.evaluation_results['pr_analysis'] = pr_results
        self.logger.info(f"PR analysis completed: PR-AUC = {pr_auc:.3f}")
        
        return pr_results
        
    def compare_threshold_performance(self, custom_thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        複数閾値での性能比較
        
        Args:
            custom_thresholds: カスタム閾値リスト（None の場合は標準設定を使用）
            
        Returns:
            閾値比較結果辞書
        """
        if custom_thresholds is None:
            # デフォルト、最適、および代表的な閾値
            optimal_threshold = self.evaluation_results.get('threshold_optimization', {}).get('optimal_threshold', 0.5)
            custom_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, optimal_threshold]
            custom_thresholds = sorted(list(set(custom_thresholds)))  # 重複除去・ソート
            
        self.logger.info(f"Comparing performance across {len(custom_thresholds)} thresholds...")
        
        comparison_results = []
        
        for threshold in custom_thresholds:
            y_pred = (self.y_proba >= threshold).astype(int)
            
            # 基本メトリクス
            cm = confusion_matrix(self.y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            accuracy = accuracy_score(self.y_true, y_pred)
            precision = precision_score(self.y_true, y_pred, zero_division=0)
            recall = recall_score(self.y_true, y_pred, zero_division=0)
            f1 = f1_score(self.y_true, y_pred, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # 詳細メトリクス
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            # 予測分布
            total_positive_predictions = tp + fp
            total_negative_predictions = tn + fn
            
            result = {
                'threshold': float(threshold),
                'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'fpr': float(fpr),
                'fnr': float(fnr),
                'positive_predictions': int(total_positive_predictions),
                'negative_predictions': int(total_negative_predictions)
            }
            
            comparison_results.append(result)
            
        comparison_data = {
            'thresholds_compared': custom_thresholds,
            'results': comparison_results
        }
        
        self.evaluation_results['threshold_comparison'] = comparison_data
        self.logger.info("Threshold comparison completed")
        
        return comparison_data
        
    def compute_business_metrics(self, fn_cost: float = 1000000, fp_cost: float = 7200) -> Dict[str, Any]:
        """
        ビジネス指標の計算
        
        Args:
            fn_cost: False Negative 1件あたりのコスト（円）
            fp_cost: False Positive 1件あたりのコスト（円）
            
        Returns:
            ビジネス指標辞書
        """
        self.logger.info("Computing business metrics...")
        
        # 最適閾値とデフォルト閾値での比較
        optimal_threshold = self.evaluation_results.get('threshold_optimization', {}).get('optimal_threshold', 0.5)
        
        business_results = {}
        
        for threshold_name, threshold_value in [('default', 0.5), ('optimal', optimal_threshold)]:
            y_pred = (self.y_proba >= threshold_value).astype(int)
            cm = confusion_matrix(self.y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # コスト計算
            opportunity_cost = fn * fn_cost  # 見逃しによる機会損失
            labor_cost = fp * fp_cost        # 誤検出による工数増加
            total_cost = opportunity_cost + labor_cost
            
            # 効果計算（全て手動確認する場合との比較）
            total_samples = len(self.y_true)
            manual_cost = total_samples * fp_cost  # 全件手動確認のコスト
            cost_reduction = manual_cost - total_cost
            
            business_results[threshold_name] = {
                'threshold': float(threshold_value),
                'opportunity_cost': float(opportunity_cost),
                'labor_cost': float(labor_cost), 
                'total_cost': float(total_cost),
                'manual_cost': float(manual_cost),
                'cost_reduction': float(cost_reduction),
                'roi': float(cost_reduction / manual_cost) if manual_cost > 0 else 0.0,
                'fn_count': int(fn),
                'fp_count': int(fp)
            }
            
        business_results['cost_assumptions'] = {
            'fn_cost_per_case': float(fn_cost),
            'fp_cost_per_case': float(fp_cost)
        }
        
        self.evaluation_results['business_metrics'] = business_results
        self.logger.info("Business metrics computation completed")
        
        return business_results
        
    def generate_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        包括的評価メトリクスの生成
        
        Returns:
            全評価結果を統合した辞書
        """
        self.logger.info("Generating comprehensive evaluation metrics...")
        
        # 各分析の実行（未実行の場合）
        if 'roc_analysis' not in self.evaluation_results:
            self.compute_roc_analysis()
            
        if 'distribution_analysis' not in self.evaluation_results:
            self.analyze_prediction_distribution()
            
        if 'threshold_optimization' not in self.evaluation_results:
            self.optimize_threshold_f1()
            
        if 'pr_analysis' not in self.evaluation_results:
            self.compute_precision_recall_analysis()
            
        if 'threshold_comparison' not in self.evaluation_results:
            self.compare_threshold_performance()
            
        if 'business_metrics' not in self.evaluation_results:
            self.compute_business_metrics()
            
        # メタデータ追加
        comprehensive_results = {
            'metadata': {
                'evaluation_timestamp': datetime.now().isoformat(),
                'sample_count': len(self.y_true),
                'positive_class_count': int(np.sum(self.y_true)),
                'negative_class_count': int(len(self.y_true) - np.sum(self.y_true)),
                'random_seed': self.random_seed,
                'threshold_resolution': self.threshold_resolution
            },
            'evaluation_results': self.evaluation_results
        }
        
        self.logger.info("Comprehensive metrics generation completed")
        return comprehensive_results
        
    def save_metrics(self, output_path: str, eval_id: str) -> str:
        """
        評価メトリクスをJSON形式で保存
        
        Args:
            output_path: 保存先パス
            eval_id: 評価ID
            
        Returns:
            保存されたファイルパス
        """
        comprehensive_metrics = self.generate_comprehensive_metrics()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ファイル名にeval_IDを含める
        if eval_id and eval_id not in str(output_path.name):
            stem = output_path.stem
            suffix = output_path.suffix
            filename = f"{stem}_{eval_id}{suffix}"
            output_path = output_path.parent / filename
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_metrics, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved comprehensive metrics to: {output_path}")
        return str(output_path)


def main():
    """テスト実行用メイン関数"""
    print("=== Strict Binary Evaluator Test ===")
    
    # テストデータ生成
    np.random.seed(42)
    n_samples = 1000
    
    # 不均衡な分類問題をシミュレート
    n_positive = int(n_samples * 0.25)  # 25%が正例
    
    y_true = np.concatenate([
        np.ones(n_positive),
        np.zeros(n_samples - n_positive)
    ])
    
    # 正例と負例で異なる分布から予測確率を生成
    y_proba = np.concatenate([
        np.random.beta(3, 1.5, n_positive),      # 正例: 高い確率に偏り
        np.random.beta(1.5, 3, n_samples - n_positive)  # 負例: 低い確率に偏り
    ])
    
    # データをシャッフル
    indices = np.random.permutation(n_samples)
    y_true = y_true[indices]
    y_proba = y_proba[indices]
    
    # テスト用DataFrame作成
    test_df = pd.DataFrame({
        'publication_number': [f'TEST-{i:04d}' for i in range(n_samples)],
        'y_true': y_true,
        'y_proba': y_proba
    })
    
    # 一時ファイルに保存
    temp_dir = Path("analysis/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    test_file = temp_dir / "test_evaluation_data.csv"
    test_df.to_csv(test_file, index=False)
    
    print(f"Generated test data: {n_samples} samples ({n_positive} positive, {n_samples-n_positive} negative)")
    
    try:
        # 厳密評価エンジン初期化・実行
        evaluator = StrictBinaryEvaluator(threshold_resolution=100)  # テスト用に分割数削減
        
        print("\n1. Loading test data...")
        evaluator.load_evaluation_data(str(test_file))
        
        print("2. Computing ROC analysis...")
        roc_results = evaluator.compute_roc_analysis()
        print(f"   AUC: {roc_results['auc']:.3f}")
        
        print("3. Analyzing prediction distribution...")
        dist_results = evaluator.analyze_prediction_distribution()
        print(f"   Positive class mean: {dist_results['positive_class']['mean']:.3f}")
        print(f"   Negative class mean: {dist_results['negative_class']['mean']:.3f}")
        
        print("4. Optimizing threshold...")
        opt_results = evaluator.optimize_threshold_f1()
        print(f"   Optimal threshold: {opt_results['optimal_threshold']:.3f}")
        print(f"   Optimal F1: {opt_results['optimal_f1_score']:.3f}")
        
        print("5. Computing business metrics...")
        business_results = evaluator.compute_business_metrics()
        default_cost = business_results['default']['total_cost']
        optimal_cost = business_results['optimal']['total_cost']
        print(f"   Cost reduction: {default_cost - optimal_cost:.0f} yen")
        
        print("6. Saving comprehensive metrics...")
        eval_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metrics_file = evaluator.save_metrics("analysis/temp/test_metrics.json", eval_id)
        print(f"   Saved to: {metrics_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("\n=== Test completed successfully ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())