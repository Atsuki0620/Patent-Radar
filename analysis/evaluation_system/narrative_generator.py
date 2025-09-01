#!/usr/bin/env python3
"""
Patent-Radar 包括的評価システム - 自然言語解説生成
評価結果の自然言語での解釈・解説を生成
"""

import json
import math
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging


class PatentEvaluationNarrativeGenerator:
    """自然言語解説生成クラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        
        # AUC評価基準
        self.auc_thresholds = {
            0.95: "優秀",
            0.90: "良好", 
            0.80: "改善要",
            0.70: "要再設計",
            0.50: "実用化困難"
        }
        
        # F1スコア評価基準
        self.f1_thresholds = {
            0.90: "優秀",
            0.80: "良好",
            0.70: "改善要", 
            0.60: "要改良",
            0.50: "大幅改良必要"
        }
        
    def generate_executive_summary(self, evaluation_results: Dict) -> str:
        """
        エグゼクティブサマリー生成
        
        Args:
            evaluation_results: 評価結果辞書
            
        Returns:
            エグゼクティブサマリーのHTML文字列
        """
        try:
            # 基本指標取得
            metrics = evaluation_results["optimal_metrics"]
            roc_data = evaluation_results["roc_analysis"]
            threshold_data = evaluation_results["threshold_optimization"]
            data_stats = evaluation_results["data_statistics"]
            
            accuracy = metrics["accuracy"]
            precision = metrics["precision"] 
            recall = metrics["recall"]
            f1_score = metrics["f1_score"]
            auc_score = roc_data["auc"]
            optimal_threshold = threshold_data["optimal_threshold"]
            
            # AUC評価
            auc_rating = self._get_rating_from_thresholds(auc_score, self.auc_thresholds)
            f1_rating = self._get_rating_from_thresholds(f1_score, self.f1_thresholds)
            
            # ビジネス影響
            total_samples = data_stats["total_samples"]
            hit_samples = data_stats["positive_samples"]
            miss_samples = data_stats["negative_samples"]
            
            fn_count = metrics["fn"]
            fp_count = metrics["fp"]
            
            summary_html = f"""
            <div class="executive-summary">
                <h3>エグゼクティブサマリー</h3>
                
                <div class="key-findings">
                    <h4>主要な評価結果</h4>
                    <ul>
                        <li><strong>総合性能</strong>: AUC {auc_score:.3f} ({auc_rating}) - F1スコア {f1_score:.3f} ({f1_rating})</li>
                        <li><strong>精度指標</strong>: Accuracy {accuracy:.1%}, Precision {precision:.1%}, Recall {recall:.1%}</li>
                        <li><strong>最適閾値</strong>: {optimal_threshold:.3f} (F1スコア最大化による決定)</li>
                        <li><strong>データ規模</strong>: 全{total_samples:,}件 (HIT: {hit_samples:,}件, MISS: {miss_samples:,}件)</li>
                    </ul>
                </div>
                
                <div class="business-impact">
                    <h4>ビジネス影響評価</h4>
                    <ul>
                        <li><strong>見逃しリスク</strong>: {fn_count}件のHIT特許を見逃し ({fn_count/hit_samples:.1%} of HITs)</li>
                        <li><strong>工数増加</strong>: {fp_count}件の誤検出により追加確認が必要</li>
                        <li><strong>システム効果</strong>: {self._assess_system_effectiveness(auc_score, f1_score, fn_count, fp_count)}</li>
                    </ul>
                </div>
                
                <div class="recommendations">
                    <h4>推奨事項</h4>
                    {self._generate_recommendations(auc_score, f1_score, fn_count, fp_count, recall)}
                </div>
            </div>
            """
            
            return summary_html
            
        except Exception as e:
            self.logger.error(f"Executive summary generation failed: {e}")
            return "<p>エグゼクティブサマリーの生成中にエラーが発生しました。</p>"
    
    def generate_data_overview_explanation(self, data_stats: Dict) -> str:
        """
        データ概要の解説生成
        
        Args:
            data_stats: データ統計辞書
            
        Returns:
            データ概要解説のHTML文字列
        """
        try:
            total = data_stats["total_samples"]
            positive = data_stats["positive_samples"] 
            negative = data_stats["negative_samples"]
            balance_ratio = positive / total if total > 0 else 0
            
            explanation = f"""
            <div class="data-explanation">
                <h4>目的</h4>
                <p>評価に使用するゴールドスタンダードデータセットの品質・代表性・均衡性を確認し、
                評価結果の信頼性を検証します。</p>
                
                <h4>結果</h4>
                <ul>
                    <li>総サンプル数: {total:,}件の特許データ</li>
                    <li>HIT特許: {positive:,}件 ({balance_ratio:.1%})</li>
                    <li>MISS特許: {negative:,}件 ({1-balance_ratio:.1%})</li>
                    <li>クラス均衡度: {self._assess_class_balance(balance_ratio)}</li>
                </ul>
                
                <h4>分析</h4>
                <p>{self._generate_data_quality_analysis(total, balance_ratio)}</p>
            </div>
            """
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Data overview explanation generation failed: {e}")
            return "<p>データ概要の説明生成中にエラーが発生しました。</p>"
    
    def generate_roc_analysis_explanation(self, roc_data: Dict, optimal_metrics: Dict) -> str:
        """
        ROC分析の解説生成
        
        Args:
            roc_data: ROC分析結果
            optimal_metrics: 最適閾値での指標
            
        Returns:
            ROC分析解説のHTML文字列
        """
        try:
            auc_score = roc_data["auc"]
            auc_rating = self._get_rating_from_thresholds(auc_score, self.auc_thresholds)
            
            explanation = f"""
            <div class="roc-explanation">
                <h4>目的</h4>
                <p>ROC曲線とAUC（Area Under the Curve）により、閾値によらない分類器の本質的な識別能力を評価し、
                ランダム分類との比較でシステムの有効性を検証します。</p>
                
                <h4>結果</h4>
                <ul>
                    <li>AUCスコア: {auc_score:.3f}</li>
                    <li>性能評価: {auc_rating}</li>
                    <li>ランダム分類との差: {auc_score - 0.5:.3f}ポイント向上</li>
                    <li>完全分類との距離: {1.0 - auc_score:.3f}ポイント</li>
                </ul>
                
                <h4>分析</h4>
                <p>{self._generate_auc_interpretation(auc_score, auc_rating)}</p>
                
                <div class="technical-details">
                    <h5>技術的詳細</h5>
                    <p>ROC曲線は、各閾値でのTrue Positive Rate (TPR=Recall)とFalse Positive Rate (FPR)の関係を示します。
                    理想的な分類器は左上隅（TPR=1, FPR=0）を通り、AUC=1.0となります。
                    現在のAUC={auc_score:.3f}は、{self._interpret_auc_business_meaning(auc_score)}。</p>
                </div>
            </div>
            """
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"ROC analysis explanation generation failed: {e}")
            return "<p>ROC分析の説明生成中にエラーが発生しました。</p>"
    
    def generate_threshold_optimization_explanation(self, threshold_data: Dict) -> str:
        """
        閾値最適化の解説生成
        
        Args:
            threshold_data: 閾値最適化結果
            
        Returns:
            閾値最適化解説のHTML文字列
        """
        try:
            optimal_threshold = threshold_data["optimal_threshold"]
            max_f1 = threshold_data["max_f1_score"]
            f1_at_05 = threshold_data.get("f1_at_0.5", "N/A")
            
            improvement = ""
            if isinstance(f1_at_05, float):
                improvement = f"標準閾値0.5でのF1={f1_at_05:.3f}と比較して{max_f1 - f1_at_05:+.3f}の改善"
            
            explanation = f"""
            <div class="threshold-explanation">
                <h4>目的</h4>
                <p>1000分割の網羅的探索により、F1スコア（Precision・Recall調和平均）を最大化する
                最適な判定閾値を決定し、標準閾値0.5との比較で改善効果を定量化します。</p>
                
                <h4>結果</h4>
                <ul>
                    <li>最適閾値: {optimal_threshold:.3f}</li>
                    <li>最大F1スコア: {max_f1:.3f}</li>
                    <li>標準閾値との比較: {improvement if improvement else "比較データなし"}</li>
                    <li>探索範囲: 0.000-1.000 (1000分割による精密最適化)</li>
                </ul>
                
                <h4>分析</h4>
                <p>{self._generate_threshold_interpretation(optimal_threshold, max_f1)}</p>
                
                <div class="practical-implications">
                    <h5>実運用への示唆</h5>
                    <p>{self._generate_threshold_practical_advice(optimal_threshold)}</p>
                </div>
            </div>
            """
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Threshold optimization explanation generation failed: {e}")
            return "<p>閾値最適化の説明生成中にエラーが発生しました。</p>"
    
    def generate_confusion_matrix_explanation(self, optimal_metrics: Dict) -> str:
        """
        混同行列の解説生成
        
        Args:
            optimal_metrics: 最適閾値での指標
            
        Returns:
            混同行列解説のHTML文字列
        """
        try:
            tp, fp, tn, fn = optimal_metrics["tp"], optimal_metrics["fp"], optimal_metrics["tn"], optimal_metrics["fn"]
            total = tp + fp + tn + fn
            
            explanation = f"""
            <div class="confusion-matrix-explanation">
                <h4>目的</h4>
                <p>最適閾値での分類結果を4象限（TP/FP/TN/FN）に分解し、
                各種エラーの発生パターンとビジネス影響を詳細に分析します。</p>
                
                <h4>結果</h4>
                <div class="confusion-breakdown">
                    <ul>
                        <li><strong>True Positive (TP)</strong>: {tp}件 - 正しくHITと判定</li>
                        <li><strong>False Positive (FP)</strong>: {fp}件 - 誤ってHITと判定（工数増加）</li>
                        <li><strong>True Negative (TN)</strong>: {tn}件 - 正しくMISSと判定</li>
                        <li><strong>False Negative (FN)</strong>: {fn}件 - 見逃されたHIT（機会損失）</li>
                    </ul>
                </div>
                
                <h4>分析</h4>
                <div class="error-analysis">
                    <h5>エラー分析</h5>
                    <p><strong>False Negative (見逃し)</strong>: {fn}件 ({fn/total:.1%})</p>
                    <p>{self._analyze_false_negatives(fn, tp)}</p>
                    
                    <p><strong>False Positive (誤検出)</strong>: {fp}件 ({fp/total:.1%})</p> 
                    <p>{self._analyze_false_positives(fp, tn)}</p>
                </div>
                
                <div class="business-priorities">
                    <h5>ビジネス優先順位</h5>
                    <p>{self._prioritize_error_types(fn, fp, tp, tn)}</p>
                </div>
            </div>
            """
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Confusion matrix explanation generation failed: {e}")
            return "<p>混同行列の説明生成中にエラーが発生しました。</p>"
    
    def generate_distribution_analysis_explanation(self, distribution_data: Dict) -> str:
        """
        予測分布分析の解説生成
        
        Args:
            distribution_data: 分布分析結果
            
        Returns:
            分布分析解説のHTML文字列
        """
        try:
            hit_stats = distribution_data["hit_statistics"]
            miss_stats = distribution_data["miss_statistics"] 
            overlap_info = distribution_data.get("overlap_analysis", {})
            
            explanation = f"""
            <div class="distribution-explanation">
                <h4>目的</h4>
                <p>HIT特許とMISS特許の予測確率分布を比較し、分類器の判定根拠の妥当性、
                分布の分離度、および境界領域の困難事例を分析します。</p>
                
                <h4>結果</h4>
                <div class="distribution-stats">
                    <h5>HIT特許の予測確率分布</h5>
                    <ul>
                        <li>平均: {hit_stats['mean']:.3f}</li>
                        <li>中央値: {hit_stats['median']:.3f}</li>
                        <li>標準偏差: {hit_stats['std']:.3f}</li>
                        <li>範囲: {hit_stats['min']:.3f} - {hit_stats['max']:.3f}</li>
                    </ul>
                    
                    <h5>MISS特許の予測確率分布</h5>
                    <ul>
                        <li>平均: {miss_stats['mean']:.3f}</li>
                        <li>中央値: {miss_stats['median']:.3f}</li>
                        <li>標準偏差: {miss_stats['std']:.3f}</li>
                        <li>範囲: {miss_stats['min']:.3f} - {miss_stats['max']:.3f}</li>
                    </ul>
                </div>
                
                <h4>分析</h4>
                <p>{self._analyze_distribution_separation(hit_stats, miss_stats)}</p>
                
                <div class="calibration-assessment">
                    <h5>キャリブレーション評価</h5>
                    <p>{self._assess_calibration(hit_stats, miss_stats)}</p>
                </div>
                
                <div class="boundary-cases">
                    <h5>境界事例の考察</h5>
                    <p>{self._analyze_boundary_cases(hit_stats, miss_stats)}</p>
                </div>
            </div>
            """
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Distribution analysis explanation generation failed: {e}")
            return "<p>分布分析の説明生成中にエラーが発生しました。</p>"
    
    def generate_performance_comparison_explanation(self, evaluation_results: Dict) -> str:
        """
        性能比較の解説生成（将来の比較評価用）
        
        Args:
            evaluation_results: 評価結果辞書
            
        Returns:
            性能比較解説のHTML文字列
        """
        optimal_metrics = evaluation_results["optimal_metrics"]
        standard_metrics = evaluation_results.get("standard_threshold_metrics", {})
        
        explanation = f"""
        <div class="performance-comparison">
            <h4>目的</h4>
            <p>最適閾値設定と標準閾値（0.5）での性能を比較し、
            閾値最適化による改善効果を定量化します。</p>
            
            <h4>結果</h4>
            <div class="comparison-table">
                <table border="1" cellpadding="5" cellspacing="0">
                    <thead>
                        <tr>
                            <th>指標</th>
                            <th>最適閾値</th>
                            <th>標準閾値(0.5)</th>
                            <th>改善</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Accuracy</td>
                            <td>{optimal_metrics['accuracy']:.3f}</td>
                            <td>{standard_metrics.get('accuracy', 'N/A')}</td>
                            <td>{self._calculate_improvement(optimal_metrics.get('accuracy'), standard_metrics.get('accuracy'))}</td>
                        </tr>
                        <tr>
                            <td>Precision</td>
                            <td>{optimal_metrics['precision']:.3f}</td>
                            <td>{standard_metrics.get('precision', 'N/A')}</td>
                            <td>{self._calculate_improvement(optimal_metrics.get('precision'), standard_metrics.get('precision'))}</td>
                        </tr>
                        <tr>
                            <td>Recall</td>
                            <td>{optimal_metrics['recall']:.3f}</td>
                            <td>{standard_metrics.get('recall', 'N/A')}</td>
                            <td>{self._calculate_improvement(optimal_metrics.get('recall'), standard_metrics.get('recall'))}</td>
                        </tr>
                        <tr>
                            <td>F1-Score</td>
                            <td>{optimal_metrics['f1_score']:.3f}</td>
                            <td>{standard_metrics.get('f1_score', 'N/A')}</td>
                            <td>{self._calculate_improvement(optimal_metrics.get('f1_score'), standard_metrics.get('f1_score'))}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <h4>分析</h4>
            <p>閾値最適化により、{self._summarize_optimization_benefits(optimal_metrics, standard_metrics)}</p>
        </div>
        """
        
        return explanation
    
    # ヘルパーメソッド群
    def _get_rating_from_thresholds(self, score: float, thresholds: Dict[float, str]) -> str:
        """閾値辞書から評価を取得"""
        for threshold in sorted(thresholds.keys(), reverse=True):
            if score >= threshold:
                return thresholds[threshold]
        return "非常に低い"
    
    def _assess_system_effectiveness(self, auc: float, f1: float, fn: int, fp: int) -> str:
        """システム効果の総合評価"""
        if auc >= 0.95 and f1 >= 0.90:
            return "優秀 - 実運用強く推奨"
        elif auc >= 0.90 and f1 >= 0.80:
            return "良好 - 実運用推奨"
        elif auc >= 0.80:
            return "改善要 - 追加改良後に実用化検討"
        else:
            return "要再設計 - 大幅改良が必要"
    
    def _generate_recommendations(self, auc: float, f1: float, fn: int, fp: int, recall: float) -> str:
        """推奨事項生成"""
        recommendations = []
        
        if fn > 0:
            recommendations.append(f"• HIT特許見逃し({fn}件)の削減: 閾値調整または追加特徴量検討")
        
        if fp > fn * 2:
            recommendations.append("• 誤検出(FP)削減: Precision向上のためのモデル改良")
        
        if recall < 0.85:
            recommendations.append("• Recall向上: 見逃し防止のための感度向上")
        
        if auc < 0.90:
            recommendations.append("• 基本識別能力向上: モデルアーキテクチャの見直し")
        
        return "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"
    
    def _assess_class_balance(self, ratio: float) -> str:
        """クラス均衡度評価"""
        if 0.4 <= ratio <= 0.6:
            return "良好な均衡"
        elif 0.3 <= ratio <= 0.7:
            return "やや不均衡"
        else:
            return "大幅な不均衡"
    
    def _generate_data_quality_analysis(self, total: int, balance_ratio: float) -> str:
        """データ品質分析"""
        quality_notes = []
        
        if total < 100:
            quality_notes.append("サンプル数が少ないため、評価の信頼性に注意が必要")
        elif total >= 1000:
            quality_notes.append("十分なサンプル数により、信頼性の高い評価が可能")
        
        if balance_ratio < 0.2 or balance_ratio > 0.8:
            quality_notes.append("クラス不均衡により、少数クラスの評価精度に影響の可能性")
        
        return "。".join(quality_notes) + "。" if quality_notes else "データ品質は良好です。"
    
    def _generate_auc_interpretation(self, auc: float, rating: str) -> str:
        """AUC解釈生成"""
        base_text = f"AUC={auc:.3f}は{rating}レベルの性能を示しています。"
        
        if auc >= 0.95:
            return base_text + "実用レベルの高い識別能力を持ち、ビジネス価値の高いシステムです。"
        elif auc >= 0.90:
            return base_text + "実用可能な識別能力を持ち、適切な運用により効果が期待できます。"
        elif auc >= 0.80:
            return base_text + "基本的な識別能力はあるものの、追加改良により性能向上が必要です。"
        else:
            return base_text + "十分な識別能力が不足しており、大幅な改良が必要です。"
    
    def _interpret_auc_business_meaning(self, auc: float) -> str:
        """AUCのビジネス的意味"""
        improvement = (auc - 0.5) * 100
        return f"ランダム分類と比較して{improvement:.1f}%の改善を実現しており、特許分析業務の効率化に寄与します"
    
    def _generate_threshold_interpretation(self, threshold: float, f1: float) -> str:
        """閾値解釈生成"""
        if threshold < 0.3:
            return f"最適閾値{threshold:.3f}は保守的設定です。Recall重視によりHIT特許の見逃しを最小化する戦略です。"
        elif threshold > 0.7:
            return f"最適閾値{threshold:.3f}は積極的設定です。Precision重視により誤検出を最小化する戦略です。"
        else:
            return f"最適閾値{threshold:.3f}はバランス型設定です。Precision・Recall双方を考慮した均衡戦略です。"
    
    def _generate_threshold_practical_advice(self, threshold: float) -> str:
        """閾値の実用的助言"""
        if threshold < 0.3:
            return "低い閾値設定により多くの特許がHIT候補となるため、確認工数の増加に注意が必要です。"
        elif threshold > 0.7:
            return "高い閾値設定により確実な特許のみHIT判定されるため、見逃しリスクの監視が重要です。"
        else:
            return "バランスの取れた閾値設定により、効率的な特許スクリーニングが期待できます。"
    
    def _analyze_false_negatives(self, fn: int, tp: int) -> str:
        """False Negative分析"""
        if fn == 0:
            return "HIT特許の見逃しはありません。優秀な検出性能です。"
        
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        if fn_rate <= 0.1:
            return f"見逃し率{fn_rate:.1%}は良好な水準です。機会損失リスクは限定的です。"
        elif fn_rate <= 0.2:
            return f"見逃し率{fn_rate:.1%}はやや高めです。重要特許の見逃し防止策を検討してください。"
        else:
            return f"見逃し率{fn_rate:.1%}は高く、改善が必要です。感度向上が急務です。"
    
    def _analyze_false_positives(self, fp: int, tn: int) -> str:
        """False Positive分析"""
        if fp == 0:
            return "誤検出はありません。優秀な精度性能です。"
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        if fpr <= 0.1:
            return f"誤検出率{fpr:.1%}は良好な水準です。工数増加への影響は限定的です。"
        elif fpr <= 0.2:
            return f"誤検出率{fpr:.1%}はやや高めです。確認工数の効率化を検討してください。"
        else:
            return f"誤検出率{fpr:.1%}は高く、改善が必要です。精度向上が急務です。"
    
    def _prioritize_error_types(self, fn: int, fp: int, tp: int, tn: int) -> str:
        """エラータイプの優先順位付け"""
        fn_cost = fn * 1000000  # 1件100万円の機会損失
        fp_cost = fp * 20000    # 1件2万円の確認工数
        
        if fn_cost > fp_cost * 2:
            return f"見逃し({fn}件)による機会損失リスク（推定{fn_cost:,}円）が誤検出コスト（推定{fp_cost:,}円）を大きく上回るため、Recall向上を最優先とすべきです。"
        elif fp_cost > fn_cost * 2:
            return f"誤検出({fp}件)による工数コスト（推定{fp_cost:,}円）が機会損失リスク（推定{fn_cost:,}円）を上回るため、Precision向上を優先すべきです。"
        else:
            return f"見逃しリスク（推定{fn_cost:,}円）と誤検出コスト（推定{fp_cost:,}円）がバランスしているため、F1スコア最大化が適切です。"
    
    def _analyze_distribution_separation(self, hit_stats: Dict, miss_stats: Dict) -> str:
        """分布分離度分析"""
        mean_diff = hit_stats['mean'] - miss_stats['mean']
        
        if mean_diff >= 0.6:
            return f"HIT分布平均({hit_stats['mean']:.3f})とMISS分布平均({miss_stats['mean']:.3f})の差{mean_diff:.3f}は大きく、良好な分離を示しています。"
        elif mean_diff >= 0.3:
            return f"分布の平均差{mean_diff:.3f}は中程度で、基本的な識別は可能ですが改善余地があります。"
        else:
            return f"分布の平均差{mean_diff:.3f}は小さく、分離度の改善が必要です。"
    
    def _assess_calibration(self, hit_stats: Dict, miss_stats: Dict) -> str:
        """キャリブレーション評価"""
        hit_mean = hit_stats['mean']
        
        if hit_mean >= 0.7:
            return "HIT特許の平均予測確率が高く、モデルの信頼度は適切に調整されています。"
        elif hit_mean >= 0.5:
            return "HIT特許の平均予測確率は中程度で、キャリブレーションの改善余地があります。"
        else:
            return "HIT特許の平均予測確率が低く、モデルのキャリブレーション調整が必要です。"
    
    def _analyze_boundary_cases(self, hit_stats: Dict, miss_stats: Dict) -> str:
        """境界事例分析"""
        return "予測確率0.4-0.6の境界領域には、判定困難な特許が集中している可能性があり、専門家レビューの対象として適切です。"
    
    def _calculate_improvement(self, optimal_val, standard_val) -> str:
        """改善度計算"""
        if optimal_val is None or standard_val is None or standard_val == 'N/A':
            return "N/A"
        
        try:
            improvement = optimal_val - standard_val
            return f"{improvement:+.3f}"
        except:
            return "N/A"
    
    def _summarize_optimization_benefits(self, optimal: Dict, standard: Dict) -> str:
        """最適化効果のサマリー"""
        if not standard or 'f1_score' not in standard:
            return "閾値最適化の効果測定にはベースライン比較データが必要です。"
        
        f1_improvement = optimal['f1_score'] - standard['f1_score']
        
        if f1_improvement >= 0.05:
            return f"F1スコアが{f1_improvement:.3f}向上し、大幅な性能改善を実現しています。"
        elif f1_improvement >= 0.02:
            return f"F1スコアが{f1_improvement:.3f}向上し、有意な性能改善が確認されます。"
        elif f1_improvement >= 0:
            return f"F1スコアが{f1_improvement:.3f}向上していますが、改善幅は限定的です。"
        else:
            return "標準閾値と比較して性能改善が見られません。最適化プロセスの見直しが必要です。"


def main():
    """テスト実行用メイン関数"""
    print("=== Patent Evaluation Narrative Generator Test ===")
    
    # サンプル評価結果データ
    sample_results = {
        "data_statistics": {
            "total_samples": 59,
            "positive_samples": 25,
            "negative_samples": 34
        },
        "optimal_metrics": {
            "accuracy": 0.8135,
            "precision": 0.8000,
            "recall": 0.8000,
            "f1_score": 0.8000,
            "tp": 20, "fp": 5, "tn": 28, "fn": 6
        },
        "roc_analysis": {
            "auc": 0.8853
        },
        "threshold_optimization": {
            "optimal_threshold": 0.485,
            "max_f1_score": 0.8000
        }
    }
    
    # 分布データサンプル
    distribution_data = {
        "hit_statistics": {
            "mean": 0.651, "median": 0.670, "std": 0.241,
            "min": 0.152, "max": 0.975
        },
        "miss_statistics": {
            "mean": 0.284, "median": 0.245, "std": 0.198,
            "min": 0.012, "max": 0.798
        }
    }
    
    generator = PatentEvaluationNarrativeGenerator()
    
    print("\n1. Executive Summary:")
    summary = generator.generate_executive_summary(sample_results)
    print(summary[:300] + "..." if len(summary) > 300 else summary)
    
    print("\n2. ROC Analysis:")
    roc_explanation = generator.generate_roc_analysis_explanation(
        sample_results["roc_analysis"], 
        sample_results["optimal_metrics"]
    )
    print(roc_explanation[:300] + "..." if len(roc_explanation) > 300 else roc_explanation)
    
    print("\n=== Test completed ===")


if __name__ == "__main__":
    main()