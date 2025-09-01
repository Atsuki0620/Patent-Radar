#!/usr/bin/env python3
"""
Patent-Radar 可視化エンジン
高品質グラフ生成（日本語対応・業務仕様特化）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

# 日本語フォント設定（Windows環境対応）
import matplotlib.font_manager as fm
import re

# Windows環境での日本語フォント設定
try:
    # フォント検索とキャッシュ更新
    available_font_names = [font.name for font in fm.fontManager.ttflist]
    jp_fonts = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'MS UI Gothic', 'BIZ UDGothic']
    
    # 利用可能な日本語フォントを検索
    selected_font = None
    for font in jp_fonts:
        if font in available_font_names:
            selected_font = font
            break
    
    if selected_font:
        # フォント設定
        plt.rcParams['font.family'] = [selected_font, 'DejaVu Sans']
        plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"Japanese font set to: {selected_font}")
    else:
        # 日本語フォントが見つからない場合のフォールバック
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("Warning: No Japanese fonts found, using DejaVu Sans")
        
except Exception as e:
    # フォント設定エラーは警告のみで継続
    print(f"Warning: Japanese font setup failed: {e}")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# スタイル設定  
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class PatentVisualizationEngine:
    """高品質グラフ生成エンジン"""
    
    def __init__(self, dpi: int = 300, figsize: Tuple[int, int] = (10, 8)):
        """
        初期化
        
        Args:
            dpi: 画像解像度
            figsize: 図のサイズ (width, height)
        """
        self.dpi = dpi
        self.figsize = figsize
        
        # カラーパレット（カラーブラインド対応）
        self.colors = {
            'primary': '#1f77b4',      # 青
            'secondary': '#ff7f0e',    # オレンジ  
            'success': '#2ca02c',      # 緑
            'warning': '#d62728',      # 赤
            'info': '#9467bd',         # 紫
            'neutral': '#7f7f7f',      # グレー
            'hit': '#d62728',          # HIT用（赤）
            'miss': '#1f77b4',         # MISS用（青）
            'optimal': '#2ca02c',      # 最適点用（緑）
            'baseline': '#7f7f7f'      # ベースライン用（グレー）
        }
        
        # グリッド・スタイル設定
        self.grid_style = {'alpha': 0.3, 'linestyle': '-', 'linewidth': 0.5}
        self.legend_style = {'framealpha': 0.9, 'fancybox': True, 'shadow': True}
        
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
            
    def create_roc_curve(self, roc_data: Dict, output_path: str, eval_id: str) -> str:
        """
        ROC曲線の生成
        
        Args:
            roc_data: ROC分析結果辞書
            output_path: 保存先パス
            eval_id: 評価ID
            
        Returns:
            保存されたファイルパス
        """
        self.logger.info("Creating ROC curve visualization...")
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # データ取得
        fpr = np.array(roc_data['fpr'])
        tpr = np.array(roc_data['tpr'])
        auc_score = roc_data['auc']
        optimal_point = roc_data['optimal_point']
        
        # ROC曲線
        ax.plot(fpr, tpr, color=self.colors['primary'], lw=3,
               label=f'ROC曲線 (AUC = {auc_score:.3f})')
        
        # ランダム分類器の線
        ax.plot([0, 1], [0, 1], color=self.colors['baseline'], lw=2, 
               linestyle='--', alpha=0.8, label='ランダム分類器 (AUC = 0.500)')
        
        # 最適動作点
        ax.scatter(optimal_point['fpr'], optimal_point['tpr'], 
                  color=self.colors['optimal'], s=150, zorder=5,
                  edgecolors='white', linewidth=2,
                  label=f'最適動作点 (FPR={optimal_point["fpr"]:.2f}, TPR={optimal_point["tpr"]:.2f})')
        
        # AUC領域の塗りつぶし
        ax.fill_between(fpr, 0, tpr, alpha=0.1, color=self.colors['primary'])
        
        # 装飾
        ax.grid(True, **self.grid_style)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('偽陽性率 (False Positive Rate)', fontsize=12, fontweight='bold')
        ax.set_ylabel('真陽性率 (True Positive Rate)', fontsize=12, fontweight='bold')
        ax.set_title('ROC曲線 - 受信者動作特性', fontsize=14, fontweight='bold', pad=20)
        
        # 凡例
        ax.legend(loc='lower right', **self.legend_style)
        
        # AUC解釈テキスト
        interpretation = self._interpret_auc(auc_score)
        ax.text(0.02, 0.98, f'AUC解釈: {interpretation}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        # 保存
        output_path = self._add_eval_id_to_path(output_path, eval_id)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"ROC curve saved to: {output_path}")
        return str(output_path)
        
    def create_prediction_distribution(self, dist_data: Dict, output_path: str, eval_id: str,
                                     optimal_threshold: float = 0.5) -> str:
        """
        予測確率分布の可視化
        
        Args:
            dist_data: 分布分析結果辞書
            output_path: 保存先パス
            eval_id: 評価ID
            optimal_threshold: 最適閾値
            
        Returns:
            保存されたファイルパス
        """
        self.logger.info("Creating prediction distribution visualization...")
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # データ取得
        pos_stats = dist_data['positive_class']
        neg_stats = dist_data['negative_class']
        hist_data = dist_data['histogram_data']
        
        # ヒストグラム描画
        bins = np.array(hist_data['bins'])
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        width = bins[1] - bins[0]
        
        ax.bar(bin_centers, hist_data['negative_hist'], width=width*0.8, alpha=0.6,
              color=self.colors['miss'], edgecolor='black', linewidth=0.5,
              label=f'MISS (n={neg_stats["count"]})')
        
        ax.bar(bin_centers, hist_data['positive_hist'], width=width*0.8, alpha=0.6,
              color=self.colors['hit'], edgecolor='black', linewidth=0.5,
              label=f'HIT (n={pos_stats["count"]})')
        
        # 統計値の線
        ax.axvline(pos_stats['mean'], color=self.colors['hit'], linestyle='--',
                  linewidth=2, alpha=0.8, label=f'HIT平均 ({pos_stats["mean"]:.3f})')
        ax.axvline(neg_stats['mean'], color=self.colors['miss'], linestyle='--',
                  linewidth=2, alpha=0.8, label=f'MISS平均 ({neg_stats["mean"]:.3f})')
        
        # 閾値線
        ax.axvline(optimal_threshold, color=self.colors['optimal'], linestyle='-',
                  linewidth=3, alpha=0.9, label=f'最適閾値 ({optimal_threshold:.3f})')
        ax.axvline(0.5, color=self.colors['baseline'], linestyle=':',
                  linewidth=2, alpha=0.7, label='標準閾値 (0.500)')
        
        # 装飾
        ax.grid(True, **self.grid_style)
        ax.set_xlabel('予測確率スコア', fontsize=12, fontweight='bold')
        ax.set_ylabel('件数', fontsize=12, fontweight='bold')
        ax.set_title('予測確率分布 - HIT/MISS別', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim([0, 1])
        
        # 凡例
        ax.legend(loc='upper right', **self.legend_style)
        
        # 分離度情報
        separation_info = dist_data.get('separation_metrics', {})
        cohens_d = separation_info.get('cohens_d', 0)
        overlap_ratio = separation_info.get('overlap_ratio', 0)
        
        info_text = f'分離度指標:\nCohen\'s d = {cohens_d:.3f}\n重複率 = {overlap_ratio:.1%}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=9, family='monospace')
        
        plt.tight_layout()
        
        # 保存
        output_path = self._add_eval_id_to_path(output_path, eval_id)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Prediction distribution saved to: {output_path}")
        return str(output_path)
        
    def create_f1_optimization_curve(self, opt_data: Dict, output_path: str, eval_id: str) -> str:
        """
        F1最適化曲線の生成
        
        Args:
            opt_data: 閾値最適化結果辞書
            output_path: 保存先パス  
            eval_id: 評価ID
            
        Returns:
            保存されたファイルパス
        """
        self.logger.info("Creating F1 optimization curve...")
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # データ取得
        thresholds = np.array(opt_data['all_thresholds'])
        f1_scores = np.array(opt_data['all_f1_scores'])
        precision_scores = np.array(opt_data['all_precision_scores'])
        recall_scores = np.array(opt_data['all_recall_scores'])
        
        optimal_threshold = opt_data['optimal_threshold']
        optimal_f1 = opt_data['optimal_f1_score']
        
        # 各メトリクスの曲線
        ax.plot(thresholds, f1_scores, color=self.colors['primary'], linewidth=3,
               label='F1スコア', marker='o', markersize=2, alpha=0.8)
        ax.plot(thresholds, precision_scores, color=self.colors['warning'], linewidth=2,
               linestyle='--', alpha=0.7, label='Precision')
        ax.plot(thresholds, recall_scores, color=self.colors['success'], linewidth=2,
               linestyle='--', alpha=0.7, label='Recall')
        
        # 最適点のマーキング
        ax.axvline(optimal_threshold, color=self.colors['optimal'], linestyle=':',
                  linewidth=2, alpha=0.8, label=f'最適閾値 ({optimal_threshold:.3f})')
        
        ax.scatter([optimal_threshold], [optimal_f1], color=self.colors['optimal'],
                  s=200, zorder=5, edgecolors='white', linewidth=3,
                  label=f'最適F1 ({optimal_f1:.3f})')
        
        # 標準閾値での値
        default_idx = np.argmin(np.abs(thresholds - 0.5))
        default_f1 = f1_scores[default_idx] if default_idx < len(f1_scores) else 0
        
        ax.axvline(0.5, color=self.colors['baseline'], linestyle=':', 
                  linewidth=2, alpha=0.6, label=f'標準閾値 (F1={default_f1:.3f})')
        
        # 装飾
        ax.grid(True, **self.grid_style)
        ax.set_xlabel('分類閾値', fontsize=12, fontweight='bold')
        ax.set_ylabel('スコア', fontsize=12, fontweight='bold')
        ax.set_title('閾値最適化分析 - F1スコア最大化', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        # 凡例
        ax.legend(loc='best', **self.legend_style)
        
        # 改善度情報
        improvement = opt_data.get('improvement_over_default', 0)
        improvement_text = f'改善度: {improvement:+.3f}\n({improvement*100:+.1f}%)'
        
        color = 'lightgreen' if improvement > 0 else 'lightcoral'
        ax.text(0.02, 0.02, improvement_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
                verticalalignment='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        output_path = self._add_eval_id_to_path(output_path, eval_id)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"F1 optimization curve saved to: {output_path}")
        return str(output_path)
        
    def create_confusion_matrix_comparison(self, comp_data: Dict, output_path: str, eval_id: str) -> str:
        """
        混同行列比較の可視化（標準閾値 vs 最適閾値）
        
        Args:
            comp_data: 閾値比較結果辞書
            output_path: 保存先パス
            eval_id: 評価ID
            
        Returns:
            保存されたファイルパス
        """
        self.logger.info("Creating confusion matrix comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # 標準閾値(0.5)と最適閾値のデータを取得
        results = comp_data['results']
        
        # 閾値0.5に最も近いデータ
        default_result = min(results, key=lambda x: abs(x['threshold'] - 0.5))
        
        # 最適閾値のデータ（リストの最後が最適値と仮定）
        optimal_result = results[-1] if results else default_result
        
        # 各混同行列の描画
        for idx, (result, title) in enumerate([
            (default_result, f'標準閾値 ({default_result["threshold"]:.3f})'),
            (optimal_result, f'最適閾値 ({optimal_result["threshold"]:.3f})')
        ]):
            cm = result['confusion_matrix']
            matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
            
            # ヒートマップ
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['予測: MISS', '予測: HIT'],
                       yticklabels=['実際: MISS', '実際: HIT'],
                       ax=axes[idx], cbar_kws={'label': '件数'})
            
            # パーセンテージを追加
            total = matrix.sum()
            for i in range(2):
                for j in range(2):
                    percentage = matrix[i, j] / total * 100
                    axes[idx].text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                                  ha='center', va='center', fontsize=9, color='gray')
            
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            
            # 性能指標を下部に追加
            metrics_text = (f'Accuracy: {result["accuracy"]:.3f}  '
                          f'Precision: {result["precision"]:.3f}  '
                          f'Recall: {result["recall"]:.3f}  '
                          f'F1: {result["f1_score"]:.3f}')
            
            axes[idx].text(0.5, -0.15, metrics_text, transform=axes[idx].transAxes,
                          ha='center', va='top', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('混同行列比較 - 閾値変更による影響分析', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存
        output_path = self._add_eval_id_to_path(output_path, eval_id)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix comparison saved to: {output_path}")
        return str(output_path)
        
    def create_performance_dashboard(self, metrics: Dict, output_path: str, eval_id: str) -> str:
        """
        性能ダッシュボードの生成（主要指標一覧）
        
        Args:
            metrics: 包括的評価メトリクス
            output_path: 保存先パス
            eval_id: 評価ID
            
        Returns:
            保存されたファイルパス
        """
        self.logger.info("Creating performance dashboard...")
        
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # メトリクス抽出
        roc_data = metrics.get('evaluation_results', {}).get('roc_analysis', {})
        opt_data = metrics.get('evaluation_results', {}).get('threshold_optimization', {})
        business_data = metrics.get('evaluation_results', {}).get('business_metrics', {})
        meta_data = metrics.get('metadata', {})
        
        # 1. AUC スコアカード
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_score_card(ax1, 'ROC-AUC', roc_data.get('auc', 0), 'AUC', 
                               threshold_good=0.9, threshold_ok=0.8)
        
        # 2. F1 スコアカード
        ax2 = fig.add_subplot(gs[0, 1])  
        optimal_f1 = opt_data.get('optimal_f1_score', 0)
        self._create_score_card(ax2, 'F1スコア', optimal_f1, 'F1', 
                               threshold_good=0.85, threshold_ok=0.75)
        
        # 3. 見逃し率カード
        ax3 = fig.add_subplot(gs[0, 2])
        optimal_metrics = opt_data.get('optimal_metrics', {})
        fnr = optimal_metrics.get('fn', 0) / (optimal_metrics.get('fn', 0) + optimal_metrics.get('tp', 1))
        self._create_score_card(ax3, '見逃し率', fnr, 'FNR', 
                               threshold_good=0.1, threshold_ok=0.2, lower_is_better=True)
        
        # 4. 総合判定カード
        ax4 = fig.add_subplot(gs[0, 3])
        overall_score = (roc_data.get('auc', 0) + optimal_f1 + (1-fnr)) / 3
        self._create_score_card(ax4, '総合評価', overall_score, '総合',
                               threshold_good=0.8, threshold_ok=0.7)
        
        # 5-8. 詳細指標（小さなグラフ）
        detail_metrics = [
            ('Precision', optimal_metrics.get('precision', 0)),
            ('Recall', optimal_metrics.get('recall', 0)),
            ('Specificity', optimal_metrics.get('tn', 0) / (optimal_metrics.get('tn', 0) + optimal_metrics.get('fp', 1))),
            ('Accuracy', optimal_metrics.get('accuracy', 0))
        ]
        
        for i, (name, value) in enumerate(detail_metrics):
            ax = fig.add_subplot(gs[1, i])
            self._create_mini_gauge(ax, name, value)
        
        # 9. コスト分析（大きなグラフ）
        ax9 = fig.add_subplot(gs[2, :2])
        if business_data:
            self._create_cost_analysis(ax9, business_data)
        
        # 10. データセット情報
        ax10 = fig.add_subplot(gs[2, 2:])
        self._create_dataset_info(ax10, meta_data)
        
        plt.suptitle(f'Patent-Radar 性能ダッシュボード - {eval_id}', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 保存
        output_path = self._add_eval_id_to_path(output_path, eval_id)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance dashboard saved to: {output_path}")
        return str(output_path)
        
    def _create_score_card(self, ax, title: str, score: float, unit: str,
                          threshold_good: float, threshold_ok: float, 
                          lower_is_better: bool = False):
        """スコアカード作成（ヘルパー関数）"""
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 色決定
        if lower_is_better:
            if score <= threshold_good:
                color = self.colors['success']
                status = '良好'
            elif score <= threshold_ok:
                color = self.colors['warning'] 
                status = '注意'
            else:
                color = self.colors['warning']
                status = '要改善'
        else:
            if score >= threshold_good:
                color = self.colors['success']
                status = '良好'
            elif score >= threshold_ok:
                color = self.colors['warning']
                status = '注意' 
            else:
                color = self.colors['warning']
                status = '要改善'
        
        # カード背景
        rect = patches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8, 
                                     boxstyle="round,pad=0.02",
                                     facecolor=color, alpha=0.2,
                                     edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # スコア表示
        ax.text(0.5, 0.65, f'{score:.3f}', ha='center', va='center',
               fontsize=20, fontweight='bold', color=color)
        
        # タイトル・単位
        ax.text(0.5, 0.85, title, ha='center', va='center',
               fontsize=12, fontweight='bold')
        ax.text(0.5, 0.45, unit, ha='center', va='center', 
               fontsize=8, style='italic')
        
        # ステータス
        ax.text(0.5, 0.25, status, ha='center', va='center',
               fontsize=10, fontweight='bold', color=color)
        
    def _create_mini_gauge(self, ax, title: str, value: float):
        """ミニゲージ作成（ヘルパー関数）"""
        ax.clear()
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        
        # 半円ゲージ
        theta = np.linspace(np.pi, 0, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        
        # ベースライン
        ax.plot(x, y, color='lightgray', linewidth=5)
        
        # 値の表示（半円上に点）
        value_theta = np.pi * (1 - value)
        value_x = np.cos(value_theta)
        value_y = np.sin(value_theta)
        
        ax.scatter(value_x, value_y, s=100, color=self.colors['primary'], zorder=5)
        
        # タイトル・数値
        ax.text(0, -0.05, title, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(0, 0.5, f'{value:.3f}', ha='center', va='center', 
               fontsize=12, fontweight='bold', color=self.colors['primary'])
        
    def _create_cost_analysis(self, ax, business_data: Dict):
        """コスト分析グラフ作成（ヘルパー関数）"""
        default_data = business_data.get('default', {})
        optimal_data = business_data.get('optimal', {})
        
        categories = ['機会損失', '工数コスト', '総コスト']
        default_costs = [default_data.get('opportunity_cost', 0), 
                        default_data.get('labor_cost', 0),
                        default_data.get('total_cost', 0)]
        optimal_costs = [optimal_data.get('opportunity_cost', 0),
                        optimal_data.get('labor_cost', 0), 
                        optimal_data.get('total_cost', 0)]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [c/10000 for c in default_costs], width,
                      label='標準閾値', color=self.colors['baseline'], alpha=0.7)
        bars2 = ax.bar(x + width/2, [c/10000 for c in optimal_costs], width,
                      label='最適閾値', color=self.colors['optimal'], alpha=0.7)
        
        ax.set_xlabel('コスト項目')
        ax.set_ylabel('コスト (万円)')
        ax.set_title('コスト比較分析', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _create_dataset_info(self, ax, meta_data: Dict):
        """データセット情報表示（ヘルパー関数）"""
        ax.axis('off')
        
        info_text = f"""
        データセット情報:
        
        総サンプル数: {meta_data.get('sample_count', 0):,} 件
        正例数: {meta_data.get('positive_class_count', 0):,} 件
        負例数: {meta_data.get('negative_class_count', 0):,} 件
        
        実行設定:
        閾値解像度: {meta_data.get('threshold_resolution', 1000):,} 分割
        乱数シード: {meta_data.get('random_seed', 42)}
        
        実行日時:
        {meta_data.get('evaluation_timestamp', 'Unknown')[:19]}
        """
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
    def _interpret_auc(self, auc_score: float) -> str:
        """AUC スコアの解釈文字列生成"""
        if auc_score >= 0.95:
            return "優秀"
        elif auc_score >= 0.90:
            return "良好"
        elif auc_score >= 0.80:
            return "改善要"
        elif auc_score >= 0.70:
            return "要再設計"
        else:
            return "実用困難"
            
    def _add_eval_id_to_path(self, path: str, eval_id: str) -> Path:
        """ファイルパスに評価IDを追加（ファイル名安全化対応）"""
        path_obj = Path(path)
        if eval_id and eval_id not in str(path_obj.name):
            # eval_idをファイル名安全な形式に変換
            safe_eval_id = re.sub(r'[^\w\-_]', '_', eval_id)  # 英数字、ハイフン、アンダースコア以外を_に置換
            
            stem = path_obj.stem
            suffix = path_obj.suffix if path_obj.suffix else '.png'  # デフォルトでPNG形式
            filename = f"{stem}_{safe_eval_id}{suffix}"
            path_obj = path_obj.parent / filename
            
        # ディレクトリ作成（親ディレクトリも含む）
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        return path_obj


def main():
    """テスト実行用メイン関数"""
    print("=== Patent Visualization Engine Test ===")
    
    # テストデータ生成
    np.random.seed(42)
    
    # ROCデータのシミュレート
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - (1-fpr)**2  # 凸な曲線
    tpr += np.random.normal(0, 0.02, len(tpr))  # ノイズ追加
    tpr = np.clip(tpr, 0, 1)
    
    roc_data = {
        'auc': 0.85,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'optimal_point': {'fpr': 0.2, 'tpr': 0.8, 'threshold': 0.6}
    }
    
    # 分布データのシミュレート
    dist_data = {
        'positive_class': {
            'count': 250,
            'mean': 0.7,
            'std': 0.2
        },
        'negative_class': {
            'count': 750, 
            'mean': 0.3,
            'std': 0.2
        },
        'histogram_data': {
            'bins': np.linspace(0, 1, 21).tolist(),
            'positive_hist': np.random.poisson(10, 20).tolist(),
            'negative_hist': np.random.poisson(15, 20).tolist()
        },
        'separation_metrics': {
            'cohens_d': 1.5,
            'overlap_ratio': 0.3
        }
    }
    
    # 最適化データのシミュレート
    thresholds = np.linspace(0, 1, 101)
    f1_scores = -(thresholds - 0.6)**2 + 0.8  # 0.6で最大となる二次関数
    f1_scores = np.clip(f1_scores, 0, 1)
    
    opt_data = {
        'optimal_threshold': 0.6,
        'optimal_f1_score': 0.8,
        'all_thresholds': thresholds.tolist(),
        'all_f1_scores': f1_scores.tolist(),
        'all_precision_scores': (f1_scores * 0.9).tolist(),
        'all_recall_scores': (f1_scores * 1.1).tolist(),
        'improvement_over_default': 0.05
    }
    
    try:
        # 可視化エンジン初期化
        engine = PatentVisualizationEngine()
        
        # 出力ディレクトリ作成
        output_dir = Path("analysis/temp/visualization_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        eval_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n1. Creating ROC curve...")
        roc_path = engine.create_roc_curve(roc_data, str(output_dir / "roc_test.png"), eval_id)
        
        print(f"2. Creating prediction distribution...")
        dist_path = engine.create_prediction_distribution(dist_data, str(output_dir / "dist_test.png"), eval_id, 0.6)
        
        print(f"3. Creating F1 optimization curve...")
        f1_path = engine.create_f1_optimization_curve(opt_data, str(output_dir / "f1_test.png"), eval_id)
        
        print(f"\n4. Generated files:")
        print(f"  ROC curve: {roc_path}")
        print(f"  Distribution: {dist_path}")
        print(f"  F1 optimization: {f1_path}")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("\n=== Test completed successfully ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())