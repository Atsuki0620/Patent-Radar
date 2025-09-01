#!/usr/bin/env python3
"""
汎用バイナリ分類器評価システム - HTML統合レポート構築
Generic Binary Classifier Evaluation System - HTML Report Builder
全評価結果を統合したHTMLレポートを生成
"""

import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging


class PatentReportBuilder:
    """汎用バイナリ分類器 HTML統合レポート構築クラス"""
    
    def __init__(self, model_info: Optional[Dict] = None):
        """
        初期化
        
        Args:
            model_info: モデル情報辞書 (name, description, positive_class_name, negative_class_name)
        """
        self.model_info = model_info or {
            'name': 'Binary Classifier',
            'description': 'Binary Classification Model',
            'positive_class_name': 'POSITIVE',
            'negative_class_name': 'NEGATIVE'
        }
        self.logger = logging.getLogger(__name__)
    
    def build_comprehensive_report(
        self,
        comprehensive_metrics: Dict,
        narratives: Dict,
        visualizations: List[str],
        eval_dir: Path,
        eval_id: str,
        model_info: Optional[Dict] = None
    ) -> Path:
        """
        包括的HTMLレポート生成
        
        Args:
            comprehensive_metrics: 包括的評価メトリクス
            narratives: 解説テキスト辞書
            visualizations: 可視化ファイルパス一覧
            eval_dir: 評価ディレクトリ
            eval_id: 評価ID
            model_info: モデル情報（上書き用）
            
        Returns:
            生成されたHTMLファイルのパス
        """
        if model_info:
            self.model_info.update(model_info)
            
        eval_dir = Path(eval_dir)
        report_path = eval_dir / "report.html"
        try:
            # HTMLコンテンツ構築
            html_content = self._build_html_structure(
                eval_id, comprehensive_metrics, narratives, visualizations
            )
            
            # HTMLファイル保存
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Comprehensive report generated: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise
    
    def _build_html_structure(
        self, 
        eval_id: str, 
        comprehensive_metrics: Dict, 
        narratives: Dict, 
        visualizations: List[str]
    ) -> str:
        """
        HTML構造構築
        
        Args:
            eval_id: 評価ID
            comprehensive_metrics: 包括的評価メトリクス
            narratives: 解説テキスト
            visualizations: 可視化ファイル一覧
            
        Returns:
            完全なHTMLコンテンツ
        """
        # CSS・JavaScript組み込み
        css_styles = self._generate_css_styles()
        js_scripts = self._generate_javascript()
        
        # メタデータ取得
        metadata = self._extract_metadata(eval_id, comprehensive_metrics)
        
        # セクション構築
        sections = {
            "executive_summary": self._build_executive_summary_section(
                comprehensive_metrics, narratives.get("executive_summary", "")
            ),
            "data_overview": self._build_data_overview_section(
                evaluation_results, narratives.get("data_overview", ""), 
                visualization_files.get("data_overview")
            ),
            "roc_analysis": self._build_roc_analysis_section(
                evaluation_results, narratives.get("roc_analysis", ""),
                visualization_files.get("roc_curve")
            ),
            "threshold_optimization": self._build_threshold_optimization_section(
                evaluation_results, narratives.get("threshold_optimization", ""),
                visualization_files.get("f1_optimization")
            ),
            "confusion_matrix": self._build_confusion_matrix_section(
                evaluation_results, narratives.get("confusion_matrix", ""),
                visualization_files.get("confusion_matrix")
            ),
            "distribution_analysis": self._build_distribution_analysis_section(
                evaluation_results, narratives.get("distribution_analysis", ""),
                visualization_files.get("prediction_distribution")
            ),
            "performance_dashboard": self._build_performance_dashboard_section(
                visualization_files.get("performance_dashboard")
            ),
            "detailed_metrics": self._build_detailed_metrics_section(evaluation_results)
        }
        
        # 完全HTML構築
        html_template = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.model_info['name']} 評価レポート - {eval_id}</title>
            {css_styles}
            {js_scripts}
        </head>
        <body>
            <div class="container">
                {self._build_header_section(metadata)}
                {self._build_navigation_menu()}
                {self._build_table_of_contents(sections)}
                
                <main class="main-content">
                    {sections["executive_summary"]}
                    {sections["data_overview"]}
                    {sections["roc_analysis"]}
                    {sections["threshold_optimization"]}
                    {sections["confusion_matrix"]}
                    {sections["distribution_analysis"]}
                    {sections["performance_dashboard"]}
                    {sections["detailed_metrics"]}
                </main>
                
                {self._build_footer_section(eval_id)}
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_css_styles(self) -> str:
        """CSS スタイル生成"""
        return """
        <style>
            /* リセット・基本スタイル */
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f5f5f5;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                min-height: 100vh;
            }
            
            /* ヘッダー */
            .report-header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 2rem;
                text-align: center;
            }
            
            .report-title {
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                font-weight: 300;
            }
            
            .report-subtitle {
                font-size: 1.2rem;
                opacity: 0.9;
                margin-bottom: 1rem;
            }
            
            .metadata-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-top: 1.5rem;
            }
            
            .metadata-item {
                background: rgba(255,255,255,0.1);
                padding: 1rem;
                border-radius: 8px;
                text-align: left;
            }
            
            .metadata-label {
                font-size: 0.9rem;
                opacity: 0.8;
                margin-bottom: 0.3rem;
            }
            
            .metadata-value {
                font-size: 1.1rem;
                font-weight: bold;
            }
            
            /* ナビゲーション */
            .navigation {
                background: #2c3e50;
                padding: 0;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            
            .nav-list {
                list-style: none;
                display: flex;
                flex-wrap: wrap;
            }
            
            .nav-item a {
                display: block;
                color: white;
                text-decoration: none;
                padding: 1rem 1.5rem;
                transition: background-color 0.3s;
            }
            
            .nav-item a:hover {
                background-color: #34495e;
            }
            
            /* 目次 */
            .table-of-contents {
                background: #ecf0f1;
                padding: 2rem;
                margin: 2rem 0;
            }
            
            .toc-title {
                font-size: 1.5rem;
                margin-bottom: 1rem;
                color: #2c3e50;
            }
            
            .toc-list {
                list-style: none;
            }
            
            .toc-item {
                margin-bottom: 0.8rem;
            }
            
            .toc-item a {
                color: #3498db;
                text-decoration: none;
                font-size: 1.1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .toc-item a:hover {
                color: #2980b9;
                text-decoration: underline;
            }
            
            .section-status {
                font-size: 0.9rem;
                padding: 0.2rem 0.8rem;
                border-radius: 12px;
                background: #27ae60;
                color: white;
            }
            
            /* メインコンテンツ */
            .main-content {
                padding: 2rem;
            }
            
            .section {
                margin-bottom: 4rem;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 2rem;
            }
            
            .section:last-child {
                border-bottom: none;
            }
            
            .section-header {
                display: flex;
                align-items: center;
                margin-bottom: 2rem;
                padding-bottom: 1rem;
                border-bottom: 3px solid #3498db;
            }
            
            .section-number {
                background: #3498db;
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 1.2rem;
                margin-right: 1rem;
            }
            
            .section-title {
                font-size: 2rem;
                color: #2c3e50;
                font-weight: 300;
            }
            
            /* 内容ブロック */
            .content-block {
                background: white;
                border: 1px solid #e1e8ed;
                border-radius: 8px;
                overflow: hidden;
                margin-bottom: 2rem;
            }
            
            .block-header {
                background: #f8f9fa;
                padding: 1rem 1.5rem;
                border-bottom: 1px solid #e1e8ed;
                font-weight: bold;
                color: #495057;
            }
            
            .block-content {
                padding: 1.5rem;
            }
            
            /* 可視化コンテナ */
            .visualization-container {
                text-align: center;
                margin: 2rem 0;
                padding: 1rem;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .visualization-image {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            }
            
            .visualization-caption {
                margin-top: 1rem;
                font-style: italic;
                color: #6c757d;
                font-size: 0.9rem;
            }
            
            /* メトリクステーブル */
            .metrics-table {
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
                background: white;
            }
            
            .metrics-table th,
            .metrics-table td {
                padding: 1rem;
                text-align: left;
                border-bottom: 1px solid #dee2e6;
            }
            
            .metrics-table th {
                background: #f8f9fa;
                font-weight: bold;
                color: #495057;
            }
            
            .metrics-table tbody tr:hover {
                background: #f8f9fa;
            }
            
            /* 統計カード */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }
            
            .stat-card {
                background: white;
                border-radius: 8px;
                padding: 1.5rem;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: transform 0.3s;
            }
            
            .stat-card:hover {
                transform: translateY(-2px);
            }
            
            .stat-value {
                font-size: 2.5rem;
                font-weight: bold;
                color: #3498db;
                margin-bottom: 0.5rem;
            }
            
            .stat-label {
                color: #6c757d;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .stat-description {
                margin-top: 0.5rem;
                font-size: 0.85rem;
                color: #495057;
            }
            
            /* アラート・通知 */
            .alert {
                padding: 1rem 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
                border-left: 4px solid;
            }
            
            .alert-success {
                background: #d4edda;
                border-left-color: #28a745;
                color: #155724;
            }
            
            .alert-warning {
                background: #fff3cd;
                border-left-color: #ffc107;
                color: #856404;
            }
            
            .alert-danger {
                background: #f8d7da;
                border-left-color: #dc3545;
                color: #721c24;
            }
            
            .alert-info {
                background: #d1ecf1;
                border-left-color: #17a2b8;
                color: #0c5460;
            }
            
            /* フッター */
            .report-footer {
                background: #2c3e50;
                color: white;
                padding: 2rem;
                text-align: center;
                margin-top: 4rem;
            }
            
            .footer-content {
                max-width: 800px;
                margin: 0 auto;
            }
            
            .generation-info {
                margin-bottom: 1rem;
                opacity: 0.8;
            }
            
            .footer-links {
                list-style: none;
                display: flex;
                justify-content: center;
                gap: 2rem;
                margin-top: 1rem;
            }
            
            .footer-links a {
                color: #ecf0f1;
                text-decoration: none;
                transition: color 0.3s;
            }
            
            .footer-links a:hover {
                color: #3498db;
            }
            
            /* レスポンシブ */
            @media (max-width: 768px) {
                .container {
                    margin: 0;
                }
                
                .main-content {
                    padding: 1rem;
                }
                
                .report-title {
                    font-size: 2rem;
                }
                
                .metadata-grid {
                    grid-template-columns: 1fr;
                }
                
                .nav-list {
                    flex-direction: column;
                }
                
                .stats-grid {
                    grid-template-columns: 1fr;
                }
                
                .section-header {
                    flex-direction: column;
                    text-align: center;
                }
            }
            
            /* プリント用スタイル */
            @media print {
                .navigation,
                .table-of-contents {
                    display: none;
                }
                
                .section {
                    page-break-after: auto;
                }
                
                .visualization-image {
                    max-height: 400px;
                }
            }
        </style>
        """
    
    def _generate_javascript(self) -> str:
        """JavaScript生成"""
        return """
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // スムーズスクロール
                const navLinks = document.querySelectorAll('.nav-item a, .toc-item a');
                navLinks.forEach(link => {
                    link.addEventListener('click', function(e) {
                        const href = this.getAttribute('href');
                        if (href.startsWith('#')) {
                            e.preventDefault();
                            const target = document.querySelector(href);
                            if (target) {
                                target.scrollIntoView({
                                    behavior: 'smooth',
                                    block: 'start'
                                });
                            }
                        }
                    });
                });
                
                // セクション表示アニメーション
                const sections = document.querySelectorAll('.section');
                const observerOptions = {
                    threshold: 0.1,
                    rootMargin: '0px 0px -50px 0px'
                };
                
                const observer = new IntersectionObserver(function(entries) {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            entry.target.style.opacity = '1';
                            entry.target.style.transform = 'translateY(0)';
                        }
                    });
                }, observerOptions);
                
                sections.forEach(section => {
                    section.style.opacity = '0';
                    section.style.transform = 'translateY(20px)';
                    section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
                    observer.observe(section);
                });
                
                // 画像読み込みエラーハンドリング
                const images = document.querySelectorAll('.visualization-image');
                images.forEach(img => {
                    img.addEventListener('error', function() {
                        this.style.display = 'none';
                        const container = this.closest('.visualization-container');
                        if (container) {
                            const errorMsg = document.createElement('div');
                            errorMsg.className = 'alert alert-warning';
                            errorMsg.textContent = '可視化の読み込みに失敗しました';
                            container.appendChild(errorMsg);
                        }
                    });
                });
                
                // 数値のアニメーション
                const animateNumbers = function() {
                    const numberElements = document.querySelectorAll('.stat-value');
                    numberElements.forEach(element => {
                        const target = parseFloat(element.textContent);
                        if (!isNaN(target)) {
                            let current = 0;
                            const increment = target / 30;
                            const timer = setInterval(() => {
                                current += increment;
                                if (current >= target) {
                                    current = target;
                                    clearInterval(timer);
                                }
                                element.textContent = current.toFixed(3);
                            }, 50);
                        }
                    });
                };
                
                // 初期読み込み完了後に数値アニメーション実行
                setTimeout(animateNumbers, 500);
            });
            
            // ユーティリティ関数
            function toggleSection(sectionId) {
                const section = document.getElementById(sectionId);
                if (section) {
                    const content = section.querySelector('.block-content');
                    const isVisible = content.style.display !== 'none';
                    content.style.display = isVisible ? 'none' : 'block';
                }
            }
            
            function exportSection(sectionId) {
                const section = document.getElementById(sectionId);
                if (section) {
                    const printWindow = window.open('', '_blank');
                    printWindow.document.write(`
                        <html>
                            <head><title>Export - ${sectionId}</title></head>
                            <body>${section.outerHTML}</body>
                        </html>
                    `);
                    printWindow.document.close();
                    printWindow.print();
                }
            }
        </script>
        """
    
    def _extract_metadata(self, eval_id: str, evaluation_results: Dict) -> Dict:
        """メタデータ抽出"""
        data_stats = evaluation_results.get("data_statistics", {})
        optimal_metrics = evaluation_results.get("optimal_metrics", {})
        roc_data = evaluation_results.get("roc_analysis", {})
        
        return {
            "eval_id": eval_id,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": data_stats.get("total_samples", 0),
            "accuracy": optimal_metrics.get("accuracy", 0),
            "f1_score": optimal_metrics.get("f1_score", 0), 
            "auc_score": roc_data.get("auc", 0),
            "system_version": f"{self.model_info['name']} v1.0"
        }
    
    def _build_header_section(self, metadata: Dict) -> str:
        """ヘッダーセクション構築"""
        return f"""
        <header class="report-header">
            <h1 class="report-title">{self.model_info['name']} 評価レポート</h1>
            <p class="report-subtitle">{self.model_info['description']} - Performance Analysis</p>
            
            <div class="metadata-grid">
                <div class="metadata-item">
                    <div class="metadata-label">評価ID</div>
                    <div class="metadata-value">{metadata['eval_id']}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">生成日時</div>
                    <div class="metadata-value">{metadata['generation_time']}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">データ件数</div>
                    <div class="metadata-value">{metadata['total_samples']:,} 件</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Accuracy</div>
                    <div class="metadata-value">{metadata['accuracy']:.3f}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">F1-Score</div>
                    <div class="metadata-value">{metadata['f1_score']:.3f}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">AUC</div>
                    <div class="metadata-value">{metadata['auc_score']:.3f}</div>
                </div>
            </div>
        </header>
        """
    
    def _build_navigation_menu(self) -> str:
        """ナビゲーションメニュー構築"""
        return """
        <nav class="navigation">
            <ul class="nav-list">
                <li class="nav-item"><a href="#executive-summary">要約</a></li>
                <li class="nav-item"><a href="#data-overview">データ概要</a></li>
                <li class="nav-item"><a href="#roc-analysis">ROC分析</a></li>
                <li class="nav-item"><a href="#threshold-optimization">閾値最適化</a></li>
                <li class="nav-item"><a href="#confusion-matrix">混同行列</a></li>
                <li class="nav-item"><a href="#distribution-analysis">分布分析</a></li>
                <li class="nav-item"><a href="#performance-dashboard">性能ダッシュボード</a></li>
                <li class="nav-item"><a href="#detailed-metrics">詳細指標</a></li>
            </ul>
        </nav>
        """
    
    def _build_table_of_contents(self, sections: Dict) -> str:
        """目次構築"""
        return """
        <section class="table-of-contents">
            <h2 class="toc-title">目次</h2>
            <ul class="toc-list">
                <li class="toc-item">
                    <a href="#executive-summary">
                        <span>1. エグゼクティブサマリー</span>
                        <span class="section-status">完了</span>
                    </a>
                </li>
                <li class="toc-item">
                    <a href="#data-overview">
                        <span>2. データ概要・品質評価</span>
                        <span class="section-status">完了</span>
                    </a>
                </li>
                <li class="toc-item">
                    <a href="#roc-analysis">
                        <span>3. ROC曲線・AUC分析</span>
                        <span class="section-status">完了</span>
                    </a>
                </li>
                <li class="toc-item">
                    <a href="#threshold-optimization">
                        <span>4. 閾値最適化・F1最大化</span>
                        <span class="section-status">完了</span>
                    </a>
                </li>
                <li class="toc-item">
                    <a href="#confusion-matrix">
                        <span>5. 混同行列・エラー分析</span>
                        <span class="section-status">完了</span>
                    </a>
                </li>
                <li class="toc-item">
                    <a href="#distribution-analysis">
                        <span>6. 予測分布・キャリブレーション</span>
                        <span class="section-status">完了</span>
                    </a>
                </li>
                <li class="toc-item">
                    <a href="#performance-dashboard">
                        <span>7. 総合性能ダッシュボード</span>
                        <span class="section-status">完了</span>
                    </a>
                </li>
                <li class="toc-item">
                    <a href="#detailed-metrics">
                        <span>8. 詳細メトリクス・技術仕様</span>
                        <span class="section-status">完了</span>
                    </a>
                </li>
            </ul>
        </section>
        """
    
    def _build_executive_summary_section(self, evaluation_results: Dict, narrative: str) -> str:
        """エグゼクティブサマリーセクション構築"""
        return f"""
        <section id="executive-summary" class="section">
            <div class="section-header">
                <div class="section-number">1</div>
                <h2 class="section-title">エグゼクティブサマリー</h2>
            </div>
            
            <div class="content-block">
                <div class="block-header">総合評価・推奨事項</div>
                <div class="block-content">
                    {narrative}
                </div>
            </div>
        </section>
        """
    
    def _build_data_overview_section(self, evaluation_results: Dict, narrative: str, viz_file: Optional[str]) -> str:
        """データ概要セクション構築"""
        viz_html = ""
        if viz_file and Path(viz_file).exists():
            viz_html = f"""
            <div class="visualization-container">
                <img src="{self._image_to_base64(viz_file)}" alt="Data Overview" class="visualization-image">
                <div class="visualization-caption">データセット構成と品質指標</div>
            </div>
            """
        
        return f"""
        <section id="data-overview" class="section">
            <div class="section-header">
                <div class="section-number">2</div>
                <h2 class="section-title">データ概要・品質評価</h2>
            </div>
            
            <div class="content-block">
                <div class="block-header">データセット分析</div>
                <div class="block-content">
                    {narrative}
                    {viz_html}
                </div>
            </div>
        </section>
        """
    
    def _build_roc_analysis_section(self, evaluation_results: Dict, narrative: str, viz_file: Optional[str]) -> str:
        """ROC分析セクション構築"""
        viz_html = ""
        if viz_file and Path(viz_file).exists():
            viz_html = f"""
            <div class="visualization-container">
                <img src="{self._image_to_base64(viz_file)}" alt="ROC Curve" class="visualization-image">
                <div class="visualization-caption">ROC曲線とAUC性能評価</div>
            </div>
            """
        
        return f"""
        <section id="roc-analysis" class="section">
            <div class="section-header">
                <div class="section-number">3</div>
                <h2 class="section-title">ROC曲線・AUC分析</h2>
            </div>
            
            <div class="content-block">
                <div class="block-header">識別能力評価</div>
                <div class="block-content">
                    {narrative}
                    {viz_html}
                </div>
            </div>
        </section>
        """
    
    def _build_threshold_optimization_section(self, evaluation_results: Dict, narrative: str, viz_file: Optional[str]) -> str:
        """閾値最適化セクション構築"""
        viz_html = ""
        if viz_file and Path(viz_file).exists():
            viz_html = f"""
            <div class="visualization-container">
                <img src="{self._image_to_base64(viz_file)}" alt="F1 Optimization" class="visualization-image">
                <div class="visualization-caption">F1スコア最適化による閾値決定</div>
            </div>
            """
        
        return f"""
        <section id="threshold-optimization" class="section">
            <div class="section-header">
                <div class="section-number">4</div>
                <h2 class="section-title">閾値最適化・F1最大化</h2>
            </div>
            
            <div class="content-block">
                <div class="block-header">最適閾値決定プロセス</div>
                <div class="block-content">
                    {narrative}
                    {viz_html}
                </div>
            </div>
        </section>
        """
    
    def _build_confusion_matrix_section(self, evaluation_results: Dict, narrative: str, viz_file: Optional[str]) -> str:
        """混同行列セクション構築"""
        viz_html = ""
        if viz_file and Path(viz_file).exists():
            viz_html = f"""
            <div class="visualization-container">
                <img src="{self._image_to_base64(viz_file)}" alt="Confusion Matrix" class="visualization-image">
                <div class="visualization-caption">混同行列とエラー分析</div>
            </div>
            """
        
        return f"""
        <section id="confusion-matrix" class="section">
            <div class="section-header">
                <div class="section-number">5</div>
                <h2 class="section-title">混同行列・エラー分析</h2>
            </div>
            
            <div class="content-block">
                <div class="block-header">分類結果詳細分析</div>
                <div class="block-content">
                    {narrative}
                    {viz_html}
                </div>
            </div>
        </section>
        """
    
    def _build_distribution_analysis_section(self, evaluation_results: Dict, narrative: str, viz_file: Optional[str]) -> str:
        """分布分析セクション構築"""
        viz_html = ""
        if viz_file and Path(viz_file).exists():
            viz_html = f"""
            <div class="visualization-container">
                <img src="{self._image_to_base64(viz_file)}" alt="Prediction Distribution" class="visualization-image">
                <div class="visualization-caption">予測確率分布とキャリブレーション評価</div>
            </div>
            """
        
        return f"""
        <section id="distribution-analysis" class="section">
            <div class="section-header">
                <div class="section-number">6</div>
                <h2 class="section-title">予測分布・キャリブレーション</h2>
            </div>
            
            <div class="content-block">
                <div class="block-header">予測確率分析</div>
                <div class="block-content">
                    {narrative}
                    {viz_html}
                </div>
            </div>
        </section>
        """
    
    def _build_performance_dashboard_section(self, viz_file: Optional[str]) -> str:
        """性能ダッシュボードセクション構築"""
        viz_html = ""
        if viz_file and Path(viz_file).exists():
            viz_html = f"""
            <div class="visualization-container">
                <img src="{self._image_to_base64(viz_file)}" alt="Performance Dashboard" class="visualization-image">
                <div class="visualization-caption">総合性能ダッシュボード</div>
            </div>
            """
        
        return f"""
        <section id="performance-dashboard" class="section">
            <div class="section-header">
                <div class="section-number">7</div>
                <h2 class="section-title">総合性能ダッシュボード</h2>
            </div>
            
            <div class="content-block">
                <div class="block-header">統合性能ビュー</div>
                <div class="block-content">
                    <div class="content-block">
                        <div class="block-header">目的</div>
                        <div class="block-content">
                            <p>全評価指標を統合したダッシュボード形式で、システムの総合性能を一覧し、
                            ビジネス判断に必要な情報を集約表示します。</p>
                        </div>
                    </div>
                    
                    {viz_html}
                    
                    <div class="content-block">
                        <div class="block-header">分析</div>
                        <div class="block-content">
                            <p>このダッシュボードにより、技術的指標とビジネス指標の双方から
                            システムの実用性を総合的に評価できます。各指標の相互関係を理解し、
                            改善ポイントの特定に活用してください。</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        """
    
    def _build_detailed_metrics_section(self, evaluation_results: Dict) -> str:
        """詳細メトリクスセクション構築"""
        optimal_metrics = evaluation_results.get("optimal_metrics", {})
        roc_data = evaluation_results.get("roc_analysis", {})
        threshold_data = evaluation_results.get("threshold_optimization", {})
        
        return f"""
        <section id="detailed-metrics" class="section">
            <div class="section-header">
                <div class="section-number">8</div>
                <h2 class="section-title">詳細メトリクス・技術仕様</h2>
            </div>
            
            <div class="content-block">
                <div class="block-header">完全メトリクス一覧</div>
                <div class="block-content">
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>指標</th>
                                <th>値</th>
                                <th>説明</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Accuracy</td>
                                <td>{optimal_metrics.get('accuracy', 0):.3f}</td>
                                <td>全体の正解率</td>
                            </tr>
                            <tr>
                                <td>Precision</td>
                                <td>{optimal_metrics.get('precision', 0):.3f}</td>
                                <td>HIT予測の精度</td>
                            </tr>
                            <tr>
                                <td>Recall</td>
                                <td>{optimal_metrics.get('recall', 0):.3f}</td>
                                <td>{self.model_info['positive_class_name']}の検出率</td>
                            </tr>
                            <tr>
                                <td>F1-Score</td>
                                <td>{optimal_metrics.get('f1_score', 0):.3f}</td>
                                <td>Precision・Recall調和平均</td>
                            </tr>
                            <tr>
                                <td>Specificity</td>
                                <td>{optimal_metrics.get('specificity', 0):.3f}</td>
                                <td>{self.model_info['negative_class_name']}の正識別率</td>
                            </tr>
                            <tr>
                                <td>AUC</td>
                                <td>{roc_data.get('auc', 0):.3f}</td>
                                <td>ROC曲線下面積</td>
                            </tr>
                            <tr>
                                <td>Optimal Threshold</td>
                                <td>{threshold_data.get('optimal_threshold', 0):.3f}</td>
                                <td>F1最大化閾値</td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <div class="content-block">
                        <div class="block-header">混同行列詳細</div>
                        <div class="block-content">
                            <div class="stats-grid">
                                <div class="stat-card">
                                    <div class="stat-value">{optimal_metrics.get('tp', 0)}</div>
                                    <div class="stat-label">True Positive</div>
                                    <div class="stat-description">正しくHITと判定</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-value">{optimal_metrics.get('fp', 0)}</div>
                                    <div class="stat-label">False Positive</div>
                                    <div class="stat-description">誤ってHITと判定</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-value">{optimal_metrics.get('tn', 0)}</div>
                                    <div class="stat-label">True Negative</div>
                                    <div class="stat-description">正しくMISSと判定</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-value">{optimal_metrics.get('fn', 0)}</div>
                                    <div class="stat-label">False Negative</div>
                                    <div class="stat-description">見逃されたHIT</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        """
    
    def _build_footer_section(self, eval_id: str) -> str:
        """フッターセクション構築"""
        return f"""
        <footer class="report-footer">
            <div class="footer-content">
                <div class="generation-info">
                    <p>このレポートは {self.model_info['name']} 評価システムにより自動生成されました</p>
                    <p>評価ID: {eval_id} | 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <ul class="footer-links">
                    <li><a href="#top">ページトップ</a></li>
                    <li><a href="#executive-summary">要約</a></li>
                    <li><a href="javascript:window.print()">印刷</a></li>
                </ul>
            </div>
        </footer>
        """
    
    def _image_to_base64(self, image_path: str) -> str:
        """画像をBase64エンコードしてdata URLに変換"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # ファイル拡張子から MIME type を決定
            path = Path(image_path)
            if path.suffix.lower() == '.png':
                mime_type = 'image/png'
            elif path.suffix.lower() in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            else:
                mime_type = 'image/png'  # デフォルト
            
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:{mime_type};base64,{base64_data}"
            
        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}: {e}")
            return ""


def main():
    """テスト実行用メイン関数"""
    print("=== Patent Evaluation Report Builder Test ===")
    
    # テストディレクトリ作成
    test_eval_dir = Path("test_evaluation")
    test_eval_dir.mkdir(exist_ok=True)
    (test_eval_dir / "visualizations").mkdir(exist_ok=True)
    (test_eval_dir / "data").mkdir(exist_ok=True)
    
    # サンプルデータ
    sample_evaluation_results = {
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
            "specificity": 0.8235,
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
    
    sample_narratives = {
        "executive_summary": "<p>テストサマリー</p>",
        "data_overview": "<p>テストデータ概要</p>",
        "roc_analysis": "<p>テストROC分析</p>",
        "threshold_optimization": "<p>テスト閾値最適化</p>",
        "confusion_matrix": "<p>テスト混同行列</p>",
        "distribution_analysis": "<p>テスト分布分析</p>"
    }
    
    sample_visualizations = {}  # 実際のテストでは画像ファイルパスを指定
    
    try:
        builder = PatentEvaluationReportBuilder(test_eval_dir)
        
        report_path = builder.generate_comprehensive_report(
            "PE_20250901_120000_TestModel",
            sample_evaluation_results,
            sample_narratives, 
            sample_visualizations
        )
        
        print(f"\n1. Report generated successfully: {report_path}")
        print(f"2. File size: {Path(report_path).stat().st_size:,} bytes")
        print("3. Report sections included:")
        print("   - Executive Summary")
        print("   - Data Overview") 
        print("   - ROC Analysis")
        print("   - Threshold Optimization")
        print("   - Confusion Matrix")
        print("   - Distribution Analysis")
        print("   - Performance Dashboard")
        print("   - Detailed Metrics")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Test completed ===")


if __name__ == "__main__":
    main()