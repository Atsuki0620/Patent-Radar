#!/usr/bin/env python3
"""
汎用バイナリ分類器評価システム メイン実行スクリプト
Generic Binary Classifier Evaluation System - Main Execution Script

使用例 / Usage Examples:
  # Patent-Radar評価（デフォルト設定）
  python run_generic_evaluation.py --gold-labels testing/data/labels.jsonl --predictions archive/outputs/goldset_results.jsonl

  # スパムフィルター評価
  python run_generic_evaluation.py --gold-labels spam_labels.jsonl --predictions spam_predictions.jsonl --config-name spam_filter

  # カスタムモデル評価
  python run_generic_evaluation.py --gold-labels my_labels.jsonl --predictions my_predictions.jsonl --model-name "MyClassifier" --positive-class "GOOD" --negative-class "BAD"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# 相対パスでモジュールをインポート
sys.path.append(str(Path(__file__).parent))

from evaluation_system.evaluation_master import PatentEvaluationMaster
from evaluation_system.data_processor import PatentDataProcessor  
from evaluation_system.strict_evaluator import StrictBinaryEvaluator
from evaluation_system.visualization_engine import PatentVisualizationEngine
from evaluation_system.narrative_generator import PatentEvaluationNarrativeGenerator
from evaluation_system.report_builder import PatentReportBuilder
from evaluation_system.config import ConfigManager, get_evaluation_config, create_custom_evaluation_config


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="汎用バイナリ分類器評価システム / Generic Binary Classifier Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例 / Examples:
  # デフォルト設定（Patent-Radar）での評価
  %(prog)s --gold-labels testing/data/labels.jsonl --predictions archive/outputs/goldset_results.jsonl
  
  # 事前定義済み設定での評価
  %(prog)s --gold-labels data/labels.jsonl --predictions results.jsonl --config-name spam_filter
  
  # カスタム設定での評価
  %(prog)s --gold-labels data/labels.jsonl --predictions results.jsonl \\
           --model-name "MyClassifier" --positive-class "FRAUD" --negative-class "NORMAL" \\
           --description "詐欺検出システム"

利用可能な事前定義設定 / Available Predefined Configs:
  - patent_radar: 特許衝突検出システム（デフォルト）
  - spam_filter: スパムメール検出システム  
  - fraud_detection: 詐欺検出システム
        """)
    
    # 必須パラメータ
    parser.add_argument(
        '--gold-labels', 
        required=True,
        help='ゴールドラベルファイル（JSONL形式）/ Gold labels file (JSONL format)'
    )
    
    parser.add_argument(
        '--predictions',
        required=True, 
        help='予測結果ファイル（JSONL形式）/ Predictions file (JSONL format)'
    )
    
    # 設定関連パラメータ
    parser.add_argument(
        '--config-name',
        default='patent_radar',
        help='使用する事前定義設定名 / Predefined config name (default: patent_radar)'
    )
    
    parser.add_argument(
        '--config-file',
        help='カスタム設定ファイルパス / Custom config file path'
    )
    
    # カスタム設定パラメータ（config-nameより優先）
    parser.add_argument(
        '--model-name',
        help='モデル名 / Model name (overrides config)'
    )
    
    parser.add_argument(
        '--description',
        help='モデル説明 / Model description'
    )
    
    parser.add_argument(
        '--positive-class',
        help='正例クラス名 / Positive class name (e.g., HIT, SPAM, FRAUD)'
    )
    
    parser.add_argument(
        '--negative-class', 
        help='負例クラス名 / Negative class name (e.g., MISS, HAM, NORMAL)'
    )
    
    parser.add_argument(
        '--false-negative-cost',
        type=float,
        help='見逃しコスト / False negative cost'
    )
    
    parser.add_argument(
        '--false-positive-cost',
        type=float,
        help='誤検出コスト / False positive cost'  
    )
    
    parser.add_argument(
        '--performance-target',
        type=float,
        help='目標性能（F1スコア）/ Target performance (F1 score)'
    )
    
    # 出力関連パラメータ
    parser.add_argument(
        '--base-path',
        default='analysis/evaluations',
        help='評価結果保存ベースパス / Base path for evaluation results (default: analysis/evaluations)'
    )
    
    parser.add_argument(
        '--language',
        choices=['ja', 'en'],
        help='レポート言語 / Report language (ja/en, overrides config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細ログ出力 / Verbose logging'
    )
    
    # その他
    parser.add_argument(
        '--list-configs',
        action='store_true', 
        help='利用可能な設定一覧を表示 / List available configurations'
    )
    
    args = parser.parse_args()
    
    # 設定一覧表示
    if args.list_configs:
        manager = ConfigManager(args.config_file)
        configs = manager.list_available_configs()
        print("利用可能な事前定義設定 / Available Predefined Configurations:")
        for config_name in configs:
            config = manager.get_config(config_name)
            print(f"  {config_name}: {config.model.description}")
        return 0
    
    try:
        # 設定の準備
        if args.model_name:
            # カスタム設定を動的作成
            print(f"[設定] カスタム設定でモデル評価: {args.model_name}")
            
            custom_kwargs = {}
            if args.false_negative_cost:
                custom_kwargs['business.false_negative_cost'] = args.false_negative_cost
            if args.false_positive_cost:
                custom_kwargs['business.false_positive_cost'] = args.false_positive_cost
            if args.performance_target:
                custom_kwargs['model.performance_target'] = args.performance_target
            if args.language:
                custom_kwargs['report.language'] = args.language
            
            config = create_custom_evaluation_config(
                config_name=f"custom_{args.model_name.lower()}",
                model_name=args.model_name,
                description=args.description or f"{args.model_name} Binary Classification System",
                positive_class_name=args.positive_class or "POSITIVE",
                negative_class_name=args.negative_class or "NEGATIVE",
                config_path=args.config_file,
                **custom_kwargs
            )
        else:
            # 事前定義設定を使用
            print(f"[設定] 事前定義設定でモデル評価: {args.config_name}")
            config = get_evaluation_config(args.config_name, args.config_file)
            
            # コマンドライン引数で上書き
            if args.language:
                config.report.language = args.language
        
        print(f"[モデル] {config.model.name} - {config.model.description}")
        print(f"[クラス] {config.model.positive_class_name} vs {config.model.negative_class_name}")
        
        # 評価実行
        success = run_evaluation(
            gold_labels_path=args.gold_labels,
            predictions_path=args.predictions,
            config=config,
            base_path=args.base_path,
            verbose=args.verbose
        )
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"[エラー] 評価実行中にエラーが発生しました: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_evaluation(gold_labels_path: str,
                  predictions_path: str, 
                  config,
                  base_path: str = 'analysis/evaluations',
                  verbose: bool = False) -> bool:
    """
    評価実行
    
    Args:
        gold_labels_path: ゴールドラベルファイルパス
        predictions_path: 予測結果ファイルパス
        config: 評価設定
        base_path: 結果保存ベースパス
        verbose: 詳細ログ
        
    Returns:
        bool: 成功時True
    """
    try:
        print("\n" + "="*60)
        print(f"[開始] {config.model.name} 包括的評価システム")
        print("="*60)
        
        # Step 1: 評価環境セットアップ
        print("\n[ステップ1/8] 評価環境セットアップ")
        master = PatentEvaluationMaster()
        eval_id = master.generate_evaluation_id(config.model.name)
        eval_dir = master.setup_evaluation_environment(eval_id)
        
        print(f"評価ID: {eval_id}")
        print(f"評価ディレクトリ: {eval_dir}")
        
        # Step 2: データ読み込み・整合性チェック
        print("\n[ステップ2/8] データ読み込み・整合性チェック")
        processor = PatentDataProcessor()
        
        print(f"  - ゴールドラベル読み込み: {gold_labels_path}")
        gold_df = processor.load_gold_labels(gold_labels_path)
        
        print(f"  - 予測結果読み込み: {predictions_path}")
        pred_df = processor.load_predictions(predictions_path)
        
        print("  - データ統合...")
        processor.gold_df = gold_df
        processor.pred_df = pred_df
        processor.merge_datasets()
        eval_df = processor.create_evaluation_dataset()
        
        # データセットをファイルに保存
        eval_csv_path = eval_dir / "data" / f"evaluation_dataset_{eval_id}.csv"
        eval_df.to_csv(eval_csv_path, index=False)
        
        sample_count = len(eval_df)
        positive_count = (eval_df['y_true'] == 1).sum()
        negative_count = (eval_df['y_true'] == 0).sum()
        
        print(f"データセット作成完了: {sample_count}件")
        print(f"{config.model.positive_class_name}: {positive_count}件, {config.model.negative_class_name}: {negative_count}件")
        
        # Step 3: 厳密評価・1000分割閾値最適化
        print(f"\n[ステップ3/8] 厳密評価・{config.threshold_resolution}分割閾値最適化")
        evaluator = StrictBinaryEvaluator()
        eval_data_path = eval_dir / "data" / f"evaluation_dataset_{eval_id}.csv"
        evaluator.load_evaluation_data(str(eval_data_path))
        
        print("  - F1スコア最大化による閾値最適化...")
        threshold_results = evaluator.optimize_threshold_f1()
        
        print("  - ROC曲線・AUC計算...")
        roc_results = evaluator.compute_roc_analysis()
        
        print("  - 予測分布分析...")
        dist_results = evaluator.analyze_prediction_distribution()
        
        print("  - 包括的メトリクス生成...")
        comprehensive_metrics = evaluator.generate_comprehensive_metrics()
        
        optimal_threshold = threshold_results['optimal_threshold']
        optimal_f1 = threshold_results['optimal_f1_score'] 
        auc_score = roc_results['auc']
        
        print(f"最適閾値: {optimal_threshold:.3f}")
        print(f"最大F1スコア: {optimal_f1:.3f}")
        print(f"AUC: {auc_score:.3f}")
        
        # Step 4: 可視化生成
        print(f"\n[ステップ4/8] 可視化生成")
        viz_engine = PatentVisualizationEngine(
            dpi=config.visualization.dpi,
            figsize=(config.visualization.figsize_width, config.visualization.figsize_height)
        )
        
        # 可視化ファイル生成
        visualizations = []
        viz_dir = eval_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # ROC曲線
        if 'roc_analysis' in comprehensive_metrics['evaluation_results']:
            roc_path = viz_engine.create_roc_curve(
                comprehensive_metrics['evaluation_results']['roc_analysis'],
                str(viz_dir / "roc_curve.png"),
                eval_id
            )
            visualizations.append(roc_path)
        
        # 予測分布
        if 'distribution_analysis' in comprehensive_metrics['evaluation_results']:
            dist_path = viz_engine.create_prediction_distribution(
                comprehensive_metrics['evaluation_results']['distribution_analysis'],
                str(viz_dir / "prediction_distribution.png"),
                eval_id,
                threshold_results['optimal_threshold']
            )
            visualizations.append(dist_path)
        
        # F1最適化
        if 'threshold_optimization' in comprehensive_metrics['evaluation_results']:
            f1_path = viz_engine.create_f1_optimization_curve(
                comprehensive_metrics['evaluation_results']['threshold_optimization'],
                str(viz_dir / "f1_optimization.png"),
                eval_id
            )
            visualizations.append(f1_path)
        
        print(f"生成されたファイル数: {len(visualizations)}個")
        
        # Step 5: 文章生成
        print(f"\n[ステップ5/8] 文章生成") 
        narrator = PatentEvaluationNarrativeGenerator()
        narratives = narrator.generate_comprehensive_narrative(
            comprehensive_metrics, config.model.name
        )
        
        print(f"生成されたセクション: 全セクション")
        
        # Step 6: HTML形式レポート構築
        print(f"\n[ステップ6/8] HTML形式レポート構築")
        report_builder = PatentReportBuilder()
        
        # モデル固有の設定を適用
        model_info = {
            'name': config.model.name,
            'description': config.model.description,
            'positive_class_name': config.model.positive_class_name,
            'negative_class_name': config.model.negative_class_name
        }
        
        report_path = report_builder.build_comprehensive_report(
            comprehensive_metrics, narratives, visualizations, 
            eval_dir, eval_id, model_info
        )
        
        print(f"HTMLレポート生成完了: {report_path}")
        
        # Step 7: 評価結果インデックス更新  
        print(f"\n[ステップ7/8] 評価結果インデックス更新")
        master.update_evaluation_index(eval_id, {
            'model_name': config.model.name,
            'description': config.model.description,
            'total_samples': sample_count,
            'positive_samples': int(positive_count),
            'negative_samples': int(negative_count),
            'accuracy': float(comprehensive_metrics['evaluation_results']['threshold_optimization']['optimal_metrics']['accuracy']),
            'f1_score': float(optimal_f1),
            'auc': float(auc_score),
            'report': str(report_path),
            'eval_dir': str(eval_dir)
        })
        
        print("評価結果インデックス更新完了")
        
        # Step 8: 結果サマリー出力
        print(f"\n[ステップ8/8] 結果サマリー出力")
        
        print("\n" + "="*60)
        print(f"[評価完了] {config.model.name}の包括的評価が完了しました")
        print("="*60)
        
        print(f"\n[評価サマリー] {eval_id}")
        print(f"--------------------------------------------------")
        print(f"データ規模: {sample_count}件")
        print(f"  - {config.model.positive_class_name}: {positive_count}件 ({positive_count/sample_count*100:.1f}%)")
        print(f"  - {config.model.negative_class_name}: {negative_count}件 ({negative_count/sample_count*100:.1f}%)")
        
        optimal_metrics = comprehensive_metrics['evaluation_results']['threshold_optimization']['optimal_metrics']
        print(f"\n主要指標:")
        print(f"  - Accuracy: {optimal_metrics['accuracy']:.3f}")
        print(f"  - Precision: {optimal_metrics['precision']:.3f}") 
        print(f"  - Recall: {optimal_metrics['recall']:.3f}")
        print(f"  - F1-Score: {optimal_f1:.3f}")
        print(f"  - AUC: {auc_score:.3f}")
        print(f"  - 最適閾値: {optimal_threshold:.3f}")
        
        if config.report.include_business_metrics:
            business_metrics = comprehensive_metrics['evaluation_results'].get('business_metrics', {})
            if business_metrics:
                optimal_business = business_metrics.get('optimal', {})
                print(f"\nビジネス指標:")
                print(f"  - 見逃し件数: {optimal_metrics.get('fn', 0)}件")
                print(f"  - 誤検出件数: {optimal_metrics.get('fp', 0)}件")
                print(f"  - 機会損失: {optimal_business.get('opportunity_cost', 0):,.0f}{config.business.cost_unit_name}")
                print(f"  - 工数コスト: {optimal_business.get('labor_cost', 0):,.0f}{config.business.cost_unit_name}")
        
        print(f"\n出力ファイル:")
        print(f"  - HTMLレポート: {report_path}")
        print(f"  - 評価ディレクトリ: {eval_dir}")
        print(f"  - 可視化ファイル: {len(visualizations)}個")
        
        print(f"\n[完了] 包括的評価が正常に完了しました")
        print(f"HTMLレポート: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"[エラー] 評価実行中にエラーが発生: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)