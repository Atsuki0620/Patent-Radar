#!/usr/bin/env python3
"""
Patent-Radar データ統合・前処理モジュール  
JSONL形式のゴールドラベルと予測結果を統合してCSV形式の評価データセットを作成
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging


class PatentDataProcessor:
    """JSONL形式データの統合・CSV変換クラス"""
    
    def __init__(self, random_seed: int = 42):
        """
        初期化
        
        Args:
            random_seed: 再現性確保のための乱数シード
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # データ保存用
        self.gold_labels_df = None
        self.predictions_df = None
        self.merged_df = None
        self.evaluation_dataset = None
        
        # 統計情報保存用
        self.data_stats = {}
        
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
            
    def load_gold_labels(self, labels_path: str) -> pd.DataFrame:
        """
        ゴールドラベルJSONLファイルの読み込み・正規化
        
        Args:
            labels_path: ゴールドラベルファイルパス
            
        Returns:
            正規化されたゴールドラベルDataFrame
        """
        self.logger.info(f"Loading gold labels from: {labels_path}")
        
        labels_data = []
        
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        item = json.loads(line)
                        
                        # 必須フィールドの検証
                        if 'publication_number' not in item:
                            self.logger.warning(f"Missing publication_number in line {line_num}")
                            continue
                            
                        if 'gold_label' not in item:
                            self.logger.warning(f"Missing gold_label in line {line_num}")
                            continue
                            
                        # データ正規化
                        normalized_item = {
                            'publication_number': str(item['publication_number']).strip(),
                            'gold_label': str(item['gold_label']).lower().strip(),
                            'brief_rationale': item.get('brief_rationale', ''),
                            'source_line': line_num
                        }
                        
                        labels_data.append(normalized_item)
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON decode error in line {line_num}: {e}")
                        continue
                        
            if not labels_data:
                raise ValueError(f"No valid gold labels found in {labels_path}")
                
            self.gold_labels_df = pd.DataFrame(labels_data)
            
            # データ品質チェック
            self._validate_gold_labels()
            
            self.logger.info(f"Loaded {len(self.gold_labels_df)} gold labels")
            return self.gold_labels_df
            
        except Exception as e:
            self.logger.error(f"Failed to load gold labels: {e}")
            raise
            
    def load_predictions(self, predictions_path: str) -> pd.DataFrame:
        """
        予測結果JSONLファイルの読み込み
        
        Args:
            predictions_path: 予測結果ファイルパス
            
        Returns:
            正規化された予測結果DataFrame
        """
        self.logger.info(f"Loading predictions from: {predictions_path}")
        
        predictions_data = []
        
        try:
            with open(predictions_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        item = json.loads(line)
                        
                        # 必須フィールドの検証
                        if 'publication_number' not in item:
                            self.logger.warning(f"Missing publication_number in predictions line {line_num}")
                            continue
                            
                        # データ正規化
                        normalized_item = {
                            'publication_number': str(item['publication_number']).strip(),
                            'predicted_decision': str(item.get('decision', 'miss')).lower().strip(),
                            'confidence': float(item.get('confidence', 0.0)),
                            'rank': int(item.get('rank', 999)),
                            'title': item.get('title', ''),
                            'assignee': item.get('assignee', ''),
                            'pub_date': item.get('pub_date', ''),
                            'hit_reason_1': item.get('hit_reason_1', ''),
                            'hit_src_1': item.get('hit_src_1', ''),
                            'hit_reason_2': item.get('hit_reason_2', ''),
                            'hit_src_2': item.get('hit_src_2', ''),
                            'url_hint': item.get('url_hint', ''),
                            'source_line': line_num
                        }
                        
                        predictions_data.append(normalized_item)
                        
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        self.logger.warning(f"Data error in predictions line {line_num}: {e}")
                        continue
                        
            if not predictions_data:
                raise ValueError(f"No valid predictions found in {predictions_path}")
                
            self.predictions_df = pd.DataFrame(predictions_data)
            
            # データ品質チェック
            self._validate_predictions()
            
            self.logger.info(f"Loaded {len(self.predictions_df)} predictions")
            return self.predictions_df
            
        except Exception as e:
            self.logger.error(f"Failed to load predictions: {e}")
            raise
            
    def _validate_gold_labels(self):
        """ゴールドラベルのデータ品質チェック"""
        if self.gold_labels_df is None:
            return
            
        # 重複チェック
        duplicates = self.gold_labels_df['publication_number'].duplicated()
        if duplicates.any():
            duplicate_count = duplicates.sum()
            self.logger.warning(f"Found {duplicate_count} duplicate publication numbers in gold labels")
            
        # ラベル値チェック
        valid_labels = {'hit', 'miss', 'borderline'}
        invalid_labels = set(self.gold_labels_df['gold_label']) - valid_labels
        if invalid_labels:
            self.logger.warning(f"Found invalid gold labels: {invalid_labels}")
            
        # 統計情報記録
        label_dist = self.gold_labels_df['gold_label'].value_counts().to_dict()
        self.data_stats['gold_labels'] = {
            'total_count': len(self.gold_labels_df),
            'unique_patents': self.gold_labels_df['publication_number'].nunique(),
            'label_distribution': label_dist,
            'duplicate_count': duplicates.sum() if duplicates.any() else 0
        }
        
    def _validate_predictions(self):
        """予測結果のデータ品質チェック"""
        if self.predictions_df is None:
            return
            
        # 重複チェック
        duplicates = self.predictions_df['publication_number'].duplicated()
        if duplicates.any():
            duplicate_count = duplicates.sum()
            self.logger.warning(f"Found {duplicate_count} duplicate publication numbers in predictions")
            
        # 信頼度範囲チェック
        confidence_out_of_range = (
            (self.predictions_df['confidence'] < 0) | 
            (self.predictions_df['confidence'] > 1)
        )
        if confidence_out_of_range.any():
            out_of_range_count = confidence_out_of_range.sum()
            self.logger.warning(f"Found {out_of_range_count} confidence values out of [0,1] range")
            # [0,1]にクリップ
            self.predictions_df['confidence'] = self.predictions_df['confidence'].clip(0, 1)
            
        # 統計情報記録
        decision_dist = self.predictions_df['predicted_decision'].value_counts().to_dict()
        confidence_stats = self.predictions_df['confidence'].describe().to_dict()
        
        self.data_stats['predictions'] = {
            'total_count': len(self.predictions_df),
            'unique_patents': self.predictions_df['publication_number'].nunique(),
            'decision_distribution': decision_dist,
            'confidence_stats': confidence_stats,
            'duplicate_count': duplicates.sum() if duplicates.any() else 0
        }
        
    def merge_datasets(self) -> pd.DataFrame:
        """
        ゴールドラベルと予測結果の結合
        
        Returns:
            結合されたDataFrame
        """
        if self.gold_labels_df is None or self.predictions_df is None:
            raise ValueError("Both gold labels and predictions must be loaded first")
            
        self.logger.info("Merging gold labels and predictions")
        
        # 内結合（両方に存在するデータのみ）
        self.merged_df = pd.merge(
            self.gold_labels_df,
            self.predictions_df,
            on='publication_number',
            how='inner',
            suffixes=('_gold', '_pred')
        )
        
        # 結合結果の統計
        gold_count = len(self.gold_labels_df)
        pred_count = len(self.predictions_df)
        merged_count = len(self.merged_df)
        
        self.logger.info(f"Merge results: {gold_count} gold + {pred_count} pred = {merged_count} merged")
        
        if merged_count == 0:
            raise ValueError("No matching publication numbers found between gold labels and predictions")
            
        # マッチしなかった特許をログ出力
        gold_unmatched = set(self.gold_labels_df['publication_number']) - set(self.merged_df['publication_number'])
        pred_unmatched = set(self.predictions_df['publication_number']) - set(self.merged_df['publication_number'])
        
        if gold_unmatched:
            self.logger.warning(f"{len(gold_unmatched)} gold labels without predictions")
        if pred_unmatched:
            self.logger.warning(f"{len(pred_unmatched)} predictions without gold labels")
            
        # 統計情報記録
        self.data_stats['merge_results'] = {
            'gold_count': gold_count,
            'predictions_count': pred_count,
            'merged_count': merged_count,
            'gold_unmatched_count': len(gold_unmatched),
            'predictions_unmatched_count': len(pred_unmatched),
            'match_rate': merged_count / max(gold_count, pred_count)
        }
        
        return self.merged_df
        
    def create_evaluation_dataset(self) -> pd.DataFrame:
        """
        評価用統合データセットの作成
        - borderline除外による純粋二値分類データ作成
        - hit=1, miss=0 の数値変換
        - 品質チェック・統計サマリー
        
        Returns:
            評価用DataFrame
        """
        if self.merged_df is None:
            raise ValueError("Datasets must be merged first")
            
        self.logger.info("Creating evaluation dataset")
        
        # borderline除外
        binary_mask = self.merged_df['gold_label'].isin(['hit', 'miss'])
        self.evaluation_dataset = self.merged_df[binary_mask].copy()
        
        borderline_count = (~binary_mask).sum()
        if borderline_count > 0:
            self.logger.info(f"Excluded {borderline_count} borderline cases from evaluation")
            
        # 二値変換
        self.evaluation_dataset['y_true'] = (
            self.evaluation_dataset['gold_label'] == 'hit'
        ).astype(int)
        
        self.evaluation_dataset['y_pred'] = (
            self.evaluation_dataset['predicted_decision'] == 'hit'
        ).astype(int)
        
        # 予測確率として confidence を使用
        self.evaluation_dataset['y_proba'] = self.evaluation_dataset['confidence']
        
        # データ品質最終チェック
        self._validate_evaluation_dataset()
        
        # カラム整理
        eval_columns = [
            'publication_number', 'title', 'assignee', 'pub_date',
            'gold_label', 'y_true', 
            'predicted_decision', 'y_pred', 'y_proba',
            'rank', 'hit_reason_1', 'hit_src_1', 'hit_reason_2', 'hit_src_2',
            'url_hint', 'brief_rationale'
        ]
        
        available_columns = [col for col in eval_columns if col in self.evaluation_dataset.columns]
        self.evaluation_dataset = self.evaluation_dataset[available_columns]
        
        self.logger.info(f"Created evaluation dataset with {len(self.evaluation_dataset)} samples")
        return self.evaluation_dataset
        
    def _validate_evaluation_dataset(self):
        """評価データセットの品質チェック"""
        if self.evaluation_dataset is None:
            return
            
        # クラス分布
        class_dist = self.evaluation_dataset['y_true'].value_counts().to_dict()
        hit_rate = class_dist.get(1, 0) / len(self.evaluation_dataset)
        
        # 予測確率の分布
        proba_stats = self.evaluation_dataset['y_proba'].describe().to_dict()
        
        # 異常値チェック
        na_count = self.evaluation_dataset.isnull().sum().sum()
        inf_count = np.isinf(self.evaluation_dataset.select_dtypes(include=[np.number])).sum().sum()
        
        # クラス不均衡度（少数クラス比率）
        imbalance_ratio = min(class_dist.values()) / max(class_dist.values()) if class_dist else 0
        
        self.data_stats['evaluation_dataset'] = {
            'total_samples': len(self.evaluation_dataset),
            'class_distribution': class_dist,
            'hit_rate': hit_rate,
            'imbalance_ratio': imbalance_ratio,
            'proba_stats': proba_stats,
            'na_count': int(na_count),
            'inf_count': int(inf_count),
            'columns': list(self.evaluation_dataset.columns)
        }
        
        # 警告出力
        if hit_rate < 0.1 or hit_rate > 0.9:
            self.logger.warning(f"Severe class imbalance detected: hit rate = {hit_rate:.3f}")
            
        if na_count > 0:
            self.logger.warning(f"Found {na_count} missing values in evaluation dataset")
            
        if inf_count > 0:
            self.logger.warning(f"Found {inf_count} infinite values in evaluation dataset")
            
    def save_evaluation_dataset(self, output_path: str, eval_id: str) -> str:
        """
        評価データセットのCSV保存
        
        Args:
            output_path: 保存先パス
            eval_id: 評価ID
            
        Returns:
            保存されたファイルパス
        """
        if self.evaluation_dataset is None:
            raise ValueError("Evaluation dataset not created yet")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ファイル名にeval_IDを含める
        if eval_id and not eval_id in str(output_path.name):
            stem = output_path.stem
            suffix = output_path.suffix
            filename = f"{stem}_{eval_id}{suffix}"
            output_path = output_path.parent / filename
            
        # CSV保存
        self.evaluation_dataset.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"Saved evaluation dataset to: {output_path}")
        return str(output_path)
        
    def save_data_stats(self, output_path: str, eval_id: str) -> str:
        """
        データ統計情報のJSON保存
        
        Args:
            output_path: 保存先パス
            eval_id: 評価ID
            
        Returns:
            保存されたファイルパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ファイル名にeval_IDを含める
        if eval_id and not eval_id in str(output_path.name):
            stem = output_path.stem
            suffix = output_path.suffix
            filename = f"{stem}_{eval_id}{suffix}"
            output_path = output_path.parent / filename
            
        # 統計情報にメタデータ追加
        stats_with_meta = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'eval_id': eval_id,
                'random_seed': self.random_seed
            },
            'data_statistics': self.data_stats
        }
        
        # JSON保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_with_meta, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved data statistics to: {output_path}")
        return str(output_path)
        
    def get_data_summary(self) -> str:
        """データ処理結果のサマリー文字列生成"""
        if not self.data_stats:
            return "No data processed yet"
            
        summary_lines = ["=== Data Processing Summary ==="]
        
        if 'gold_labels' in self.data_stats:
            gold_stats = self.data_stats['gold_labels']
            summary_lines.extend([
                f"Gold Labels: {gold_stats['total_count']} total",
                f"  Distribution: {gold_stats['label_distribution']}"
            ])
            
        if 'predictions' in self.data_stats:
            pred_stats = self.data_stats['predictions']
            summary_lines.extend([
                f"Predictions: {pred_stats['total_count']} total",
                f"  Confidence: mean={pred_stats['confidence_stats']['mean']:.3f}"
            ])
            
        if 'evaluation_dataset' in self.data_stats:
            eval_stats = self.data_stats['evaluation_dataset']
            summary_lines.extend([
                f"Evaluation Dataset: {eval_stats['total_samples']} samples",
                f"  HIT rate: {eval_stats['hit_rate']:.3f}",
                f"  Imbalance ratio: {eval_stats['imbalance_ratio']:.3f}"
            ])
            
        return "\n".join(summary_lines)


def main():
    """テスト実行用メイン関数"""
    print("=== Patent Data Processor Test ===")
    
    # データプロセッサ初期化
    processor = PatentDataProcessor()
    
    # テストファイルパス
    labels_path = "testing/data/labels.jsonl"
    predictions_path = "archive/outputs/goldset_results.jsonl"
    
    try:
        # データ読み込み
        print(f"\n1. Loading gold labels from: {labels_path}")
        processor.load_gold_labels(labels_path)
        
        print(f"\n2. Loading predictions from: {predictions_path}")
        processor.load_predictions(predictions_path)
        
        # データ結合
        print("\n3. Merging datasets...")
        processor.merge_datasets()
        
        # 評価データセット作成
        print("\n4. Creating evaluation dataset...")
        processor.create_evaluation_dataset()
        
        # サマリー表示
        print("\n5. Processing Summary:")
        print(processor.get_data_summary())
        
        # テスト保存
        eval_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        csv_path = processor.save_evaluation_dataset("analysis/temp/test_dataset.csv", eval_id)
        stats_path = processor.save_data_stats("analysis/temp/test_stats.json", eval_id)
        
        print(f"\n6. Test files saved:")
        print(f"  Dataset: {csv_path}")
        print(f"  Statistics: {stats_path}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("\n=== Test completed successfully ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())