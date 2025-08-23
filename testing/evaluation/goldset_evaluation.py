#!/usr/bin/env python3
"""
ゴールドラベル付きデータによるシステム評価スクリプト
60件の正解データでPatentScreenerの精度を評価
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

# パスの追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.screener import PatentScreener

class GoldsetEvaluation:
    """ゴールドラベルデータによる評価クラス"""
    
    def __init__(self):
        """初期化"""
        self.start_time = None
        self.results = {
            'predictions': [],
            'gold_labels': {},
            'stats': {
                'total': 0,
                'processed': 0,
                'errors': 0,
                'processing_times': []
            }
        }
        
    def load_goldset(self):
        """ゴールドラベルデータの読み込み"""
        print("\n[Step 1] ゴールドラベルデータ読み込み")
        print("-" * 50)
        
        # 特許データ
        patents_path = Path("testing/data/patents.jsonl")
        patents_data = []
        
        with open(patents_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    patents_data.append(json.loads(line))
        
        print(f"  特許データ: {len(patents_data)}件読み込み完了")
        
        # ゴールドラベル
        labels_path = Path("testing/data/labels.jsonl")
        gold_labels = {}
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    label_data = json.loads(line)
                    gold_labels[label_data['publication_number']] = label_data
        
        print(f"  ゴールドラベル: {len(gold_labels)}件読み込み完了")
        
        # ラベル分布確認
        label_counts = {}
        for label_data in gold_labels.values():
            label = label_data['gold_label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"  ラベル分布: {label_counts}")
        
        return patents_data, gold_labels
    
    def run_evaluation(self):
        """メイン評価実行"""
        print("\n" + "=" * 80)
        print("  ゴールドラベルデータによるシステム評価")
        print("=" * 80)
        
        self.start_time = datetime.now()
        
        try:
            # Step 1: データ読み込み
            patents_data, gold_labels = self.load_goldset()
            self.results['gold_labels'] = gold_labels
            self.results['stats']['total'] = len(patents_data)
            
            # Step 2: スクリーナー初期化
            print("\n[Step 2] システム初期化")
            print("-" * 50)
            
            # API keyチェック
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            print(f"  API Key: {api_key[:20]}...{api_key[-4:]}")
            
            # 発明データ（テスト用の一般的な液体分離設備発明）
            invention_data = {
                "title": "液体分離設備の効率改善システム",
                "problem": "従来の膜分離装置では分離効率が低く、エネルギー消費が大きい。特に高粘度流体の処理において、膜の目詰まりが頻繁に発生し、メンテナンスコストが増大する",
                "solution": "新規ポリマー材料と最適化された流路設計により分離性能を向上。AIベースの予測制御システムを導入し、運転条件を自動最適化",
                "effects": "分離効率50%向上、エネルギー消費30%削減、メンテナンス頻度60%低減",
                "key_elements": [
                    "新規ポリマー膜材料",
                    "最適化流路設計", 
                    "AI予測制御システム",
                    "リアルタイム監視機能"
                ],
                "constraints": "処理量100L/h以上、運転圧力0.5MPa以下"
            }
            
            # スクリーナー初期化
            screener = PatentScreener()
            print("  [OK] PatentScreener初期化成功")
            
            # Step 3: バッチ処理実行
            print("\n[Step 3] 特許分類実行")
            print("-" * 50)
            
            # 出力先準備
            output_dir = Path("archive/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_csv = output_dir / "goldset_results.csv"
            output_jsonl = output_dir / "goldset_results.jsonl"
            
            print(f"  処理対象: {len(patents_data)}件")
            print("  バッチ処理開始...")
            
            # 処理実行
            batch_start = time.time()
            
            result = screener.analyze(
                invention=invention_data,
                patents=patents_data,
                output_csv=output_csv,
                output_jsonl=output_jsonl,
                continue_on_error=True
            )
            
            processing_time = time.time() - batch_start
            self.results['stats']['processing_times'].append(processing_time)
            
            print(f"  [OK] 処理完了 ({processing_time:.1f}秒)")
            
            # Step 4: 結果読み込み
            self._load_predictions(output_jsonl)
            
            # Step 5: 基本統計
            self._calculate_basic_stats()
            
        except Exception as e:
            print(f"\n[ERROR] 評価失敗: {e}")
            traceback.print_exc()
            return False
        
        return True
    
    def _load_predictions(self, jsonl_path: Path):
        """予測結果の読み込み"""
        print("\n[Step 4] 予測結果読み込み")
        print("-" * 50)
        
        predictions = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
        
        self.results['predictions'] = predictions
        self.results['stats']['processed'] = len(predictions)
        
        print(f"  予測結果: {len(predictions)}件読み込み完了")
    
    def _calculate_basic_stats(self):
        """基本統計の計算"""
        print("\n[Step 5] 基本統計計算")
        print("-" * 50)
        
        predictions = self.results['predictions']
        gold_labels = self.results['gold_labels']
        
        # 分類結果統計
        pred_counts = {'hit': 0, 'miss': 0, 'borderline': 0, 'error': 0}
        gold_counts = {'hit': 0, 'miss': 0, 'borderline': 0}
        
        matches = 0
        confidence_scores = []
        
        for pred in predictions:
            pub_num = pred['publication_number']
            pred_decision = pred.get('decision', 'error')
            confidence = pred.get('confidence', 0.0)
            
            if confidence > 0:
                confidence_scores.append(confidence)
            
            # 予測統計
            if pred_decision in pred_counts:
                pred_counts[pred_decision] += 1
            else:
                pred_counts['error'] += 1
            
            # 正解統計とマッチング
            if pub_num in gold_labels:
                gold_label = gold_labels[pub_num]['gold_label']
                gold_counts[gold_label] = gold_counts.get(gold_label, 0) + 1
                
                # 完全一致確認
                if pred_decision == gold_label:
                    matches += 1
        
        # 統計更新
        total_valid = len([p for p in predictions if p.get('confidence', 0) > 0])
        accuracy = matches / total_valid if total_valid > 0 else 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        self.results['stats'].update({
            'pred_counts': pred_counts,
            'gold_counts': gold_counts,
            'exact_matches': matches,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'confidence_scores': confidence_scores
        })
        
        print(f"  予測分布: {pred_counts}")
        print(f"  正解分布: {gold_counts}")
        print(f"  完全一致: {matches}/{total_valid} ({accuracy*100:.1f}%)")
        print(f"  平均信頼度: {avg_confidence:.3f}")
    
    def save_evaluation_results(self):
        """評価結果の保存"""
        output_path = Path("archive/outputs/goldset_evaluation_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 結果データ準備
        results_data = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_patents': self.results['stats']['total'],
                'processed_patents': self.results['stats']['processed'],
                'processing_time': sum(self.results['stats']['processing_times']),
                'data_source': 'tests/data/patents.jsonl'
            },
            'basic_statistics': self.results['stats'],
            'predictions': self.results['predictions'][:5],  # サンプルのみ保存
            'gold_label_summary': {
                pub_num: label_data for pub_num, label_data in 
                list(self.results['gold_labels'].items())[:5]  # サンプルのみ
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n評価結果保存完了: {output_path}")
        return output_path

def main():
    """メイン関数"""
    print("\n開始時刻:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 評価実行
    evaluator = GoldsetEvaluation()
    success = evaluator.run_evaluation()
    
    if success:
        # 結果保存
        result_path = evaluator.save_evaluation_results()
        
        print("\n" + "=" * 80)
        print("  [SUCCESS] ゴールドラベル評価完了")
        print("=" * 80)
        
        stats = evaluator.results['stats']
        print(f"処理件数: {stats['processed']}/{stats['total']}")
        print(f"精度: {stats['accuracy']*100:.1f}%")
        print(f"平均信頼度: {stats['avg_confidence']:.3f}")
        print(f"結果ファイル: {result_path}")
        
    else:
        print("\n[FAIL] 評価に失敗しました")
        return 1
    
    print("\n終了時刻:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return 0

if __name__ == "__main__":
    sys.exit(main())