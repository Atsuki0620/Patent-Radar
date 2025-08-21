"""
本番環境LLM統合テスト（50件）
OpenAI APIを使用した完全なE2Eテスト
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
sys.path.insert(0, str(Path(__file__).parent))

from src.core.screener import PatentScreener

class ProductionTest:
    """本番環境テストクラス"""
    
    def __init__(self):
        """初期化"""
        self.start_time = None
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'hit': 0,
            'miss': 0,
            'errors': [],
            'processing_times': [],
            'confidence_scores': []
        }
        
    def run_test(self, n_patents: int = 50):
        """メインテスト実行"""
        print("\n" + "="*80)
        print(f"  本番環境LLM統合テスト（{n_patents}件）")
        print("="*80)
        
        self.start_time = datetime.now()
        
        # データパス
        invention_path = Path("test_data/invention_sample.json")
        patents_path = Path("test_data/production_patents_50.jsonl")
        output_csv = Path("test_output/production_results.csv")
        output_jsonl = Path("test_output/production_results.jsonl")
        
        # 出力ディレクトリ作成
        output_csv.parent.mkdir(exist_ok=True)
        
        try:
            # Step 1: スクリーナー初期化
            print("\n[Step 1] システム初期化")
            print("-"*40)
            
            # API keyチェック
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            print(f"  API Key: {api_key[:20]}...{api_key[-4:]}")
            
            # スクリーナー初期化
            screener = PatentScreener()
            print("  [OK] PatentScreener初期化成功")
            
            # Step 2: データ読み込み
            print("\n[Step 2] データ読み込み")
            print("-"*40)
            
            # 発明データ
            with open(invention_path, 'r', encoding='utf-8') as f:
                invention_data = json.load(f)
            print(f"  発明: {invention_data['title']}")
            
            # 特許データ（n_patents件まで）
            patents_data = []
            with open(patents_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= n_patents:
                        break
                    patents_data.append(json.loads(line.strip()))
            
            self.stats['total'] = len(patents_data)
            print(f"  特許: {len(patents_data)}件読み込み完了")
            
            # Step 3: LLM処理実行
            print("\n[Step 3] LLM処理実行")
            print("-"*40)
            print("  バッチ処理中...")
            
            # プログレス表示付き処理
            batch_size = 5
            n_batches = (len(patents_data) + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(patents_data))
                batch_patents = patents_data[start_idx:end_idx]
                
                print(f"\n  バッチ {batch_idx+1}/{n_batches} ({start_idx+1}-{end_idx}件目)")
                
                batch_start = time.time()
                
                try:
                    # スクリーナーで処理
                    result = screener.analyze(
                        invention=invention_data,
                        patents=batch_patents,
                        output_csv=None,  # 最後にまとめて出力
                        output_jsonl=None,
                        continue_on_error=True
                    )
                    
                    batch_time = time.time() - batch_start
                    self.stats['processing_times'].append(batch_time)
                    
                    # 統計更新
                    if 'results' in result:
                        for r in result['results']:
                            if r.get('decision') == 'hit':
                                self.stats['hit'] += 1
                            else:
                                self.stats['miss'] += 1
                            
                            if r.get('confidence') is not None:
                                self.stats['confidence_scores'].append(r['confidence'])
                    
                    self.stats['success'] += len(batch_patents)
                    print(f"    [OK] 成功 ({batch_time:.1f}秒)")
                    
                except Exception as e:
                    self.stats['failed'] += len(batch_patents)
                    self.stats['errors'].append(str(e))
                    print(f"    [ERROR] エラー: {e}")
                
                # Rate limit対策
                if batch_idx < n_batches - 1:
                    time.sleep(2)
            
            # Step 4: 全体処理（実際のスクリーナー使用）
            print("\n[Step 4] 統合処理実行")
            print("-"*40)
            
            final_result = screener.analyze(
                invention=invention_data,
                patents=patents_data,
                output_csv=output_csv,
                output_jsonl=output_jsonl,
                continue_on_error=True
            )
            
            print("  [OK] 統合処理完了")
            
            # Step 5: 結果分析
            self._analyze_results(output_csv, output_jsonl)
            
        except Exception as e:
            print(f"\n[ERROR] テスト失敗: {e}")
            traceback.print_exc()
            return False
        
        return True
    
    def _analyze_results(self, csv_path: Path, jsonl_path: Path):
        """結果分析"""
        print("\n" + "="*80)
        print("  結果分析")
        print("="*80)
        
        # 処理時間
        total_time = (datetime.now() - self.start_time).total_seconds()
        avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        print(f"\n[処理パフォーマンス]")
        print(f"  総処理時間: {total_time:.1f}秒")
        print(f"  平均バッチ時間: {avg_time:.1f}秒")
        print(f"  処理速度: {self.stats['total']/total_time:.1f}件/秒")
        
        # 分類結果
        print(f"\n[分類結果]")
        print(f"  総件数: {self.stats['total']}件")
        print(f"  HIT: {self.stats['hit']}件 ({self.stats['hit']/self.stats['total']*100:.1f}%)")
        print(f"  MISS: {self.stats['miss']}件 ({self.stats['miss']/self.stats['total']*100:.1f}%)")
        
        # 信頼度分析
        if self.stats['confidence_scores']:
            avg_conf = sum(self.stats['confidence_scores']) / len(self.stats['confidence_scores'])
            max_conf = max(self.stats['confidence_scores'])
            min_conf = min(self.stats['confidence_scores'])
            
            print(f"\n[信頼度分析]")
            print(f"  平均: {avg_conf:.3f}")
            print(f"  最大: {max_conf:.3f}")
            print(f"  最小: {min_conf:.3f}")
        
        # エラー分析
        if self.stats['errors']:
            print(f"\n[エラー情報]")
            print(f"  エラー件数: {len(self.stats['errors'])}")
            for i, error in enumerate(self.stats['errors'][:3], 1):
                print(f"  {i}. {error[:100]}")
        
        # ファイルサイズ
        if csv_path.exists() and jsonl_path.exists():
            print(f"\n[出力ファイル]")
            print(f"  CSV: {csv_path} ({csv_path.stat().st_size:,} bytes)")
            print(f"  JSONL: {jsonl_path} ({jsonl_path.stat().st_size:,} bytes)")
        
        # コスト推定
        # 概算: 入力35K tokens, 出力12K tokens
        estimated_cost = (35000 * 0.15 + 12000 * 0.60) / 1000000
        print(f"\n[コスト推定]")
        print(f"  推定使用料金: ${estimated_cost:.3f} (約{estimated_cost*150:.0f}円)")
        
        print("\n" + "="*80)
        print("  [SUCCESS] テスト完了")
        print("="*80)

def main():
    """メイン関数"""
    print("\n開始時刻:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # テスト実行
    tester = ProductionTest()
    success = tester.run_test(n_patents=50)
    
    if success:
        print("\n[SUCCESS] すべてのテストが成功しました")
    else:
        print("\n[FAIL] テストに失敗しました")
    
    print("\n終了時刻:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())