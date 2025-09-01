#!/usr/bin/env python3
"""
Patent-Radar 包括的評価システム - 評価マスター
評価の統括制御、ID管理、履歴管理を担当
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging


class PatentEvaluationMaster:
    """評価統括制御クラス"""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        初期化
        
        Args:
            base_path: ベースディレクトリパス（デフォルト: analysis）
        """
        if base_path is None:
            # スクリプトの場所から相対的にanalysisディレクトリを特定
            script_dir = Path(__file__).parent
            self.base_path = script_dir.parent
        else:
            self.base_path = Path(base_path)
        
        self.evaluations_dir = self.base_path / "evaluations"
        self.docs_dir = self.base_path / "docs"
        self.index_file = self.evaluations_dir / "index.json"
        
        # ロギング設定
        self._setup_logging()
        
    def _setup_logging(self):
        """ロギング設定"""
        log_dir = self.base_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"evaluation_master_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def generate_evaluation_id(self, model_name: str) -> str:
        """
        評価ID生成（PE_{YYYYMMDD}_{HHMMSS}_{MODEL_NAME}形式）
        
        Args:
            model_name: モデル名
            
        Returns:
            評価ID文字列
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # モデル名のサニタイズ（ファイル名に使用できない文字を除去）
        safe_model_name = "".join(c for c in model_name if c.isalnum() or c in "._-")
        
        eval_id = f"PE_{timestamp}_{safe_model_name}"
        
        self.logger.info(f"Generated evaluation ID: {eval_id}")
        return eval_id
        
    def setup_evaluation_environment(self, eval_id: str) -> Path:
        """
        評価用ディレクトリ・ファイル構造作成
        
        Args:
            eval_id: 評価ID
            
        Returns:
            評価ディレクトリのパス
        """
        eval_dir = self.evaluations_dir / eval_id
        
        # ディレクトリ構造作成
        directories = [
            eval_dir,
            eval_dir / "data",
            eval_dir / "visualizations", 
            eval_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
            
        self.logger.info(f"Evaluation environment setup completed: {eval_dir}")
        return eval_dir
        
    def load_evaluation_index(self) -> Dict:
        """
        評価履歴索引の読み込み
        
        Returns:
            索引データ辞書
        """
        if not self.index_file.exists():
            # 初回実行時は空の索引を作成
            index_data = {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_evaluations": 0,
                "evaluations": []
            }
            
            self.evaluations_dir.mkdir(parents=True, exist_ok=True)
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Created new evaluation index: {self.index_file}")
            return index_data
            
        with open(self.index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
            
        self.logger.debug(f"Loaded evaluation index with {len(index_data['evaluations'])} entries")
        return index_data
        
    def update_evaluation_index(self, eval_id: str, evaluation_info: Dict) -> bool:
        """
        評価履歴索引の更新
        
        Args:
            eval_id: 評価ID
            evaluation_info: 評価情報辞書
            
        Returns:
            更新成功フラグ
        """
        try:
            index_data = self.load_evaluation_index()
            
            # 既存エントリのチェック（重複防止）
            existing_ids = [eval_data["eval_id"] for eval_data in index_data["evaluations"]]
            if eval_id in existing_ids:
                self.logger.warning(f"Evaluation ID {eval_id} already exists in index")
                return False
                
            # 新しい評価エントリを追加
            evaluation_entry = {
                "eval_id": eval_id,
                "timestamp": datetime.now().isoformat(),
                "model_name": evaluation_info.get("model_name", "Unknown"),
                "description": evaluation_info.get("description", ""),
                "data_stats": evaluation_info.get("data_stats", {}),
                "performance": evaluation_info.get("performance", {}),
                "file_paths": {
                    "report": str(self.evaluations_dir / eval_id / "report.html"),
                    "metrics": str(self.evaluations_dir / eval_id / "data" / f"comprehensive_metrics_{eval_id}.json"),
                    "predictions": str(self.evaluations_dir / eval_id / "data" / f"final_predictions_{eval_id}.csv"),
                    "evaluation_dir": str(self.evaluations_dir / eval_id)
                }
            }
            
            index_data["evaluations"].append(evaluation_entry)
            index_data["total_evaluations"] = len(index_data["evaluations"])
            index_data["last_updated"] = datetime.now().isoformat()
            
            # インデックスファイル更新
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Updated evaluation index with {eval_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update evaluation index: {e}")
            return False
            
    def get_evaluation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        評価履歴取得
        
        Args:
            limit: 取得件数上限（None=全件）
            
        Returns:
            評価履歴リスト（新しい順）
        """
        index_data = self.load_evaluation_index()
        evaluations = sorted(
            index_data["evaluations"], 
            key=lambda x: x["timestamp"], 
            reverse=True
        )
        
        if limit is not None:
            evaluations = evaluations[:limit]
            
        self.logger.debug(f"Retrieved {len(evaluations)} evaluation history entries")
        return evaluations
        
    def find_evaluation_by_id(self, eval_id: str) -> Optional[Dict]:
        """
        評価ID による評価情報取得
        
        Args:
            eval_id: 評価ID
            
        Returns:
            評価情報辞書（見つからない場合はNone）
        """
        index_data = self.load_evaluation_index()
        
        for evaluation in index_data["evaluations"]:
            if evaluation["eval_id"] == eval_id:
                self.logger.debug(f"Found evaluation: {eval_id}")
                return evaluation
                
        self.logger.warning(f"Evaluation not found: {eval_id}")
        return None
        
    def find_evaluations_by_model(self, model_name: str, exact_match: bool = True) -> List[Dict]:
        """
        モデル名による評価検索
        
        Args:
            model_name: モデル名
            exact_match: 完全一致フラグ
            
        Returns:
            マッチした評価リスト
        """
        index_data = self.load_evaluation_index()
        matching_evaluations = []
        
        for evaluation in index_data["evaluations"]:
            eval_model_name = evaluation.get("model_name", "")
            
            if exact_match:
                if eval_model_name == model_name:
                    matching_evaluations.append(evaluation)
            else:
                if model_name.lower() in eval_model_name.lower():
                    matching_evaluations.append(evaluation)
                    
        self.logger.debug(f"Found {len(matching_evaluations)} evaluations for model: {model_name}")
        return sorted(matching_evaluations, key=lambda x: x["timestamp"], reverse=True)
        
    def cleanup_evaluation(self, eval_id: str, remove_files: bool = False) -> bool:
        """
        評価データのクリーンアップ
        
        Args:
            eval_id: 評価ID
            remove_files: ファイル削除フラグ
            
        Returns:
            クリーンアップ成功フラグ
        """
        try:
            # インデックスから削除
            index_data = self.load_evaluation_index()
            original_count = len(index_data["evaluations"])
            index_data["evaluations"] = [
                eval_data for eval_data in index_data["evaluations"]
                if eval_data["eval_id"] != eval_id
            ]
            
            if len(index_data["evaluations"]) == original_count:
                self.logger.warning(f"Evaluation {eval_id} not found in index")
                return False
                
            index_data["total_evaluations"] = len(index_data["evaluations"])
            index_data["last_updated"] = datetime.now().isoformat()
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
                
            # ファイル削除（オプション）
            if remove_files:
                eval_dir = self.evaluations_dir / eval_id
                if eval_dir.exists():
                    import shutil
                    shutil.rmtree(eval_dir)
                    self.logger.info(f"Removed evaluation directory: {eval_dir}")
                    
            self.logger.info(f"Cleaned up evaluation: {eval_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup evaluation {eval_id}: {e}")
            return False
            
    def get_system_stats(self) -> Dict:
        """
        システム統計情報取得
        
        Returns:
            統計情報辞書
        """
        index_data = self.load_evaluation_index()
        
        if not index_data["evaluations"]:
            return {
                "total_evaluations": 0,
                "first_evaluation": None,
                "last_evaluation": None,
                "models_evaluated": [],
                "disk_usage_mb": 0
            }
            
        evaluations = index_data["evaluations"]
        timestamps = [eval_data["timestamp"] for eval_data in evaluations]
        model_names = list(set(eval_data.get("model_name", "Unknown") for eval_data in evaluations))
        
        # ディスク使用量計算
        total_size = 0
        if self.evaluations_dir.exists():
            for file_path in self.evaluations_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    
        stats = {
            "total_evaluations": len(evaluations),
            "first_evaluation": min(timestamps),
            "last_evaluation": max(timestamps),
            "models_evaluated": sorted(model_names),
            "disk_usage_mb": round(total_size / (1024 * 1024), 2)
        }
        
        self.logger.debug(f"System stats: {stats}")
        return stats
        
    def validate_environment(self) -> Dict[str, bool]:
        """
        実行環境の検証
        
        Returns:
            検証結果辞書
        """
        validation_results = {}
        
        # ディレクトリ存在確認
        validation_results["base_directory"] = self.base_path.exists()
        validation_results["evaluations_directory"] = self.evaluations_dir.exists()
        validation_results["docs_directory"] = self.docs_dir.exists()
        
        # 必要ファイル確認
        required_docs = [
            "REPORT_CREATION_RULES.md",
            "USAGE_GUIDE.md", 
            "EVALUATION_METHODOLOGY.md"
        ]
        
        for doc in required_docs:
            doc_path = self.docs_dir / doc
            validation_results[f"doc_{doc}"] = doc_path.exists()
            
        # 権限確認
        try:
            test_file = self.evaluations_dir / ".write_test"
            test_file.mkdir(exist_ok=True)
            test_file.rmdir()
            validation_results["write_permissions"] = True
        except:
            validation_results["write_permissions"] = False
            
        # 全体的な状態
        validation_results["overall_status"] = all(validation_results.values())
        
        self.logger.info(f"Environment validation: {validation_results}")
        return validation_results


def main():
    """テスト実行用メイン関数"""
    print("=== Patent Evaluation Master Test ===")
    
    # 評価マスター初期化
    master = PatentEvaluationMaster()
    
    # 環境検証
    print("\n1. Environment Validation:")
    validation = master.validate_environment()
    for key, status in validation.items():
        print(f"  {key}: {'✓' if status else '✗'}")
    
    # テスト評価ID生成
    print("\n2. Evaluation ID Generation:")
    test_ids = [
        master.generate_evaluation_id("PatentRadar_v1.0"),
        master.generate_evaluation_id("PatentRadar_baseline"),
        master.generate_evaluation_id("Test-Model_v2.3")
    ]
    for eval_id in test_ids:
        print(f"  {eval_id}")
    
    # システム統計
    print("\n3. System Statistics:")
    stats = master.get_system_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 評価履歴
    print("\n4. Evaluation History (last 5):")
    history = master.get_evaluation_history(limit=5)
    if history:
        for eval_data in history:
            print(f"  {eval_data['eval_id']} - {eval_data.get('model_name', 'Unknown')} [{eval_data['timestamp'][:16]}]")
    else:
        print("  No evaluation history found")
        
    print("\n=== Test completed ===")


if __name__ == "__main__":
    main()