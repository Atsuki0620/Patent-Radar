"""
pipeline.py - LLM処理パイプライン機能

特許分析用のLLM処理パイプラインを提供：
1. 発明要約バッチ処理
2. 特許要約バッチ処理  
3. 二値分類バッチ処理
4. 統合パイプライン処理

Code-reviewerの推奨に基づくスレッドセーフ・高性能・拡張可能な実装。
"""

import os
import json
import time
import threading
import hashlib
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import yaml
from loguru import logger

# 内部モジュールのインポート
try:
    from .client import LLMClient
    from .prompts import (
        create_invention_prompt, 
        create_patent_prompt, 
        create_classification_prompt
    )
except ImportError:
    # テスト実行時の対応
    import sys
    sys.path.append(str(Path(__file__).parent))
    from client import LLMClient
    from prompts import (
        create_invention_prompt, 
        create_patent_prompt, 
        create_classification_prompt
    )


# 設定定数 - Code-reviewer推奨による構造化
class PipelineConstants:
    """パイプライン処理定数"""
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_MAX_WORKERS = 5
    DEFAULT_TIMEOUT = 60
    MAX_BATCH_SIZE = 100
    MAX_WORKERS = 20
    MIN_BATCH_SIZE = 1
    MIN_WORKERS = 1
    MAX_TIMEOUT = 600
    MIN_TIMEOUT = 1

class ProcessingStages:
    """処理段階定数"""
    INVENTION_SUMMARY = "invention_summary"
    PATENT_SUMMARIES = "patent_summaries"
    CLASSIFICATION = "classification"
    EXPORT = "export"

class ProcessingStatus:
    """処理ステータス定数"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


# エラー階層 - Code-reviewer推奨
class PipelineError(Exception):
    """パイプライン処理のベース例外"""
    pass

class PipelineConfigError(PipelineError):
    """パイプライン設定エラー"""
    pass

class PipelineProcessingError(PipelineError):
    """パイプライン処理エラー"""
    pass

class PipelineShutdownError(PipelineError):
    """パイプラインシャットダウンエラー"""
    pass


# スレッドセーフ統計管理 - Code-reviewer推奨
class PipelineStatistics:
    """パイプライン統計情報のスレッドセーフ管理"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'invention_summaries': 0,
            'patent_summaries': 0,
            'classifications': 0,
            'total_tokens_used': 0,
            'processing_time': 0.0,
            'average_processing_time': 0.0,
            'start_time': None,
            'end_time': None,
            'error_breakdown': {},
        }
    
    def increment(self, key: str, value: Union[int, float] = 1):
        """統計値をスレッドセーフに増加"""
        with self._lock:
            if key in self._stats:
                self._stats[key] += value
    
    def set_value(self, key: str, value: Union[int, float]):
        """統計値をスレッドセーフに設定"""
        with self._lock:
            self._stats[key] = value
    
    def add_error(self, error_type: str):
        """エラー種別をカウント"""
        with self._lock:
            if error_type not in self._stats['error_breakdown']:
                self._stats['error_breakdown'][error_type] = 0
            self._stats['error_breakdown'][error_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報のコピーを取得"""
        with self._lock:
            return self._stats.copy()
    
    def start_processing(self):
        """処理開始時刻を記録"""
        with self._lock:
            self._stats['start_time'] = time.time()
    
    def end_processing(self):
        """処理終了時刻を記録し平均処理時間を計算"""
        with self._lock:
            self._stats['end_time'] = time.time()
            if self._stats['start_time']:
                self._stats['processing_time'] = self._stats['end_time'] - self._stats['start_time']
                if self._stats['total_processed'] > 0:
                    self._stats['average_processing_time'] = (
                        self._stats['processing_time'] / self._stats['total_processed']
                    )


def load_pipeline_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    パイプライン設定を読み込む
    
    Args:
        config_path: 設定ファイルパス（省略時はデフォルト）
        
    Returns:
        設定辞書
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Config file parsing error: {e}")
        return {}


def validate_pipeline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    パイプライン設定を検証
    
    Args:
        config: 設定辞書
        
    Returns:
        検証済み設定辞書
        
    Raises:
        PipelineConfigError: 設定が無効な場合
    """
    # デフォルト設定
    default_config = {
        'processing': {
            'batch_size': PipelineConstants.DEFAULT_BATCH_SIZE,
            'max_workers': PipelineConstants.DEFAULT_MAX_WORKERS,
            'timeout': PipelineConstants.DEFAULT_TIMEOUT
        }
    }
    
    # 既存設定とマージ
    merged_config = {**default_config}
    if config:
        if 'processing' in config:
            merged_config['processing'].update(config['processing'])
        
        # 他の設定もマージ
        for key in ['llm', 'io', 'debug']:
            if key in config:
                merged_config[key] = config[key]
    
    # 処理設定の検証
    processing = merged_config['processing']
    
    # バッチサイズ検証
    batch_size = processing['batch_size']
    if not isinstance(batch_size, int) or batch_size < PipelineConstants.MIN_BATCH_SIZE or batch_size > PipelineConstants.MAX_BATCH_SIZE:
        raise PipelineConfigError(f"Invalid batch_size: {batch_size}. Must be {PipelineConstants.MIN_BATCH_SIZE}-{PipelineConstants.MAX_BATCH_SIZE}")
    
    # ワーカー数検証
    max_workers = processing['max_workers']
    if not isinstance(max_workers, int) or max_workers < PipelineConstants.MIN_WORKERS or max_workers > PipelineConstants.MAX_WORKERS:
        raise PipelineConfigError(f"Invalid max_workers: {max_workers}. Must be {PipelineConstants.MIN_WORKERS}-{PipelineConstants.MAX_WORKERS}")
    
    # タイムアウト検証
    timeout = processing['timeout']
    if not isinstance(timeout, (int, float)) or timeout < PipelineConstants.MIN_TIMEOUT or timeout > PipelineConstants.MAX_TIMEOUT:
        raise PipelineConfigError(f"Invalid timeout: {timeout}. Must be {PipelineConstants.MIN_TIMEOUT}-{PipelineConstants.MAX_TIMEOUT}")
    
    return merged_config


class LLMPipeline:
    """
    LLM処理パイプライン
    
    特許分析用のLLM処理を効率的にバッチ処理する。
    スレッドセーフ、エラー回復、進捗追跡、プラグインシステム対応。
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
        plugins: Optional[Dict[str, Callable]] = None,
        enable_monitoring: bool = False
    ):
        """
        パイプラインを初期化
        
        Args:
            config: パイプライン設定
            custom_prompts: カスタムプロンプト設定
            plugins: プラグイン辞書
            enable_monitoring: 監視機能を有効にするか
        """
        logger.info("Initializing LLM pipeline")
        
        # 設定読み込みと検証
        if config is None:
            config = load_pipeline_config()
        self._config = validate_pipeline_config(config)
        
        # LLMクライアント初期化
        self._client = LLMClient(
            max_workers=self._config['processing']['max_workers']
        )
        
        # 統計管理
        self._stats = PipelineStatistics()
        
        # カスタムプロンプト
        self._custom_prompts = custom_prompts or {}
        
        # プラグインシステム
        self._plugins = plugins or {}
        
        # 監視機能
        self._enable_monitoring = enable_monitoring
        self._monitor_callback: Optional[Callable] = None
        
        # シャットダウン管理
        self._shutdown_event = threading.Event()
        
        logger.info(f"Pipeline initialized: batch_size={self._config['processing']['batch_size']}, "
                   f"max_workers={self._config['processing']['max_workers']}")
    
    def set_monitor_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """監視コールバックを設定"""
        self._monitor_callback = callback
    
    def _emit_monitor_event(self, event_type: str, **kwargs):
        """監視イベントを発行"""
        if self._enable_monitoring and self._monitor_callback:
            event = {
                'type': event_type,
                'timestamp': time.time(),
                **kwargs
            }
            try:
                self._monitor_callback(event)
            except Exception as e:
                logger.warning(f"Monitor callback error: {e}")
    
    def _check_shutdown(self):
        """シャットダウン要求をチェック"""
        if self._shutdown_event.is_set():
            raise PipelineShutdownError("Pipeline shutdown requested")
    
    def _apply_plugin(self, plugin_name: str, data: Any, default_result: Any = None) -> Any:
        """プラグインを適用"""
        if plugin_name in self._plugins:
            try:
                return self._plugins[plugin_name](data)
            except Exception as e:
                logger.warning(f"Plugin {plugin_name} failed: {e}")
                return default_result if default_result is not None else data
        return default_result if default_result is not None else data
    
    def _batch_data(self, data: List[Any], batch_size: int) -> List[List[Any]]:
        """データをバッチに分割"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _emit_progress(self, callback: Optional[Callable], stage: str, current: int, total: int):
        """進捗情報を発行"""
        if callback:
            try:
                callback(stage, current, total)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def process_invention_batch(
        self,
        inventions: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        continue_on_error: bool = False
    ) -> List[Dict[str, Any]]:
        """
        発明要約のバッチ処理
        
        Args:
            inventions: 発明データのリスト
            progress_callback: 進捗コールバック
            continue_on_error: エラー時に継続するか
            
        Returns:
            処理結果のリスト
        """
        logger.info(f"Processing invention batch: {len(inventions)} items")
        self._check_shutdown()
        
        # 監視イベント発行
        self._emit_monitor_event('batch_start', stage=ProcessingStages.INVENTION_SUMMARY, count=len(inventions))
        
        # プリプロセッサープラグイン適用
        inventions = self._apply_plugin('preprocessor', inventions, inventions)
        
        results = []
        batch_size = self._config['processing']['batch_size']
        
        # バッチ処理
        batches = self._batch_data(inventions, batch_size)
        total_processed = 0
        
        for batch_idx, batch in enumerate(batches):
            self._check_shutdown()
            
            try:
                # プロンプト生成
                prompts = []
                valid_indices = []
                
                for i, invention in enumerate(batch):
                    try:
                        prompt = create_invention_prompt(invention)
                        prompts.append(prompt)
                        valid_indices.append(i)
                    except Exception as e:
                        logger.warning(f"Prompt creation failed for invention {batch_idx * batch_size + i}: {e}")
                        if continue_on_error:
                            results.append({
                                'status': ProcessingStatus.FAILED,
                                'error': str(e),
                                'error_type': type(e).__name__,
                                'original_data': invention
                            })
                            self._stats.increment('failed_processed')
                            self._stats.add_error(type(e).__name__)
                        else:
                            raise PipelineProcessingError(f"Invention prompt creation failed: {e}")
                
                if prompts:
                    # LLM処理実行
                    start_time = time.time()
                    llm_results = self._client.batch_completion(
                        prompts, 
                        continue_on_error=continue_on_error
                    )
                    processing_time = time.time() - start_time
                    
                    # 結果処理
                    for i, result in enumerate(llm_results):
                        original_idx = valid_indices[i] if i < len(valid_indices) else i
                        original_data = batch[original_idx] if original_idx < len(batch) else {}
                        
                        if 'error' in result:
                            if continue_on_error:
                                results.append({
                                    'status': ProcessingStatus.FAILED,
                                    'error': result.get('error', 'Unknown error'),
                                    'error_type': result.get('error_type', 'Unknown'),
                                    'original_data': original_data
                                })
                                self._stats.increment('failed_processed')
                                self._stats.add_error(result.get('error_type', 'Unknown'))
                            else:
                                raise PipelineProcessingError(f"LLM processing failed: {result['error']}")
                        else:
                            results.append({
                                'status': ProcessingStatus.SUCCESS,
                                'summary': result.get('content', ''),
                                'processing_time': processing_time / len(llm_results),
                                'original_data': original_data
                            })
                            self._stats.increment('successful_processed')
                            self._stats.increment('invention_summaries')
                
                # 進捗更新
                total_processed += len(batch)
                self._emit_progress(progress_callback, ProcessingStages.INVENTION_SUMMARY, 
                                  total_processed, len(inventions))
                
            except Exception as e:
                if continue_on_error:
                    logger.error(f"Batch processing failed: {e}")
                    # バッチ全体を失敗として記録
                    for invention in batch:
                        results.append({
                            'status': ProcessingStatus.FAILED,
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'original_data': invention
                        })
                        self._stats.increment('failed_processed')
                        self._stats.add_error(type(e).__name__)
                else:
                    raise
        
        # 統計更新
        self._stats.increment('total_processed', len(inventions))
        
        # ポストプロセッサープラグイン適用
        results = self._apply_plugin('postprocessor', results, results)
        
        # 監視イベント発行
        self._emit_monitor_event('batch_complete', stage=ProcessingStages.INVENTION_SUMMARY, count=len(results))
        
        logger.info(f"Invention batch processing completed: {len(results)} results")
        return results
    
    def process_patent_batch(
        self,
        patents: List[Dict[str, Any]],
        invention_summary: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        concurrent: bool = False,
        continue_on_error: bool = False
    ) -> List[Dict[str, Any]]:
        """
        特許要約のバッチ処理
        
        Args:
            patents: 特許データのリスト
            invention_summary: 発明要約
            progress_callback: 進捗コールバック
            concurrent: 並行処理を使用するか
            continue_on_error: エラー時に継続するか
            
        Returns:
            処理結果のリスト
        """
        logger.info(f"Processing patent batch: {len(patents)} items, concurrent={concurrent}")
        self._check_shutdown()
        
        # 監視イベント発行
        self._emit_monitor_event('batch_start', stage=ProcessingStages.PATENT_SUMMARIES, count=len(patents))
        
        # プリプロセッサープラグイン適用
        patents = self._apply_plugin('preprocessor', patents, patents)
        
        results = []
        batch_size = self._config['processing']['batch_size']
        
        # バッチ処理
        batches = self._batch_data(patents, batch_size)
        total_processed = 0
        
        for batch_idx, batch in enumerate(batches):
            self._check_shutdown()
            
            try:
                # プロンプト生成
                prompts = []
                valid_indices = []
                
                for i, patent in enumerate(batch):
                    try:
                        prompt = create_patent_prompt(patent, invention_summary)
                        prompts.append(prompt)
                        valid_indices.append(i)
                    except Exception as e:
                        logger.warning(f"Patent prompt creation failed for {patent.get('publication_number', 'unknown')}: {e}")
                        if continue_on_error:
                            results.append({
                                'status': ProcessingStatus.FAILED,
                                'publication_number': patent.get('publication_number', 'unknown'),
                                'error': str(e),
                                'error_type': type(e).__name__,
                                'original_data': patent
                            })
                            self._stats.increment('failed_processed')
                            self._stats.add_error(type(e).__name__)
                        else:
                            raise PipelineProcessingError(f"Patent prompt creation failed: {e}")
                
                if prompts:
                    # LLM処理実行
                    start_time = time.time()
                    if concurrent and len(prompts) > 1:
                        llm_results = self._client.concurrent_batch_completion(prompts)
                    else:
                        llm_results = self._client.batch_completion(
                            prompts,
                            continue_on_error=continue_on_error
                        )
                    processing_time = time.time() - start_time
                    
                    # 結果処理
                    for i, result in enumerate(llm_results):
                        original_idx = valid_indices[i] if i < len(valid_indices) else i
                        original_data = batch[original_idx] if original_idx < len(batch) else {}
                        
                        if 'error' in result:
                            if continue_on_error:
                                results.append({
                                    'status': ProcessingStatus.FAILED,
                                    'publication_number': original_data.get('publication_number', 'unknown'),
                                    'error': result.get('error', 'Unknown error'),
                                    'error_type': result.get('error_type', 'Unknown'),
                                    'original_data': original_data
                                })
                                self._stats.increment('failed_processed')
                                self._stats.add_error(result.get('error_type', 'Unknown'))
                            else:
                                raise PipelineProcessingError(f"Patent LLM processing failed: {result['error']}")
                        else:
                            # 欠損データがある場合はpartialステータス
                            status = ProcessingStatus.SUCCESS
                            if not original_data.get('claim_1') or not original_data.get('abstract'):
                                status = ProcessingStatus.PARTIAL
                            
                            results.append({
                                'status': status,
                                'publication_number': original_data.get('publication_number', 'unknown'),
                                'summary': result.get('content', ''),
                                'processing_time': processing_time / len(llm_results),
                                'original_data': original_data
                            })
                            self._stats.increment('successful_processed')
                            self._stats.increment('patent_summaries')
                
                # 進捗更新
                total_processed += len(batch)
                self._emit_progress(progress_callback, ProcessingStages.PATENT_SUMMARIES,
                                  total_processed, len(patents))
                
            except Exception as e:
                if continue_on_error:
                    logger.error(f"Patent batch processing failed: {e}")
                    # バッチ全体を失敗として記録
                    for patent in batch:
                        results.append({
                            'status': ProcessingStatus.FAILED,
                            'publication_number': patent.get('publication_number', 'unknown'),
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'original_data': patent
                        })
                        self._stats.increment('failed_processed')
                        self._stats.add_error(type(e).__name__)
                else:
                    raise
        
        # 統計更新
        self._stats.increment('total_processed', len(patents))
        
        # ポストプロセッサープラグイン適用
        results = self._apply_plugin('postprocessor', results, results)
        
        # 監視イベント発行
        self._emit_monitor_event('batch_complete', stage=ProcessingStages.PATENT_SUMMARIES, count=len(results))
        
        logger.info(f"Patent batch processing completed: {len(results)} results")
        return results
    
    def process_classification_batch(
        self,
        classification_pairs: List[Dict[str, str]],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        validate_json: bool = True,
        continue_on_error: bool = False
    ) -> List[Dict[str, Any]]:
        """
        二値分類のバッチ処理
        
        Args:
            classification_pairs: 分類対象ペアのリスト
            progress_callback: 進捗コールバック
            validate_json: JSON応答を検証するか
            continue_on_error: エラー時に継続するか
            
        Returns:
            分類結果のリスト
        """
        logger.info(f"Processing classification batch: {len(classification_pairs)} pairs")
        self._check_shutdown()
        
        # 監視イベント発行
        self._emit_monitor_event('batch_start', stage=ProcessingStages.CLASSIFICATION, count=len(classification_pairs))
        
        results = []
        batch_size = self._config['processing']['batch_size']
        
        # バッチ処理
        batches = self._batch_data(classification_pairs, batch_size)
        total_processed = 0
        
        for batch_idx, batch in enumerate(batches):
            self._check_shutdown()
            
            try:
                # プロンプト生成
                prompts = []
                valid_indices = []
                
                for i, pair in enumerate(batch):
                    try:
                        prompt = create_classification_prompt(
                            pair['invention_summary'],
                            pair['patent_summary']
                        )
                        prompts.append(prompt)
                        valid_indices.append(i)
                    except Exception as e:
                        logger.warning(f"Classification prompt creation failed for pair {batch_idx * batch_size + i}: {e}")
                        if continue_on_error:
                            results.append({
                                'status': ProcessingStatus.FAILED,
                                'error': str(e),
                                'error_type': type(e).__name__,
                                'original_data': pair
                            })
                            self._stats.increment('failed_processed')
                            self._stats.add_error(type(e).__name__)
                        else:
                            raise PipelineProcessingError(f"Classification prompt creation failed: {e}")
                
                if prompts:
                    # LLM処理実行（分類はJSONモード）
                    start_time = time.time()
                    llm_results = self._client.batch_completion(
                        prompts,
                        response_format='json',
                        continue_on_error=continue_on_error
                    )
                    processing_time = time.time() - start_time
                    
                    # 結果処理
                    for i, result in enumerate(llm_results):
                        original_idx = valid_indices[i] if i < len(valid_indices) else i
                        original_data = batch[original_idx] if original_idx < len(batch) else {}
                        
                        if 'error' in result:
                            if continue_on_error:
                                results.append({
                                    'status': ProcessingStatus.FAILED,
                                    'error': result.get('error', 'Unknown error'),
                                    'error_type': result.get('error_type', 'Unknown'),
                                    'original_data': original_data
                                })
                                self._stats.increment('failed_processed')
                                self._stats.add_error(result.get('error_type', 'Unknown'))
                            else:
                                raise PipelineProcessingError(f"Classification LLM processing failed: {result['error']}")
                        else:
                            try:
                                # JSON解析
                                if isinstance(result, dict) and 'content' in result:
                                    if isinstance(result['content'], str):
                                        classification_result = json.loads(result['content'])
                                    else:
                                        classification_result = result['content']
                                else:
                                    classification_result = result
                                
                                # JSON検証
                                if validate_json:
                                    self._validate_classification_result(classification_result)
                                
                                # 結果追加
                                final_result = {
                                    'status': ProcessingStatus.SUCCESS,
                                    'decision': classification_result['decision'],
                                    'confidence': float(classification_result['confidence']),
                                    'hit_reason_1': classification_result.get('hit_reason_1', ''),
                                    'hit_src_1': classification_result.get('hit_src_1', ''),
                                    'hit_reason_2': classification_result.get('hit_reason_2', ''),
                                    'hit_src_2': classification_result.get('hit_src_2', ''),
                                    'processing_time': processing_time / len(llm_results),
                                    'original_data': original_data
                                }
                                results.append(final_result)
                                self._stats.increment('successful_processed')
                                self._stats.increment('classifications')
                                
                            except (json.JSONDecodeError, KeyError, ValueError) as e:
                                if continue_on_error:
                                    results.append({
                                        'status': ProcessingStatus.FAILED,
                                        'validation_error': str(e),
                                        'error_type': type(e).__name__,
                                        'raw_result': result,
                                        'original_data': original_data
                                    })
                                    self._stats.increment('failed_processed')
                                    self._stats.add_error(type(e).__name__)
                                else:
                                    raise PipelineProcessingError(f"Classification result validation failed: {e}")
                
                # 進捗更新
                total_processed += len(batch)
                self._emit_progress(progress_callback, ProcessingStages.CLASSIFICATION,
                                  total_processed, len(classification_pairs))
                
            except Exception as e:
                if continue_on_error:
                    logger.error(f"Classification batch processing failed: {e}")
                    # バッチ全体を失敗として記録
                    for pair in batch:
                        results.append({
                            'status': ProcessingStatus.FAILED,
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'original_data': pair
                        })
                        self._stats.increment('failed_processed')
                        self._stats.add_error(type(e).__name__)
                else:
                    raise
        
        # 統計更新
        self._stats.increment('total_processed', len(classification_pairs))
        
        # 監視イベント発行
        self._emit_monitor_event('batch_complete', stage=ProcessingStages.CLASSIFICATION, count=len(results))
        
        logger.info(f"Classification batch processing completed: {len(results)} results")
        return results
    
    def _validate_classification_result(self, result: Dict[str, Any]):
        """分類結果のJSON構造を検証"""
        required_fields = ['decision', 'confidence']
        
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # decision値の検証
        if result['decision'] not in ['hit', 'miss']:
            raise ValueError(f"Invalid decision value: {result['decision']}")
        
        # confidence値の検証
        confidence = result['confidence']
        if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"Invalid confidence value: {confidence}")
    
    def process_full_pipeline(
        self,
        invention_data: Dict[str, Any],
        patent_data: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        continue_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        完全なパイプライン処理
        
        Args:
            invention_data: 発明データ
            patent_data: 特許データリスト
            progress_callback: 進捗コールバック
            continue_on_error: エラー時に継続するか
            
        Returns:
            統合処理結果
        """
        logger.info(f"Starting full pipeline processing: 1 invention, {len(patent_data)} patents")
        self._stats.start_processing()
        self._check_shutdown()
        
        try:
            # Step 1: 発明要約
            invention_result = self.process_invention_batch(
                [invention_data],
                progress_callback=progress_callback,
                continue_on_error=continue_on_error
            )
            
            if not invention_result or invention_result[0]['status'] != ProcessingStatus.SUCCESS:
                raise PipelineProcessingError("Invention summarization failed")
            
            invention_summary = invention_result[0]['summary']
            
            # Step 2: 特許要約
            patent_results = self.process_patent_batch(
                patent_data,
                invention_summary,
                progress_callback=progress_callback,
                concurrent=True,
                continue_on_error=continue_on_error
            )
            
            # 成功した特許要約のみで分類を実行
            successful_patents = [r for r in patent_results if r['status'] == ProcessingStatus.SUCCESS]
            
            if not successful_patents:
                logger.warning("No successful patent summaries for classification")
                classification_results = []
            else:
                # Step 3: 二値分類
                classification_pairs = [
                    {
                        'invention_summary': invention_summary,
                        'patent_summary': patent_result['summary']
                    }
                    for patent_result in successful_patents
                ]
                
                classification_results = self.process_classification_batch(
                    classification_pairs,
                    progress_callback=progress_callback,
                    continue_on_error=continue_on_error
                )
                
                # 分類結果を特許結果にマージ
                for i, patent_result in enumerate(successful_patents):
                    if i < len(classification_results):
                        classification_result = classification_results[i]
                        patent_result.update({
                            'decision': classification_result.get('decision'),
                            'confidence': classification_result.get('confidence'),
                            'hit_reason_1': classification_result.get('hit_reason_1'),
                            'hit_src_1': classification_result.get('hit_src_1'),
                            'hit_reason_2': classification_result.get('hit_reason_2'),
                            'hit_src_2': classification_result.get('hit_src_2'),
                        })
            
            # 最終結果
            final_result = {
                'invention_summary': invention_summary,
                'patent_results': patent_results,
                'processing_summary': {
                    'total_patents': len(patent_data),
                    'successful_summaries': len(successful_patents),
                    'successful_classifications': len([r for r in classification_results if r.get('status') == ProcessingStatus.SUCCESS]),
                    'hit_count': len([r for r in patent_results if r.get('decision') == 'hit']),
                    'miss_count': len([r for r in patent_results if r.get('decision') == 'miss'])
                }
            }
            
            self._stats.end_processing()
            
            logger.info("Full pipeline processing completed successfully")
            return final_result
            
        except Exception as e:
            self._stats.end_processing()
            logger.error(f"Full pipeline processing failed: {e}")
            if continue_on_error:
                return {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'partial_results': locals().get('final_result', {})
                }
            else:
                raise
    
    def get_stats(self) -> Dict[str, Any]:
        """基本統計情報を取得"""
        return self._stats.get_stats()
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """詳細統計情報を取得"""
        stats = self._stats.get_stats()
        
        # LLMクライアントの統計も含める
        try:
            llm_stats = self._client.get_usage_stats()
            stats['token_usage'] = llm_stats
        except Exception as e:
            logger.warning(f"Failed to get LLM client stats: {e}")
        
        # エラー率計算
        total = stats['total_processed']
        if total > 0:
            stats['error_rates'] = {
                'overall_error_rate': stats['failed_processed'] / total,
                'success_rate': stats['successful_processed'] / total
            }
        
        # 処理時間統計
        if stats['processing_time'] > 0:
            stats['processing_times'] = {
                'total_time': stats['processing_time'],
                'average_per_item': stats['average_processing_time']
            }
        
        return stats
    
    def shutdown(self):
        """パイプラインを正常にシャットダウン"""
        logger.info("Initiating pipeline shutdown")
        self._shutdown_event.set()
        
        # LLMクライアントもシャットダウン
        try:
            self._client.shutdown()
        except Exception as e:
            logger.warning(f"LLM client shutdown error: {e}")
        
        logger.info("Pipeline shutdown completed")
    
    def __del__(self):
        """デストラクタ"""
        if hasattr(self, '_shutdown_event') and not self._shutdown_event.is_set():
            self.shutdown()


# メイン実行部（テスト用）
if __name__ == "__main__":
    # 簡単なテスト
    try:
        pipeline = LLMPipeline()
        
        # テストデータ
        test_invention = {
            'title': '液体分離設備の運転データ予測システム',
            'problem': '性能劣化の予測困難',
            'solution': '機械学習アルゴリズム'
        }
        
        test_patents = [
            {
                'publication_number': 'JP2025-100001A',
                'title': '膜分離装置の監視システム',
                'claim_1': '運転データを収集する手段を備えた装置',
                'abstract': '膜分離装置の性能監視システム'
            }
        ]
        
        # 統合処理テスト
        results = pipeline.process_full_pipeline(test_invention, test_patents)
        
        print("Pipeline test completed successfully!")
        print(f"Invention summary: {results.get('invention_summary', 'N/A')[:50]}...")
        print(f"Patent results: {len(results.get('patent_results', []))}")
        
        # 統計情報表示
        stats = pipeline.get_stats()
        print(f"Processing stats: {stats}")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        logger.error(f"Pipeline test execution failed: {e}")