"""
screener.py - 特許スクリーニング統合機能

特許分析システム全体の統合実行を行う。全コンポーネント（正規化、抽出、
LLM処理、ソート、出力）を組み合わせて完全なワークフローを実現する。

Code-reviewerの推奨に基づくセキュリティ・パフォーマンス・品質重視の実装。
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from loguru import logger
import yaml

# 各コンポーネントをインポート
try:
    from ..core.extract import extract_with_fallback
    from ..core.export import export_to_csv, export_to_jsonl
    from ..llm.client import LLMClient
    from ..llm.pipeline import LLMPipeline
    from ..ranking.sorter import PatentSorter
except ImportError:
    # 絶対インポートでフォールバック
    from src.core.extract import extract_with_fallback
    from src.core.export import export_to_csv, export_to_jsonl
    from src.llm.client import LLMClient
    from src.llm.pipeline import LLMPipeline
    from src.ranking.sorter import PatentSorter

# エラー階層（Code-reviewer推奨による改善）
class ScreenerError(Exception):
    """スクリーナーのベース例外"""
    pass

class ScreenerConfigError(ScreenerError):
    """設定関連エラー"""
    pass

class ScreenerInputError(ScreenerError):
    """入力データエラー"""
    pass

class ScreenerProcessingError(ScreenerError):
    """処理エラー"""
    pass

class ScreenerOutputError(ScreenerError):
    """出力エラー"""
    pass

# 定数クラス（Code-reviewer推奨による整理）
class ScreenerConstants:
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_MAX_WORKERS = 5
    MAX_PATENTS_PER_RUN = 10000
    DEFAULT_OUTPUT_CSV = "archive/outputs/attention_patents.csv"
    DEFAULT_OUTPUT_JSONL = "archive/outputs/details.jsonl"

class ProcessingLimits:
    MAX_PROCESSING_TIME = 1800  # 30分
    WARN_PROCESSING_TIME = 300  # 5分
    MAX_FILE_SIZE_MB = 100
    MAX_MEMORY_INCREASE_MB = 500

class ValidationConstants:
    REQUIRED_INVENTION_FIELDS = ['title', 'problem', 'solution']
    REQUIRED_PATENT_FIELDS = ['publication_number', 'title', 'abstract', 'claims']
    SUPPORTED_FILE_EXTENSIONS = {'.json', '.jsonl'}


def load_screener_config() -> Dict[str, Any]:
    """スクリーナー用設定ファイルを読み込む"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Config file parsing error: {e}")
        return {}


def validate_screener_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    スクリーナー設定の検証・サニタイズ（Code-reviewer推奨による改善）
    
    Args:
        config: 設定辞書
        
    Returns:
        検証済み設定辞書
        
    Raises:
        ScreenerConfigError: 設定が無効な場合
    """
    try:
        validated_config = {}
        
        # LLM設定の検証
        llm_config = config.get('llm', {})
        
        # モデルの検証
        model = llm_config.get('model', 'gpt-4o-mini')
        allowed_models = ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo']
        if model not in allowed_models:
            raise ScreenerConfigError(f"Invalid LLM model: {model}. Allowed: {allowed_models}")
        
        # 温度の検証
        temperature = float(llm_config.get('temperature', 0.0))
        if temperature < 0 or temperature > 2:
            raise ScreenerConfigError(f"Invalid temperature: {temperature}. Must be 0-2")
        
        # 最大トークン数の検証
        max_tokens = int(llm_config.get('max_tokens', 320))
        if max_tokens < 1 or max_tokens > 4096:
            raise ScreenerConfigError(f"Invalid max_tokens: {max_tokens}. Must be 1-4096")
        
        # タイムアウトの検証
        timeout = float(llm_config.get('timeout', 30))
        if timeout <= 0 or timeout > 300:
            raise ScreenerConfigError(f"Invalid timeout: {timeout}. Must be 0-300 seconds")
        
        validated_config['llm'] = {
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'timeout': timeout
        }
        
        # 処理設定の検証
        processing_config = config.get('processing', {})
        batch_size = processing_config.get('batch_size', ScreenerConstants.DEFAULT_BATCH_SIZE)
        max_workers = processing_config.get('max_workers', ScreenerConstants.DEFAULT_MAX_WORKERS)
        
        if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 100:
            raise ScreenerConfigError(f"Invalid batch_size: {batch_size}")
        if not isinstance(max_workers, int) or max_workers < 1 or max_workers > 20:
            raise ScreenerConfigError(f"Invalid max_workers: {max_workers}")
            
        validated_config['processing'] = {
            'batch_size': batch_size,
            'max_workers': max_workers
        }
        
        # ランキング設定の検証
        ranking_config = config.get('ranking', {})
        
        # ランキング方法の検証
        method = ranking_config.get('method', 'llm_only')
        allowed_methods = ['llm_only']
        if method not in allowed_methods:
            raise ScreenerConfigError(f"Invalid ranking method: {method}. Allowed: {allowed_methods}")
        
        # タイブレーカーの検証
        tiebreaker = ranking_config.get('tiebreaker', 'pub_number')
        allowed_tiebreakers = ['pub_number', 'pub_date', 'title']
        if tiebreaker not in allowed_tiebreakers:
            raise ScreenerConfigError(f"Invalid tiebreaker: {tiebreaker}. Allowed: {allowed_tiebreakers}")
        
        validated_config['ranking'] = {
            'method': method,
            'tiebreaker': tiebreaker
        }
        
        # 入出力設定の検証
        io_config = config.get('io', {})
        validated_config['io'] = {
            'out_csv': io_config.get('out_csv', ScreenerConstants.DEFAULT_OUTPUT_CSV),
            'out_jsonl': io_config.get('out_jsonl', ScreenerConstants.DEFAULT_OUTPUT_JSONL),
            'log_dir': io_config.get('log_dir', 'archive/logs')
        }
        
        # 抽出設定の検証
        extraction_config = config.get('extraction', {})
        validated_config['extraction'] = {
            'max_abstract_length': int(extraction_config.get('max_abstract_length', 500)),
            'include_dependent_claims': bool(extraction_config.get('include_dependent_claims', False))
        }
        
        return validated_config
        
    except (ValueError, TypeError) as e:
        raise ScreenerConfigError(f"Config validation error: {e}")


def validate_input_file(file_path: Union[str, Path], file_type: str) -> Path:
    """
    入力ファイルの検証（Code-reviewer推奨によるセキュリティ強化）
    
    Args:
        file_path: ファイルパス
        file_type: ファイルタイプ（'invention', 'patents'）
        
    Returns:
        検証済みファイルパス
        
    Raises:
        ScreenerInputError: ファイルが無効な場合
    """
    path = Path(file_path)
    
    # ファイル存在確認
    if not path.exists():
        raise ScreenerInputError(f"{file_type} file not found: {path}")
    
    # ファイルサイズ確認
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > ProcessingLimits.MAX_FILE_SIZE_MB:
        raise ScreenerInputError(f"{file_type} file too large: {file_size_mb:.1f}MB > {ProcessingLimits.MAX_FILE_SIZE_MB}MB")
    
    # 拡張子確認
    if path.suffix not in ValidationConstants.SUPPORTED_FILE_EXTENSIONS:
        raise ScreenerInputError(f"Unsupported file extension: {path.suffix}")
    
    # パストラバーサル対策
    resolved_path = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        resolved_path.relative_to(cwd)
    except ValueError:
        # 相対パス以外も許可するが、危険なパスはチェック
        if any(part.startswith('..') for part in resolved_path.parts):
            raise ScreenerInputError(f"Potentially unsafe path: {path}")
    
    return resolved_path


def load_invention_data(file_path: Path) -> Dict[str, Any]:
    """
    発明アイデアデータを読み込む
    
    Args:
        file_path: 発明アイデアファイルのパス
        
    Returns:
        発明アイデアデータ
        
    Raises:
        ScreenerInputError: 読み込みまたは検証に失敗した場合
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            invention_data = json.load(f)
        
        # 必須フィールドの確認
        for field in ValidationConstants.REQUIRED_INVENTION_FIELDS:
            if field not in invention_data:
                raise ScreenerInputError(f"Missing required invention field: {field}")
        
        # データサニタイズ
        sanitized_data = {}
        for key, value in invention_data.items():
            if isinstance(value, str):
                # 制御文字削除
                sanitized_value = ''.join(char for char in value if char.isprintable() or char.isspace())
                sanitized_data[key] = sanitized_value
            else:
                sanitized_data[key] = value
        
        return sanitized_data
        
    except json.JSONDecodeError as e:
        raise ScreenerInputError(f"Invalid JSON in invention file: {e}")
    except Exception as e:
        raise ScreenerInputError(f"Error loading invention data: {e}")


def load_patents_data(file_path: Path) -> List[Dict[str, Any]]:
    """
    特許データを読み込む（JSONL形式）
    
    Args:
        file_path: 特許データファイルのパス
        
    Returns:
        特許データのリスト
        
    Raises:
        ScreenerInputError: 読み込みまたは検証に失敗した場合
    """
    patents = []
    invalid_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    patent_data = json.loads(line)
                    
                    # 基本検証（必須フィールドの存在確認は緩く）
                    if not isinstance(patent_data, dict):
                        invalid_count += 1
                        logger.warning(f"Line {line_num}: Patent data must be a dictionary")
                        continue
                    
                    # 特許番号の存在確認（最低限必要）
                    if 'publication_number' not in patent_data:
                        invalid_count += 1
                        logger.warning(f"Line {line_num}: Missing publication_number")
                        continue
                    
                    patents.append(patent_data)
                    
                except json.JSONDecodeError as e:
                    invalid_count += 1
                    logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                
                # データ量制限チェック
                if len(patents) >= ScreenerConstants.MAX_PATENTS_PER_RUN:
                    logger.warning(f"Patent limit reached: {ScreenerConstants.MAX_PATENTS_PER_RUN}")
                    break
        
        if len(patents) == 0:
            raise ScreenerInputError("No valid patents found in file")
        
        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid patent entries")
        
        return patents
        
    except Exception as e:
        if isinstance(e, ScreenerInputError):
            raise
        raise ScreenerInputError(f"Error loading patents data: {e}")


class ProcessingStatistics:
    """
    処理統計管理クラス（スレッドセーフ）
    Code-reviewer推奨によるスレッドセーフ統計管理
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._stats = {
            'total_patents_processed': 0,
            'successful_extractions': 0,
            'successful_classifications': 0,
            'processing_time': 0.0,
            'average_time_per_patent': 0.0,
            'extraction_errors': 0,
            'llm_errors': 0,
            'sorting_time': 0.0,
            'export_time': 0.0,
            'total_hits': 0,
            'total_misses': 0
        }
    
    def update_processing_stats(
        self, 
        patents_processed: int = 0,
        successful_extractions: int = 0,
        successful_classifications: int = 0,
        processing_time: float = 0.0,
        extraction_errors: int = 0,
        llm_errors: int = 0,
        sorting_time: float = 0.0,
        export_time: float = 0.0,
        hits: int = 0,
        misses: int = 0
    ) -> None:
        """処理統計を更新"""
        with self._lock:
            self._stats['total_patents_processed'] += patents_processed
            self._stats['successful_extractions'] += successful_extractions
            self._stats['successful_classifications'] += successful_classifications
            self._stats['processing_time'] += processing_time
            self._stats['extraction_errors'] += extraction_errors
            self._stats['llm_errors'] += llm_errors
            self._stats['sorting_time'] += sorting_time
            self._stats['export_time'] += export_time
            self._stats['total_hits'] += hits
            self._stats['total_misses'] += misses
            
            # 平均時間計算
            if self._stats['total_patents_processed'] > 0:
                self._stats['average_time_per_patent'] = (
                    self._stats['processing_time'] / self._stats['total_patents_processed']
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        with self._lock:
            return self._stats.copy()
    
    def reset_stats(self) -> None:
        """統計情報をリセット"""
        with self._lock:
            self._stats = {
                'total_patents_processed': 0,
                'successful_extractions': 0,
                'successful_classifications': 0,
                'processing_time': 0.0,
                'average_time_per_patent': 0.0,
                'extraction_errors': 0,
                'llm_errors': 0,
                'sorting_time': 0.0,
                'export_time': 0.0,
                'total_hits': 0,
                'total_misses': 0
            }
            logger.debug("Processing statistics reset")


class PatentScreener:
    """
    特許スクリーニング統合機能
    
    特許分析システム全体のワークフローを統合管理する。
    各コンポーネント（抽出、LLM処理、ソート、出力）を組み合わせて
    完全な特許分析パイプラインを提供する。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        特許スクリーナーの初期化
        
        Args:
            config: 設定辞書（オプション、設定ファイルから自動取得）
            
        Raises:
            ScreenerConfigError: 設定が無効な場合
        """
        logger.debug("Initializing patent screener")
        
        # 設定読み込み・検証
        if config is None:
            config = load_screener_config()
        
        self._config = validate_screener_config(config)
        
        # コンポーネント初期化
        self._initialize_components()
        
        # 統計管理
        self._stats = ProcessingStatistics()
        
        # スレッドセーフティ
        self._lock = threading.RLock()
        
        logger.info(f"Patent screener initialized with model: {self._config['llm']['model']}")
    
    def _initialize_components(self) -> None:
        """各コンポーネントの初期化"""
        try:
            # 抽出機能（関数ベース）
            self._extract_function = extract_with_fallback
            
            # LLMクライアント・パイプライン（テスト用にAPI key不要の場合はスキップ）
            try:
                self._client = LLMClient(
                    max_retries=3,
                    max_workers=self._config['processing']['max_workers']
                )
                self._pipeline = LLMPipeline(config=self._config)
            except Exception as llm_error:
                # テスト環境でAPI keyがない場合は、モック用の属性を設定
                logger.warning(f"LLM components initialization failed (likely missing API key): {llm_error}")
                self._client = None
                self._pipeline = None
            
            # ソートコンポーネント
            self._sorter = PatentSorter(config=self._config)
            
            # エクスポート機能（関数ベース）
            self._export_csv_function = export_to_csv
            self._export_jsonl_function = export_to_jsonl
            
            logger.debug("All components initialized successfully")
            
        except Exception as e:
            raise ScreenerConfigError(f"Component initialization failed: {e}")
    
    def analyze(
        self,
        invention: Union[str, Path, Dict[str, Any]],
        patents: Union[str, Path, List[Dict[str, Any]]],
        output_csv: Optional[Union[str, Path]] = None,
        output_jsonl: Optional[Union[str, Path]] = None,
        continue_on_error: bool = False
    ) -> Dict[str, Any]:
        """
        特許分析の実行
        
        Args:
            invention: 発明アイデア（ファイルパスまたはデータ）
            patents: 特許データ（ファイルパスまたはデータリスト）
            output_csv: CSV出力先（オプション）
            output_jsonl: JSONL出力先（オプション）
            continue_on_error: エラー発生時に処理を続行するか
            
        Returns:
            処理結果の要約
            
        Raises:
            ScreenerInputError: 入力データが無効な場合
            ScreenerProcessingError: 処理中にエラーが発生した場合
            ScreenerOutputError: 出力処理でエラーが発生した場合
        """
        with self._lock:
            start_time = time.time()
            
            try:
                logger.info("Starting patent screening analysis")
                
                # 1. 入力データの読み込み・検証
                invention_data, patents_data = self._load_and_validate_inputs(invention, patents)
                
                # 2. データ抽出処理
                extracted_data = self._extract_patent_data(patents_data, continue_on_error)
                
                # 3. LLM処理（発明要約、特許要約、分類）
                classified_data = self._process_with_llm(invention_data, extracted_data, continue_on_error)
                
                # 4. ソート処理
                sorted_data = self._sort_results(classified_data)
                
                # 5. 出力処理
                self._export_results(sorted_data, output_csv, output_jsonl)
                
                # 6. 結果要約の生成
                processing_time = time.time() - start_time
                summary = self._generate_summary(sorted_data, processing_time)
                
                # 統計更新
                self._update_final_statistics(sorted_data, processing_time)
                
                logger.info(f"Patent screening completed in {processing_time:.2f}s")
                return summary
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Patent screening failed after {processing_time:.2f}s: {e}")
                raise ScreenerProcessingError(f"Analysis failed: {e}")
    
    def _load_and_validate_inputs(
        self, 
        invention: Union[str, Path, Dict[str, Any]], 
        patents: Union[str, Path, List[Dict[str, Any]]]
    ) -> tuple:
        """入力データの読み込みと検証"""
        # 発明アイデアの読み込み
        if isinstance(invention, (str, Path)):
            invention_file = validate_input_file(invention, 'invention')
            invention_data = load_invention_data(invention_file)
        else:
            invention_data = invention
        
        # 特許データの読み込み
        if isinstance(patents, (str, Path)):
            patents_file = validate_input_file(patents, 'patents')
            patents_data = load_patents_data(patents_file)
        else:
            patents_data = patents
        
        logger.info(f"Loaded invention data and {len(patents_data)} patents")
        return invention_data, patents_data
    
    def _extract_patent_data(self, patents_data: List[Dict[str, Any]], continue_on_error: bool) -> List[Dict[str, Any]]:
        """特許データの抽出処理"""
        logger.info("Extracting patent data")
        
        extracted_patents = []
        extraction_errors = 0
        
        for i, patent_data in enumerate(patents_data):
            try:
                extracted_patent = self._extract_function(patent_data)
                # 元の特許データをマージ（LLM処理に必要な情報を保持）
                merged_patent = {**patent_data, **extracted_patent}
                extracted_patents.append(merged_patent)
                
            except Exception as e:
                extraction_errors += 1
                logger.warning(f"Patent {i} extraction failed: {e}")
                
                if continue_on_error:
                    # エラーが発生した特許はスキップ
                    continue
                else:
                    raise ScreenerProcessingError(f"Patent extraction failed: {e}")
        
        # 統計更新
        self._stats.update_processing_stats(
            successful_extractions=len(extracted_patents),
            extraction_errors=extraction_errors
        )
        
        logger.info(f"Extracted data from {len(extracted_patents)} patents ({extraction_errors} errors)")
        return extracted_patents
    
    def _process_with_llm(
        self, 
        invention_data: Dict[str, Any], 
        extracted_data: List[Dict[str, Any]], 
        continue_on_error: bool
    ) -> List[Dict[str, Any]]:
        """LLM処理（発明要約、特許要約、分類）"""
        logger.info("Processing with LLM pipeline")
        
        try:
            # 発明要約の生成
            invention_summary = self._pipeline.process_invention_summary(invention_data)
            
            # 特許要約と分類の処理
            classified_results = []
            llm_errors = 0
            
            for patent_data in extracted_data:
                try:
                    # 特許要約
                    patent_summary = self._pipeline.process_patent_summary(patent_data)
                    
                    # 分類処理
                    classification_result = self._pipeline.process_classification(
                        invention_summary, patent_summary, patent_data
                    )
                    
                    classified_results.append(classification_result)
                    
                except Exception as e:
                    llm_errors += 1
                    logger.warning(f"LLM processing failed for {patent_data.get('publication_number', 'unknown')}: {e}")
                    
                    if continue_on_error:
                        # エラー時のデフォルト結果
                        default_result = {
                            **patent_data,
                            'decision': 'miss',
                            'confidence': 0.0,
                            'hit_reason_1': 'Processing error',
                            'hit_src_1': 'system',
                            'hit_reason_2': '',
                            'hit_src_2': '',
                            'processing_time': 0.0
                        }
                        classified_results.append(default_result)
                    else:
                        raise ScreenerProcessingError(f"LLM processing failed: {e}")
            
            # 統計更新
            self._stats.update_processing_stats(
                successful_classifications=len(classified_results) - llm_errors,
                llm_errors=llm_errors
            )
            
            logger.info(f"LLM processing completed for {len(classified_results)} patents ({llm_errors} errors)")
            return classified_results
            
        except Exception as e:
            if isinstance(e, ScreenerProcessingError):
                raise
            raise ScreenerProcessingError(f"LLM pipeline processing failed: {e}")
    
    def _sort_results(self, classified_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """結果のソート処理"""
        logger.info("Sorting results by LLM confidence")
        
        try:
            start_time = time.time()
            sorted_results = self._sorter.sort_patents(classified_data)
            sort_time = time.time() - start_time
            
            # 統計更新
            self._stats.update_processing_stats(sorting_time=sort_time)
            
            logger.info(f"Sorted {len(sorted_results)} results in {sort_time:.2f}s")
            return sorted_results
            
        except Exception as e:
            raise ScreenerProcessingError(f"Sorting failed: {e}")
    
    def _export_results(
        self, 
        sorted_data: List[Dict[str, Any]], 
        output_csv: Optional[Union[str, Path]], 
        output_jsonl: Optional[Union[str, Path]]
    ) -> None:
        """結果の出力処理"""
        logger.info("Exporting results")
        
        try:
            start_time = time.time()
            
            # 出力パスの決定
            csv_path = output_csv or self._config['io']['out_csv']
            jsonl_path = output_jsonl or self._config['io']['out_jsonl']
            
            # 出力実行
            if csv_path:
                self._export_csv_function(sorted_data, csv_path)
                logger.info(f"CSV exported to: {csv_path}")
            
            if jsonl_path:
                self._export_jsonl_function(sorted_data, jsonl_path)
                logger.info(f"JSONL exported to: {jsonl_path}")
            
            export_time = time.time() - start_time
            
            # 統計更新
            self._stats.update_processing_stats(export_time=export_time)
            
        except Exception as e:
            raise ScreenerOutputError(f"Export failed: {e}")
    
    def _generate_summary(self, sorted_data: List[Dict[str, Any]], processing_time: float) -> Dict[str, Any]:
        """処理結果の要約を生成"""
        hits = sum(1 for result in sorted_data if result.get('decision') == 'hit')
        misses = len(sorted_data) - hits
        
        summary = {
            'summary': 'Patent screening analysis completed successfully',
            'patents_processed': len(sorted_data),
            'total_hits': hits,
            'total_misses': misses,
            'processing_time': round(processing_time, 2),
            'average_time_per_patent': round(processing_time / len(sorted_data) if sorted_data else 0, 3),
            'hit_rate': round(hits / len(sorted_data) * 100 if sorted_data else 0, 1)
        }
        
        return summary
    
    def _update_final_statistics(self, sorted_data: List[Dict[str, Any]], processing_time: float) -> None:
        """最終統計の更新"""
        hits = sum(1 for result in sorted_data if result.get('decision') == 'hit')
        misses = len(sorted_data) - hits
        
        self._stats.update_processing_stats(
            patents_processed=len(sorted_data),
            processing_time=processing_time,
            hits=hits,
            misses=misses
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        処理統計情報を取得
        
        Returns:
            処理統計辞書
        """
        return self._stats.get_stats()
    
    def get_component_stats(self) -> Dict[str, Any]:
        """
        各コンポーネントの統計情報を取得
        
        Returns:
            コンポーネント統計辞書
        """
        return {
            'pipeline': self._pipeline.get_pipeline_stats() if self._pipeline and hasattr(self._pipeline, 'get_pipeline_stats') else {},
            'sorter': self._sorter.get_sort_stats()
        }
    
    def reset_stats(self) -> None:
        """すべての統計情報をリセット"""
        self._stats.reset_stats()
        
        # 各コンポーネントの統計もリセット
        if self._pipeline and hasattr(self._pipeline, 'reset_stats'):
            self._pipeline.reset_stats()
        if hasattr(self._sorter, 'reset_sort_stats'):
            self._sorter.reset_sort_stats()
        
        logger.debug("All statistics reset")
    
    def get_config(self) -> Dict[str, Any]:
        """
        現在の設定を取得
        
        Returns:
            設定辞書
        """
        with self._lock:
            return self._config.copy()


if __name__ == "__main__":
    # 簡単なテスト
    try:
        screener = PatentScreener()
        
        # 設定確認
        config = screener.get_config()
        print(f"Screener initialized with config: {config}")
        
        # 統計確認
        stats = screener.get_processing_stats()
        print(f"Initial stats: {stats}")
        
        print("Patent screener test completed successfully")
        
    except Exception as e:
        print(f"Test failed: {e}")
        logger.error(f"Test execution failed: {e}")