"""
export.py - CSV/JSONL出力機能

特許分析結果をCSVおよびJSONL形式で出力する。
Code-reviewerの指摘に基づき、セキュリティ・パフォーマンス・品質を重視した実装。
"""

import csv
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from loguru import logger
import threading
import yaml

# セキュリティエラー定義
class SecurityError(Exception):
    """セキュリティ関連エラー"""
    pass

# 定数クラス - Code-reviewer推奨による改善
class ExportConstants:
    # ファイルサイズ推定
    CSV_BYTES_PER_ROW = 500
    JSONL_BYTES_PER_ROW = 1024
    
    # 制限値
    MAX_FIELD_LENGTH = 10000
    MAX_PATENTS_COUNT = 100000
    BATCH_SIZE_DEFAULT = 1000
    MAX_VALUE_LENGTH = 50000
    
    # ディスク容量
    MIN_FREE_SPACE_MB = 10
    SAFETY_MARGIN_MULTIPLIER = 1.5
    
    # CSV列定義
    CSV_FIELDNAMES = [
        'rank', 'pub_number', 'title', 'assignee', 'pub_date',
        'decision', 'LLM_confidence', 'hit_reason_1', 'hit_src_1',
        'hit_reason_2', 'hit_src_2', 'url_hint'
    ]
    
    # 危険なパス
    DANGEROUS_PATHS = [
        '/etc', '/root', '/usr', '/sys', '/proc', '/dev',
        'c:\\windows', 'c:\\system32', 'c:\\program files',
        '/bin', '/sbin', '/boot', '/var/log'
    ]
    
    # 危険なファイル
    DANGEROUS_FILES = [
        'passwd', 'shadow', 'hosts', '.ssh', 'id_rsa', '.bashrc',
        'config.sys', 'autoexec.bat', 'boot.ini', 'sam'
    ]

# ファイルロック管理
_file_locks = {}
_lock_manager = threading.Lock()

# 事前コンパイル済み正規表現パターン - パフォーマンス改善
_NEWLINE_PATTERN = re.compile(r'[\r\n]+')
_WHITESPACE_PATTERN = re.compile(r'\s+')
_CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

def get_file_lock(path: Path) -> threading.Lock:
    """ファイルパスごとのロックを取得"""
    with _lock_manager:
        path_str = str(path.resolve())
        if path_str not in _file_locks:
            _file_locks[path_str] = threading.Lock()
        return _file_locks[path_str]


# 設定キャッシュ - Code-reviewer推奨によるパフォーマンス改善
from functools import lru_cache

@lru_cache(maxsize=1)
def load_config_cached(config_mtime: Optional[float] = None) -> Dict[str, Any]:
    """設定をキャッシュ付きで読み込み（更新時刻ベース）"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return get_default_config()
    except yaml.YAMLError as e:
        logger.error(f"Config file parsing error: {e}")
        return get_default_config()

def load_config() -> Dict[str, Any]:
    """設定ファイルを自動キャッシュ無効化付きで読み込む"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    try:
        mtime = os.path.getmtime(config_path)
        return load_config_cached(mtime)
    except OSError:
        return load_config_cached(None)


def get_default_config() -> Dict[str, Any]:
    """デフォルト設定を取得"""
    return {
        'io': {
            'out_csv': 'archive/outputs/attention_patents.csv',
            'out_jsonl': 'archive/outputs/details.jsonl'
        },
        'hit_reason': {
            'max_reasons': 2
        }
    }


def validate_output_path(output_path: Union[str, Path]) -> None:
    """
    出力パスのセキュリティ検証（Code-reviewer推奨による強化）
    
    Args:
        output_path: 検証する出力パス
        
    Raises:
        SecurityError: セキュリティ上の問題がある場合
    """
    path = Path(output_path).resolve()
    path_str = str(path).lower()
    
    # システムディレクトリへのアクセスを防ぐ
    for dangerous in ExportConstants.DANGEROUS_PATHS:
        if dangerous.lower() in path_str:
            raise SecurityError(f"Access to system directory denied: {path}")
    
    # 特定の危険なファイルへのアクセスを防ぐ
    for dangerous_file in ExportConstants.DANGEROUS_FILES:
        if dangerous_file in path_str:
            raise SecurityError(f"Access to sensitive file denied: {path}")
    
    # 強化されたパストラバーサル防止
    current_dir = Path.cwd().resolve()
    try:
        # 解決済みパスが現在のディレクトリ内であることを確認
        relative_path = path.relative_to(current_dir)
        if relative_path.parts and relative_path.parts[0] == '..':
            raise SecurityError(f"Path traversal attempt: {path}")
    except ValueError:
        # プロジェクト外への書き込みを許可するディレクトリ
        allowed_external = ['archive', 'outputs', 'temp', 'tmp']
        if not any(allowed in path_str for allowed in allowed_external):
            raise SecurityError(f"Writing outside project denied: {path}")
        logger.warning(f"Writing to external directory: {path}")
    
    # 簡易的なパストラバーサルチェック（追加）
    if '..' in str(output_path):
        raise SecurityError(f"Path traversal attempt detected: {output_path}")


def ensure_output_directory(output_path: Union[str, Path]) -> None:
    """
    出力ディレクトリが存在することを確認
    
    Args:
        output_path: 出力ファイルパス
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create directory {output_dir}: {e}")
            raise


def check_disk_space_smart(output_path: Path, estimated_size: int) -> None:
    """
    スマートディスク容量チェック（Code-reviewer推奨）
    
    Args:
        output_path: 出力パス
        estimated_size: 推定ファイルサイズ
        
    Raises:
        IOError: ディスク容量不足の場合
    """
    try:
        stat = shutil.disk_usage(output_path.parent)
        
        # 安全マージンを適用（1.5倍 + 100MBバッファ）
        safety_buffer = 100 * 1024 * 1024  # 100MB
        required_bytes = max(
            int(estimated_size * ExportConstants.SAFETY_MARGIN_MULTIPLIER),
            ExportConstants.MIN_FREE_SPACE_MB * 1024 * 1024
        ) + safety_buffer
        
        if stat.free < required_bytes:
            free_mb = stat.free / (1024 * 1024)
            required_mb = required_bytes / (1024 * 1024)
            raise IOError(
                f"Insufficient disk space. Required: {required_mb:.1f}MB, "
                f"Available: {free_mb:.1f}MB"
            )
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")


def sanitize_csv_value(value: Any) -> str:
    """
    CSV出力用に値をサニタイズ（Code-reviewer推奨による改善）
    
    Args:
        value: サニタイズする値
        
    Returns:
        サニタイズされた文字列
    """
    if value is None:
        return ""
    
    if not isinstance(value, str):
        value = str(value)
    
    # ReDoS攻撃防止のため入力長制限
    if len(value) > ExportConstants.MAX_VALUE_LENGTH:
        logger.warning(f"Value truncated from {len(value)} to {ExportConstants.MAX_VALUE_LENGTH} chars")
        value = value[:ExportConstants.MAX_VALUE_LENGTH] + "..."
    
    # 事前コンパイル済みパターンを使用（パフォーマンス改善）
    value = _NEWLINE_PATTERN.sub(' ', value)
    value = _WHITESPACE_PATTERN.sub(' ', value.strip())
    value = _CONTROL_CHARS_PATTERN.sub('', value)
    
    return value


def format_hit_reasons_for_csv(hit_reasons: List[Dict[str, str]]) -> Dict[str, str]:
    """
    CSV出力用にヒット理由をフォーマット
    
    Args:
        hit_reasons: ヒット理由のリスト
        
    Returns:
        フォーマットされたヒット理由辞書
    """
    formatted = {
        'hit_reason_1': '',
        'hit_src_1': '',
        'hit_reason_2': '',
        'hit_src_2': ''
    }
    
    # 最大2件のヒット理由を処理
    for i, reason in enumerate(hit_reasons[:2]):
        reason_key = f'hit_reason_{i + 1}'
        src_key = f'hit_src_{i + 1}'
        
        formatted[reason_key] = sanitize_csv_value(reason.get('quote', ''))
        formatted[src_key] = sanitize_csv_value(reason.get('source', ''))
    
    return formatted


def validate_patent_structure(patent: Dict[str, Any], index: int) -> None:
    """
    個別特許構造の詳細検証（Code-reviewer推奨）
    
    Args:
        patent: 特許データ辞書
        index: 特許のインデックス
        
    Raises:
        ValueError: データ構造が不正な場合
    """
    if not isinstance(patent, dict):
        raise ValueError(
            f"Patent at index {index} must be a dictionary, got {type(patent).__name__}. "
            f"Expected format: {{'pub_number': str, 'title': str, ...}}"
        )
    
    # 必須フィールドの確認
    required_fields = ['pub_number']
    missing_fields = [field for field in required_fields if field not in patent]
    
    if missing_fields:
        raise ValueError(
            f"Patent at index {index} missing required fields: {missing_fields}. "
            f"Available fields: {list(patent.keys())}"
        )
    
    # フィールド長制限チェック
    for key, value in patent.items():
        if isinstance(value, str) and len(value) > ExportConstants.MAX_FIELD_LENGTH:
            raise ValueError(
                f"Field '{key}' at patent {index} too long: {len(value)} chars. "
                f"Maximum allowed: {ExportConstants.MAX_FIELD_LENGTH}"
            )

def validate_input_data(patent_results: Any) -> None:
    """
    入力データの妥当性を検証（Code-reviewer推奨による強化）
    
    Args:
        patent_results: 検証する特許結果データ
        
    Raises:
        TypeError: データ型が不正な場合
        ValueError: データ構造が不正な場合
    """
    if patent_results is None:
        raise TypeError("Patent results cannot be None")
    
    if not isinstance(patent_results, list):
        raise TypeError(f"Patent results must be a list, got {type(patent_results)}")
    
    # 大量データ攻撃防止
    if len(patent_results) > ExportConstants.MAX_PATENTS_COUNT:
        raise ValueError(
            f"Too many patents: {len(patent_results)}. "
            f"Maximum allowed: {ExportConstants.MAX_PATENTS_COUNT}"
        )
    
    # 個別特許の詳細検証
    for i, patent in enumerate(patent_results):
        validate_patent_structure(patent, i)


def _prepare_csv_row(patent: Dict[str, Any]) -> Dict[str, str]:
    """
    特許データからCSV行を準備（Code-reviewer推奨による関数分解）
    
    Args:
        patent: 特許データ辞書
        
    Returns:
        CSV行データ辞書
    """
    # ヒット理由のフォーマット
    formatted_reasons = format_hit_reasons_for_csv(
        patent.get('hit_reasons', [])
    )
    
    # 行データの準備
    row = {
        'rank': patent.get('rank', ''),
        'pub_number': sanitize_csv_value(patent.get('pub_number', '')),
        'title': sanitize_csv_value(patent.get('title', '')),
        'assignee': sanitize_csv_value(patent.get('assignee', '')),
        'pub_date': patent.get('pub_date', ''),
        'decision': patent.get('decision', ''),
        'LLM_confidence': patent.get('LLM_confidence', ''),
        'url_hint': sanitize_csv_value(patent.get('url_hint', ''))
    }
    
    # ヒット理由を追加
    row.update(formatted_reasons)
    return row


def _write_csv_data(writer: csv.DictWriter, patent_results: List[Dict[str, Any]]) -> None:
    """
    CSVライターに特許データを書き込み（Code-reviewer推奨による関数分解）
    
    Args:
        writer: CSVライター
        patent_results: 特許結果のリスト
    """
    writer.writeheader()
    
    for patent in patent_results:
        row = _prepare_csv_row(patent)
        writer.writerow(row)


def export_to_csv(patent_results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    特許結果をCSV形式で出力
    
    Args:
        patent_results: 特許結果のリスト
        output_path: 出力先CSVファイルパス
        
    Raises:
        SecurityError: セキュリティ上の問題がある場合
        PermissionError: ファイル書き込み権限がない場合
        IOError: ディスク容量不足等のI/Oエラー
    """
    # 入力検証
    validate_input_data(patent_results)
    
    output_path = Path(output_path).resolve()
    
    # セキュリティ検証
    validate_output_path(output_path)
    ensure_output_directory(output_path)
    
    # ディスク容量チェック（改善されたサイズ推定）
    estimated_size = len(patent_results) * ExportConstants.CSV_BYTES_PER_ROW
    check_disk_space_smart(output_path, estimated_size)
    
    # ファイルロックを取得
    file_lock = get_file_lock(output_path)
    
    with file_lock:
        # アトミック書き込みのため一時ファイルを使用
        temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', dir=output_path.parent)
        
        try:
            with os.fdopen(temp_fd, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=ExportConstants.CSV_FIELDNAMES)
                _write_csv_data(writer, patent_results)
            
            # アトミックに本番ファイルに移動
            temp_path_obj = Path(temp_path)
            temp_path_obj.replace(output_path)
            
            logger.info(f"Successfully exported {len(patent_results)} patents to CSV: {output_path}")
            
        except Exception as e:
            # エラー時は一時ファイルを削除
            try:
                Path(temp_path).unlink()
            except:
                pass
            logger.error(f"Failed to export CSV: {e}")
            raise


def export_to_jsonl(patent_results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    特許結果をJSONL形式で出力
    
    Args:
        patent_results: 特許結果のリスト
        output_path: 出力先JSONLファイルパス
        
    Raises:
        SecurityError: セキュリティ上の問題がある場合
        ValueError: JSONシリアライズできない場合
    """
    # 入力検証
    validate_input_data(patent_results)
    
    # 空のリストの場合はファイルを作成しない
    if not patent_results:
        logger.debug("No patent results to export to JSONL")
        return
    
    output_path = Path(output_path).resolve()
    
    # セキュリティ検証
    validate_output_path(output_path)
    ensure_output_directory(output_path)
    
    # ディスク容量チェック（改善されたサイズ推定）
    estimated_size = len(patent_results) * ExportConstants.JSONL_BYTES_PER_ROW
    check_disk_space_smart(output_path, estimated_size)
    
    # ファイルロックを取得
    file_lock = get_file_lock(output_path)
    
    with file_lock:
        # アトミック書き込みのため一時ファイルを使用
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jsonl', dir=output_path.parent)
        
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as jsonl_file:
                for i, patent in enumerate(patent_results):
                    try:
                        # JSONシリアライズ可能性チェック
                        json_str = json.dumps(patent, ensure_ascii=False)
                        jsonl_file.write(json_str + '\n')
                    except (TypeError, ValueError) as e:
                        pub_number = patent.get('pub_number', f'index_{i}')
                        logger.error(f"JSON serialization error for {pub_number}: {e}")
                        raise ValueError(f"Cannot serialize patent data to JSON: {pub_number}")
            
            # アトミックに本番ファイルに移動
            temp_path_obj = Path(temp_path)
            temp_path_obj.replace(output_path)
            
            logger.info(f"Successfully exported {len(patent_results)} patents to JSONL: {output_path}")
            
        except Exception as e:
            # エラー時は一時ファイルを削除
            try:
                Path(temp_path).unlink()
            except:
                pass
            logger.error(f"Failed to export JSONL: {e}")
            raise


def batch_export(patent_results: List[Dict[str, Any]], 
                 csv_path: Union[str, Path], 
                 jsonl_path: Union[str, Path]) -> None:
    """
    CSV とJSONL形式で同時出力
    
    Args:
        patent_results: 特許結果のリスト
        csv_path: CSV出力パス
        jsonl_path: JSONL出力パス
    """
    logger.debug(f"Starting batch export: CSV={csv_path}, JSONL={jsonl_path}")
    
    # 両形式で出力
    export_to_csv(patent_results, csv_path)
    export_to_jsonl(patent_results, jsonl_path)
    
    logger.info("Batch export completed successfully")


def export_with_config(patent_results: List[Dict[str, Any]]) -> None:
    """
    設定ファイルの設定に基づいて出力
    
    Args:
        patent_results: 特許結果のリスト
    """
    config = load_config()
    
    csv_path = config['io']['out_csv']
    jsonl_path = config['io']['out_jsonl']
    
    logger.debug(f"Exporting with config: CSV={csv_path}, JSONL={jsonl_path}")
    
    batch_export(patent_results, csv_path, jsonl_path)


if __name__ == "__main__":
    # 簡単なテスト
    test_results = [
        {
            "rank": 1,
            "pub_number": "JP2025-100001A",
            "title": "液体分離設備の予測保全システム",
            "assignee": "アクアテック株式会社",
            "pub_date": "2025-03-15",
            "decision": "hit",
            "LLM_confidence": 0.95,
            "hit_reasons": [
                {"quote": "運転データから劣化予測", "source": "claim 1"},
                {"quote": "将来の性能劣化を予測", "source": "abstract"}
            ],
            "url_hint": "https://example.com/patents/JP2025-100001A",
            "reasons": [
                {"quote": "運転データから劣化予測", "source": {"field": "claim", "locator": "claim 1"}},
                {"quote": "将来の性能劣化を予測", "source": {"field": "abstract", "locator": "sent 2"}}
            ],
            "flags": {"verified": False, "used_retrieval": False, "used_topk": False}
        }
    ]
    
    # テスト出力
    export_to_csv(test_results, "test_output.csv")
    export_to_jsonl(test_results, "test_output.jsonl")
    print("Test export completed")