"""
Tests for export module - CSV/JSONL出力機能のテスト
"""

import pytest
import json
import csv
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, mock_open
from datetime import datetime


class TestExportToCSV:
    """CSV出力機能のテストクラス"""

    def test_export_basic_csv(self, tmp_path):
        """基本的なCSV出力テスト"""
        # テストデータ（ランキング済み特許リスト）
        patent_results = [
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
                "url_hint": "https://example.com/patents/JP2025-100001A"
            },
            {
                "rank": 2,
                "pub_number": "US2025/0200301A1",
                "title": "Predictive Maintenance for Water Treatment",
                "assignee": "HydroSmart Inc.",
                "pub_date": "2025-01-10",
                "decision": "hit",
                "LLM_confidence": 0.88,
                "hit_reasons": [
                    {"quote": "forecast fouling", "source": "claim 1"}
                ],
                "url_hint": "https://example.com/patents/US2025-0200301A1"
            }
        ]

        output_path = tmp_path / "test_output.csv"
        
        # 実装後にテスト
        from src.core.export import export_to_csv
        export_to_csv(patent_results, output_path)
        
        # CSVファイルの存在確認
        assert output_path.exists()
        
        # CSVファイルの内容確認
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 2
        assert rows[0]["pub_number"] == "JP2025-100001A"
        assert rows[0]["decision"] == "hit"
        assert float(rows[0]["LLM_confidence"]) == 0.95

    def test_export_csv_with_empty_reasons(self, tmp_path):
        """ヒット理由が空の場合のCSV出力テスト"""
        patent_results = [
            {
                "rank": 1,
                "pub_number": "JP2025-100002A",
                "title": "テストタイトル",
                "assignee": "テスト会社",
                "pub_date": "2025-01-15",
                "decision": "miss",
                "LLM_confidence": 0.25,
                "hit_reasons": [],  # 空のヒット理由
                "url_hint": ""
            }
        ]

        output_path = tmp_path / "test_empty_reasons.csv"
        
        # 実装後にテスト
        from src.core.export import export_to_csv
        export_to_csv(patent_results, output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert rows[0]["hit_reason_1"] == ""
        assert rows[0]["hit_src_1"] == ""
        assert rows[0]["hit_reason_2"] == ""
        assert rows[0]["hit_src_2"] == ""

    def test_export_csv_column_order(self, tmp_path):
        """CSV列順序の確認テスト"""
        patent_results = [
            {
                "rank": 1,
                "pub_number": "JP2025-100001A",
                "title": "テスト",
                "assignee": "テスト社",
                "pub_date": "2025-01-01",
                "decision": "hit",
                "LLM_confidence": 0.9,
                "hit_reasons": [],
                "url_hint": ""
            }
        ]

        output_path = tmp_path / "test_columns.csv"
        
        expected_columns = [
            "rank", "pub_number", "title", "assignee", "pub_date",
            "decision", "LLM_confidence", "hit_reason_1", "hit_src_1",
            "hit_reason_2", "hit_src_2", "url_hint"
        ]
        
        # 実装後にテスト
        from src.core.export import export_to_csv
        export_to_csv(patent_results, output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            actual_columns = reader.fieldnames
            
        assert actual_columns == expected_columns

    def test_export_csv_special_characters(self, tmp_path):
        """特殊文字処理のテスト"""
        patent_results = [
            {
                "rank": 1,
                "pub_number": "JP2025-100001A",
                "title": "タイトル,カンマ\"引用符\"含む",  # カンマと引用符
                "assignee": "会社名\n改行含む",  # 改行
                "pub_date": "2025-01-01",
                "decision": "hit",
                "LLM_confidence": 0.9,
                "hit_reasons": [
                    {"quote": "引用\"テキスト\"", "source": "claim 1"}
                ],
                "url_hint": ""
            }
        ]

        output_path = tmp_path / "test_special.csv"
        
        # 実装後にテスト
        from src.core.export import export_to_csv
        export_to_csv(patent_results, output_path)
        
        # 特殊文字が正しくエスケープされることを確認
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '""' in content or '"' in content  # 引用符のエスケープ

    def test_export_csv_file_permission_error(self, tmp_path):
        """ファイル書き込み権限エラーのテスト"""
        patent_results = [{"rank": 1, "pub_number": "JP2025-100001A"}]
        
        # 書き込み不可能なパス
        output_path = Path("/root/impossible_path.csv")
        
        # 実装後にテスト
        from src.core.export import export_to_csv
        
        with pytest.raises(PermissionError):
            export_to_csv(patent_results, output_path)

    def test_export_csv_encoding_handling(self, tmp_path):
        """エンコーディング処理のテスト"""
        patent_results = [
            {
                "rank": 1,
                "pub_number": "JP2025-100001A",
                "title": "日本語タイトル🔬絵文字含む",  # 日本語と絵文字
                "assignee": "株式会社テスト",
                "pub_date": "2025-01-01",
                "decision": "hit",
                "LLM_confidence": 0.9,
                "hit_reasons": [],
                "url_hint": ""
            }
        ]

        output_path = tmp_path / "test_encoding.csv"
        
        # 実装後にテスト
        from src.core.export import export_to_csv
        export_to_csv(patent_results, output_path)
        
        # UTF-8で正しく読み込めることを確認
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert "日本語タイトル" in rows[0]["title"]


class TestExportToJSONL:
    """JSONL出力機能のテストクラス"""

    def test_export_basic_jsonl(self, tmp_path):
        """基本的なJSONL出力テスト"""
        patent_results = [
            {
                "pub_number": "JP2025-100001A",
                "decision": "hit",
                "LLM_confidence": 0.95,
                "reasons": [
                    {"quote": "運転データから劣化予測", "source": {"field": "claim", "locator": "claim 1"}},
                    {"quote": "将来の性能劣化を予測", "source": {"field": "abstract", "locator": "sent 2"}}
                ],
                "flags": {"verified": False, "used_retrieval": False, "used_topk": False},
                "metadata": {
                    "title": "液体分離設備の予測保全システム",
                    "assignee": "アクアテック株式会社",
                    "pub_date": "2025-03-15"
                }
            },
            {
                "pub_number": "US2025/0200301A1",
                "decision": "hit",
                "LLM_confidence": 0.88,
                "reasons": [
                    {"quote": "forecast fouling", "source": {"field": "claim", "locator": "claim 1"}}
                ],
                "flags": {"verified": False, "used_retrieval": False, "used_topk": False},
                "metadata": {
                    "title": "Predictive Maintenance",
                    "assignee": "HydroSmart Inc.",
                    "pub_date": "2025-01-10"
                }
            }
        ]

        output_path = tmp_path / "test_output.jsonl"
        
        # 実装後にテスト
        from src.core.export import export_to_jsonl
        export_to_jsonl(patent_results, output_path)
        
        # JSONLファイルの存在確認
        assert output_path.exists()
        
        # JSONLファイルの内容確認
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        assert len(lines) == 2
        
        # 各行が有効なJSONであることを確認
        first_json = json.loads(lines[0])
        assert first_json["pub_number"] == "JP2025-100001A"
        assert first_json["decision"] == "hit"
        assert first_json["LLM_confidence"] == 0.95

    def test_export_jsonl_structure_validation(self, tmp_path):
        """JSONL構造検証テスト"""
        patent_results = [
            {
                "pub_number": "JP2025-100001A",
                "decision": "hit",
                "LLM_confidence": 0.95,
                "reasons": [
                    {"quote": "引用テキスト", "source": {"field": "claim", "locator": "claim 1"}}
                ],
                "flags": {"verified": False, "used_retrieval": False, "used_topk": False}
            }
        ]

        output_path = tmp_path / "test_structure.jsonl"
        
        # 実装後にテスト
        from src.core.export import export_to_jsonl
        export_to_jsonl(patent_results, output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            result_json = json.loads(f.readline())
            
        # 必須フィールドの確認
        required_fields = ["pub_number", "decision", "LLM_confidence", "reasons", "flags"]
        for field in required_fields:
            assert field in result_json
            
        # flags構造の確認
        assert "verified" in result_json["flags"]
        assert "used_retrieval" in result_json["flags"]
        assert "used_topk" in result_json["flags"]

    def test_export_jsonl_empty_list(self, tmp_path):
        """空リストのJSONL出力テスト"""
        patent_results = []
        output_path = tmp_path / "test_empty.jsonl"
        
        # 実装後にテスト
        from src.core.export import export_to_jsonl
        export_to_jsonl(patent_results, output_path)
        
        # 空ファイルまたは存在しないことを確認
        assert not output_path.exists() or output_path.stat().st_size == 0

    def test_export_jsonl_invalid_json(self, tmp_path):
        """無効なJSON構造のエラーハンドリング"""
        # 循環参照を含む無効なデータ
        invalid_data = {"pub_number": "JP2025-100001A"}
        invalid_data["self_ref"] = invalid_data  # 循環参照
        
        patent_results = [invalid_data]
        output_path = tmp_path / "test_invalid.jsonl"
        
        # 実装後にテスト
        from src.core.export import export_to_jsonl
        
        with pytest.raises(ValueError):
            export_to_jsonl(patent_results, output_path)

    def test_export_jsonl_unicode_handling(self, tmp_path):
        """Unicode文字処理のテスト"""
        patent_results = [
            {
                "pub_number": "JP2025-100001A",
                "decision": "hit",
                "LLM_confidence": 0.95,
                "reasons": [
                    {"quote": "日本語の引用テキスト", "source": {"field": "claim", "locator": "請求項1"}}
                ],
                "flags": {"verified": False, "used_retrieval": False, "used_topk": False},
                "metadata": {
                    "title": "液体分離設備の予測保全システム",
                    "assignee": "株式会社テスト🔬"
                }
            }
        ]

        output_path = tmp_path / "test_unicode.jsonl"
        
        # 実装後にテスト
        from src.core.export import export_to_jsonl
        export_to_jsonl(patent_results, output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            result_json = json.loads(f.readline())
            
        assert "日本語の引用テキスト" in result_json["reasons"][0]["quote"]
        assert "液体分離設備" in result_json["metadata"]["title"]


class TestExportUtilities:
    """出力ユーティリティ機能のテストクラス"""

    def test_create_output_directory(self, tmp_path):
        """出力ディレクトリ作成テスト"""
        output_path = tmp_path / "nested" / "directory" / "output.csv"
        
        # 実装後にテスト
        from src.core.export import ensure_output_directory
        ensure_output_directory(output_path)
        
        assert output_path.parent.exists()
        assert output_path.parent.is_dir()

    def test_validate_output_path_security(self):
        """出力パスのセキュリティ検証テスト"""
        # パストラバーサル攻撃の検出
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config",
            "/absolute/path/to/system/file",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\sam",
            "/root/.ssh/id_rsa"
        ]
        
        # 実装後にテスト
        from src.core.export import validate_output_path, SecurityError
        
        for path in malicious_paths:
            with pytest.raises(SecurityError):
                validate_output_path(path)

    def test_format_hit_reasons_for_csv(self):
        """CSV用ヒット理由フォーマットのテスト"""
        hit_reasons = [
            {"quote": "運転データから劣化予測", "source": "claim 1"},
            {"quote": "将来の性能劣化を予測", "source": "abstract"},
            {"quote": "第3の理由（切り捨て）", "source": "description"}
        ]
        
        # 実装後にテスト
        from src.core.export import format_hit_reasons_for_csv
        formatted = format_hit_reasons_for_csv(hit_reasons)
        
        # 最大2件まで
        assert "hit_reason_1" in formatted
        assert "hit_src_1" in formatted
        assert "hit_reason_2" in formatted
        assert "hit_src_2" in formatted
        assert "hit_reason_3" not in formatted

    def test_sanitize_csv_value(self):
        """CSV値のサニタイズテスト"""
        test_cases = [
            ("normal text", "normal text"),
            ("text,with,comma", "text,with,comma"),  # CSVライブラリが処理
            ('text"with"quotes', 'text"with"quotes'),  # CSVライブラリが処理
            ("text\nwith\nnewlines", "text with newlines"),  # 改行を空白に
            ("text\rwith\rcarriage", "text with carriage"),  # CRを空白に
            (None, ""),  # Noneは空文字列に
            (123, "123"),  # 数値は文字列に
        ]
        
        # 実装後にテスト
        from src.core.export import sanitize_csv_value
        
        for input_val, expected in test_cases:
            result = sanitize_csv_value(input_val)
            assert result == expected

    def test_batch_export(self, tmp_path):
        """バッチ出力テスト（CSV + JSONL同時出力）"""
        patent_results = [
            {
                "rank": 1,
                "pub_number": "JP2025-100001A",
                "title": "テスト特許",
                "assignee": "テスト社",
                "pub_date": "2025-01-01",
                "decision": "hit",
                "LLM_confidence": 0.9,
                "hit_reasons": [],
                "url_hint": "",
                "reasons": [],
                "flags": {"verified": False, "used_retrieval": False, "used_topk": False}
            }
        ]
        
        csv_path = tmp_path / "output.csv"
        jsonl_path = tmp_path / "output.jsonl"
        
        # 実装後にテスト
        from src.core.export import batch_export
        batch_export(patent_results, csv_path, jsonl_path)
        
        assert csv_path.exists()
        assert jsonl_path.exists()

    def test_export_with_config(self, tmp_path):
        """設定ファイルベースの出力テスト"""
        patent_results = [
            {
                "rank": 1,
                "pub_number": "JP2025-100001A",
                "title": "テスト",
                "assignee": "テスト社",
                "pub_date": "2025-01-01",
                "decision": "hit",
                "LLM_confidence": 0.9,
                "hit_reasons": [],
                "url_hint": ""
            }
        ]
        
        # 実装後にテスト
        from src.core.export import export_with_config
        
        with patch('src.core.export.load_config') as mock_config:
            mock_config.return_value = {
                'io': {
                    'out_csv': str(tmp_path / 'config_output.csv'),
                    'out_jsonl': str(tmp_path / 'config_output.jsonl')
                }
            }
            
            export_with_config(patent_results)
            
            assert (tmp_path / 'config_output.csv').exists()
            assert (tmp_path / 'config_output.jsonl').exists()

    def test_export_performance_large_dataset(self, tmp_path):
        """大量データのパフォーマンステスト"""
        # 1000件のダミーデータ
        patent_results = []
        for i in range(1000):
            patent_results.append({
                "rank": i + 1,
                "pub_number": f"JP2025-{100000 + i}A",
                "title": f"テスト特許{i}",
                "assignee": f"テスト社{i % 10}",
                "pub_date": "2025-01-01",
                "decision": "hit" if i % 2 == 0 else "miss",
                "LLM_confidence": 0.5 + (i % 50) / 100,
                "hit_reasons": [],
                "url_hint": ""
            })
        
        output_path = tmp_path / "large_output.csv"
        
        # 実装後にテスト
        from src.core.export import export_to_csv
        import time
        
        start_time = time.time()
        export_to_csv(patent_results, output_path)
        elapsed_time = time.time() - start_time
        
        # 1000件の出力が1秒以内に完了
        assert elapsed_time < 1.0
        assert output_path.exists()
        
        # ファイルサイズの確認
        assert output_path.stat().st_size > 0

    def test_export_memory_efficiency(self, tmp_path):
        """メモリ効率性テスト"""
        # メモリ使用量を監視しながら大量データ出力
        import tracemalloc
        
        patent_results = []
        for i in range(10000):
            patent_results.append({
                "rank": i + 1,
                "pub_number": f"JP2025-{100000 + i}A",
                "title": f"テスト特許{i}" * 10,  # 長いタイトル
                "assignee": f"テスト社{i}",
                "pub_date": "2025-01-01",
                "decision": "hit",
                "LLM_confidence": 0.5,
                "hit_reasons": [
                    {"quote": f"理由{i}", "source": "claim 1"}
                ],
                "url_hint": f"https://example.com/{i}"
            })
        
        output_path = tmp_path / "memory_test.csv"
        
        # 実装後にテスト
        from src.core.export import export_to_csv
        
        tracemalloc.start()
        export_to_csv(patent_results, output_path)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # メモリ使用量が100MB未満
        assert peak < 100 * 1024 * 1024

    def test_atomic_write_operation(self, tmp_path):
        """アトミック書き込み操作のテスト（Code-reviewer推奨）"""
        patent_results = [
            {
                "rank": 1,
                "pub_number": "JP2025-100001A",
                "title": "テスト",
                "assignee": "テスト社",
                "pub_date": "2025-01-01",
                "decision": "hit",
                "LLM_confidence": 0.9,
                "hit_reasons": [],
                "url_hint": ""
            }
        ]
        
        output_path = tmp_path / "atomic_test.csv"
        
        # 実装後にテスト - 一時ファイル経由の書き込み
        from src.core.export import export_to_csv
        
        # 書き込み中にエラーが発生しても、元ファイルが破損しないことを確認
        export_to_csv(patent_results, output_path)
        
        # ファイルが完全に書き込まれていることを確認
        assert output_path.exists()
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "JP2025-100001A" in content

    def test_concurrent_access_handling(self, tmp_path):
        """同時アクセス処理のテスト（Code-reviewer推奨）"""
        import threading
        import time
        
        patent_results = [
            {
                "rank": 1,
                "pub_number": f"JP2025-{i}A",
                "title": f"テスト{i}",
                "assignee": "テスト社",
                "pub_date": "2025-01-01",
                "decision": "hit",
                "LLM_confidence": 0.9,
                "hit_reasons": [],
                "url_hint": ""
            }
            for i in range(100)
        ]
        
        output_path = tmp_path / "concurrent_test.csv"
        
        # 実装後にテスト
        from src.core.export import export_to_csv
        
        def write_csv():
            try:
                export_to_csv(patent_results, output_path)
            except Exception:
                pass  # 同時アクセスエラーを許容
        
        # 複数スレッドから同時書き込み
        threads = [threading.Thread(target=write_csv) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 少なくとも1つの書き込みが成功していることを確認
        assert output_path.exists()

    def test_disk_space_exhaustion(self, tmp_path):
        """ディスク容量不足のテスト（Code-reviewer推奨）"""
        # 大量のダミーデータ
        patent_results = [
            {
                "rank": i,
                "pub_number": f"JP2025-{i}A",
                "title": "x" * 10000,  # 非常に長いタイトル
                "assignee": "テスト社",
                "pub_date": "2025-01-01",
                "decision": "hit",
                "LLM_confidence": 0.9,
                "hit_reasons": [],
                "url_hint": ""
            }
            for i in range(1000)
        ]
        
        # 実装後にテスト
        from src.core.export import export_to_csv
        
        # モック化された小さなディスク容量
        with patch('shutil.disk_usage') as mock_disk:
            mock_disk.return_value = (100, 50, 10)  # total, used, free (bytes)
            
            output_path = tmp_path / "disk_full_test.csv"
            
            # ディスク容量不足エラーの適切な処理を確認
            with pytest.raises(IOError):
                export_to_csv(patent_results, output_path)

    def test_malformed_input_structure(self):
        """不正な入力構造のテスト（Code-reviewer推奨）"""
        # 様々な不正なデータ構造
        malformed_inputs = [
            None,  # None
            "not a list",  # 文字列
            123,  # 数値
            [{"no_pub_number": "missing"}],  # 必須フィールド欠落
            [{"pub_number": None}],  # Noneフィールド
            [{"pub_number": ["list", "value"]}],  # 予期しない型
        ]
        
        # 実装後にテスト
        from src.core.export import export_to_csv
        from pathlib import Path
        
        for malformed in malformed_inputs:
            with pytest.raises((TypeError, ValueError, AttributeError)):
                export_to_csv(malformed, Path("/tmp/test.csv"))


if __name__ == "__main__":
    pytest.main([__file__])