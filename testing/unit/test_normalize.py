"""
Tests for normalize module - JSONデータ正規化機能のテスト
"""

import pytest
import json
from typing import Dict, Any
from unittest.mock import Mock, patch

# normalize.py実装済み - 実際の関数をインポート
from src.core.normalize import normalize_patent_json, normalize_invention_idea


class TestNormalizePatentJson:
    """特許JSONデータ正規化のテストクラス"""

    def test_normalize_basic_patent_json(self):
        """基本的な特許JSONの正規化テスト"""
        # テストデータ
        input_data = {
            "publication_number": "JP2025-100001A",
            "title": "液体分離設備の予測保全システム",
            "assignee": "アクアテック株式会社",
            "pub_date": "2025-01-15",
            "claims": [
                {"no": 1, "text": "液体分離設備において...", "is_independent": True},
                {"no": 2, "text": "請求項1に記載の...", "is_independent": False}
            ],
            "abstract": "本発明は液体分離設備の運転データを分析し...",
            "ipc": ["B01D"],
            "cpc": ["B01D61/00"]
        }

        expected_keys = ["publication_number", "title", "assignee", "pub_date", "claims", "abstract"]
        
        # 実際の正規化関数をテスト
        result = normalize_patent_json(input_data)
        for key in expected_keys:
            assert key in result
        
        # 正規化された値の確認
        assert result["publication_number"] == "JP2025-100001A"
        assert "液体分離設備" in result["title"]
        assert len(result["claims"]) == 2

    def test_normalize_handles_missing_optional_fields(self):
        """任意フィールドが欠落している場合のテスト"""
        input_data = {
            "publication_number": "JP2025-100002A",
            "title": "テストタイトル",
            "assignee": "テスト会社",
            "pub_date": "2025-01-15",
            "claims": [{"no": 1, "text": "テスト請求項", "is_independent": True}],
            "abstract": "テスト抄録"
        }

        # 必須フィールドのみでも正規化できることを確認
        result = normalize_patent_json(input_data)
        required_fields = ["publication_number", "title", "assignee", "pub_date", "claims", "abstract"]
        assert all(key in result for key in required_fields)

    def test_normalize_handles_encoding_issues(self):
        """エンコーディング問題の処理テスト"""
        input_data = {
            "publication_number": "JP2025-100003A",
            "title": "液体分離設備の監視システム",  # 日本語文字
            "assignee": "株式会社テスト",
            "pub_date": "2025-01-15",
            "claims": [{"no": 1, "text": "液体分離において...", "is_independent": True}],
            "abstract": "本発明は..."
        }

        # 日本語文字が正しく処理されることを確認
        result = normalize_patent_json(input_data)
        assert isinstance(result["title"], str)
        assert "液体分離設備" in result["title"]
        assert len(result["title"]) > 0

    def test_normalize_handles_invalid_dates(self):
        """無効な日付形式の処理テスト"""
        # 実装後にエラーハンドリングをテスト
        invalid_data = {
            "publication_number": "JP2025-100004A",
            "title": "テストタイトル",
            "assignee": "テスト会社", 
            "pub_date": "invalid-date",
            "claims": [{"no": 1, "text": "テスト請求項", "is_independent": True}],
            "abstract": "テスト抄録"
        }

        # 無効な日付で例外が発生することを確認
        with pytest.raises(ValueError):
            normalize_patent_json(invalid_data)


class TestNormalizeInventionIdea:
    """発明アイデア正規化のテストクラス"""

    def test_normalize_text_invention_idea(self):
        """テキスト形式の発明アイデア正規化テスト"""
        text_input = """
        液体分離設備において、運転データを収集し、
        将来の劣化指標を予測するシステム
        """

        # テキスト形式の発明アイデア正規化をテスト
        result = normalize_invention_idea(text_input)
        assert isinstance(result, dict)
        assert "title" in result
        assert "content" in result
        assert "液体分離設備" in result["title"]

    def test_normalize_json_invention_idea(self):
        """JSON形式の発明アイデア正規化テスト"""
        json_input = {
            "title": "液体分離設備の予測保全システム",
            "problem": "既存設備の劣化予測が困難",
            "solution": "運転データから将来の劣化を予測",
            "effects": "保全コスト削減、設備稼働率向上",
            "key_elements": ["運転データ収集", "劣化予測", "アラート機能"],
            "constraints": "液体分離設備に限定"
        }

        # JSON形式の発明アイデア正規化をテスト
        result = normalize_invention_idea(json_input)
        expected_keys = ["title", "problem", "solution", "effects"]
        assert all(key in result for key in expected_keys)
        assert "key_elements" in result
        assert len(result["key_elements"]) == 3

    def test_normalize_handles_partial_json_invention(self):
        """部分的なJSON発明アイデアの処理テスト"""
        partial_input = {
            "title": "テストタイトル",
            "solution": "テストソリューション"
        }

        # 部分的なJSONでも正規化できることを確認
        result = normalize_invention_idea(partial_input)
        assert "title" in result
        assert "solution" in result
        assert isinstance(result["title"], str)


class TestNormalizationUtilities:
    """正規化ユーティリティ関数のテストクラス"""

    def test_clean_whitespace(self):
        """空白文字の正規化テスト"""
        test_cases = [
            ("  text  ", "text"),
            ("text\n\n", "text"),
            ("\t\ttext\t", "text"),
            ("  multi   word  ", "multi word")
        ]

        from src.core.normalize import clean_text
        
        for input_text, expected in test_cases:
            result = clean_text(input_text)
            # 正規化された文字列をチェック
            assert result == expected

    def test_normalize_units(self):
        """単位の正規化テスト"""
        # 実装後にテスト予定
        # 例: "100L/h" -> {"value": 100, "unit": "L/h"}
        pass

    def test_normalize_case(self):
        """大文字・小文字の正規化テスト"""
        # 実装後にテスト予定
        pass


if __name__ == "__main__":
    pytest.main([__file__])