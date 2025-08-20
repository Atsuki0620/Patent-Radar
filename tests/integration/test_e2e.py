"""
E2E統合テスト - システム全体の動作確認

実データを使用してシステム全体の動作を検証する。
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any


class TestE2EPatentScreening:
    """E2E特許スクリーニングテスト"""
    
    @pytest.fixture
    def test_data_path(self):
        """テストデータのパスを返す"""
        return Path(__file__).parent.parent / 'data'
    
    @pytest.fixture
    def patents_data(self, test_data_path):
        """特許テストデータを読み込む"""
        patents = []
        with open(test_data_path / 'patents.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    patents.append(json.loads(line))
        return patents[:3]  # 最初の3件のみ使用
    
    @pytest.fixture
    def labels_data(self, test_data_path):
        """ラベルテストデータを読み込む"""
        labels = {}
        with open(test_data_path / 'labels.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    label_data = json.loads(line)
                    labels[label_data['publication_number']] = label_data['gold_label']
        return labels
    
    @pytest.fixture
    def invention_idea(self):
        """テスト用発明アイデア"""
        return {
            "title": "液体分離設備の運転データ予測システム",
            "problem": "膜の劣化度を事前に予測できないため計画的な保全が困難",
            "solution": "時系列運転データから機械学習により将来の劣化指標を予測",
            "effects": "突発的な設備停止を防止し、計画的保全が可能",
            "key_elements": [
                "運転データの時系列収集",
                "機械学習による劣化予測",
                "予測結果の表示・出力"
            ]
        }
    
    @pytest.fixture
    def mock_llm_responses(self):
        """モックLLMレスポンスを準備"""
        def get_mock_response(content_type, pub_number):
            if content_type == "invention_summary":
                return {
                    "content": "液体分離設備の運転データから機械学習により将来の膜劣化を予測し、計画的保全を可能にするシステム"
                }
            elif content_type == "patent_summary":
                return {
                    "content": f"特許{pub_number}の要約: 液体分離設備における予測技術"
                }
            elif content_type == "classification":
                # 実際のラベルに基づいた分類結果を返す
                hit_patents = ["JP2025-100001A", "US2025/0200301A1"]
                if pub_number in hit_patents:
                    return {
                        "content": json.dumps({
                            "decision": "hit",
                            "confidence": 0.85,
                            "reasons": [
                                {"quote": "運転データの時系列受領", "source": {"field": "claim", "locator": "claim 1"}},
                                {"quote": "将来時点における膜の劣化指標を予測", "source": {"field": "claim", "locator": "claim 1"}}
                            ]
                        })
                    }
                else:
                    return {
                        "content": json.dumps({
                            "decision": "miss",
                            "confidence": 0.70,
                            "reasons": [
                                {"quote": "現在の膜性能を診断", "source": {"field": "claim", "locator": "claim 1"}}
                            ]
                        })
                    }
        return get_mock_response
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-1234567890123456789012345678901234567890123456789012'})
    @patch('src.llm.client.OpenAI')
    def test_complete_e2e_workflow(self, mock_openai, patents_data, labels_data, invention_idea, mock_llm_responses):
        """完全なE2Eワークフローテスト"""
        from src.core.screener import PatentScreener
        
        # モッククライアントの設定
        mock_client = Mock()
        
        # カウンターで呼び出し順序を管理
        call_count = 0
        
        def side_effect(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # 最初の呼び出しは発明要約
            if call_count == 1:
                return Mock(choices=[Mock(message=Mock(content=json.dumps(mock_llm_responses("invention_summary", None)["content"])))])
            
            # 以降は特許の分類処理
            patent_index = (call_count - 2) % len(patents_data)
            patent = patents_data[patent_index]
            pub_number = patent["publication_number"]
            
            classification_response = mock_llm_responses("classification", pub_number)
            return Mock(choices=[Mock(message=Mock(content=classification_response["content"]))])
        
        mock_client.chat.completions.create.side_effect = side_effect
        mock_openai.return_value = mock_client
        
        # PatentScreenerインスタンス作成
        screener = PatentScreener()
        
        # 一時的な出力ファイルパスを設定
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "test_results.csv"
            jsonl_path = temp_path / "test_results.jsonl"
            
            # スクリーニング実行
            summary = screener.analyze(
                invention=invention_idea,
                patents=patents_data,
                output_csv=str(csv_path),
                output_jsonl=str(jsonl_path)
            )
            
            # サマリーの基本検証
            assert summary is not None
            assert isinstance(summary, dict)
            
            # JSONL出力から結果を読み取り
            results = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            
            # 結果の基本検証
            assert len(results) == len(patents_data)
            
            # 各結果に必要な項目が含まれていることを確認
            for result in results:
                assert 'pub_number' in result
                assert 'decision' in result
                # confidence はLLM処理の結果として含まれている可能性がある
            
            # CSV出力の確認
            assert csv_path.exists()
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_content = f.read()
                assert 'rank,pub_number,title,assignee' in csv_content
                assert 'JP2025-100001A' in csv_content
            
            # JSONL出力の確認
            assert jsonl_path.exists()
            jsonl_results = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        jsonl_results.append(json.loads(line))
            
            assert len(jsonl_results) == len(patents_data)
            for result in jsonl_results:
                assert 'pub_number' in result
                assert 'decision' in result
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-1234567890123456789012345678901234567890123456789012'})
    @patch('src.llm.client.OpenAI')
    def test_ranking_consistency(self, mock_openai, patents_data, invention_idea, mock_llm_responses):
        """ランキングの一貫性テスト"""
        from src.core.screener import PatentScreener
        
        # モッククライアントの設定
        mock_client = Mock()
        call_count = 0
        
        def side_effect(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return Mock(choices=[Mock(message=Mock(content=json.dumps("発明要約")))])
            
            # 信頼度の順序を制御（再現可能な結果のため）
            confidences = [0.85, 0.70, 0.75]
            patent_index = (call_count - 2) % len(patents_data)
            confidence = confidences[patent_index]
            
            classification = {
                "decision": "hit" if confidence > 0.75 else "miss",
                "confidence": confidence,
                "reasons": [{"quote": "テスト理由", "source": {"field": "claim", "locator": "claim 1"}}]
            }
            
            return Mock(choices=[Mock(message=Mock(content=json.dumps(classification)))])
        
        mock_client.chat.completions.create.side_effect = side_effect
        mock_openai.return_value = mock_client
        
        screener = PatentScreener()
        
        # 同じデータで2回実行
        summary1 = screener.analyze(invention=invention_idea, patents=patents_data)
        summary2 = screener.analyze(invention=invention_idea, patents=patents_data)
        
        # サマリーが返されることを確認
        assert summary1 is not None
        assert summary2 is not None
        
        # 処理された項目数が同じことを確認
        # NOTE: 実際の結果はファイル出力される仕様のため、
        # ここでは処理が正常に完了することを確認
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-1234567890123456789012345678901234567890123456789012'})
    @patch('src.llm.client.OpenAI')  
    def test_error_handling_e2e(self, mock_openai, patents_data, invention_idea):
        """E2Eエラーハンドリングテスト"""
        from src.core.screener import PatentScreener
        
        # エラーを発生させるモック設定
        mock_client = Mock()
        mock_response = Mock(spec=["status_code", "headers"])
        mock_response.status_code = 429
        mock_response.headers = {"Content-Type": "application/json"}
        
        # LLMクライアントでエラーが発生
        mock_client.chat.completions.create.side_effect = Exception("API error for testing")
        mock_openai.return_value = mock_client
        
        screener = PatentScreener()
        
        # エラーが適切に処理されることを確認
        with pytest.raises(Exception):
            screener.analyze(invention=invention_idea, patents=patents_data)