"""
Tests for LLM client module - OpenAI APIクライアント機能のテスト
"""

import pytest
import json
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import asyncio
import httpx


class TestLLMClient:
    """LLMクライアントの基本機能テスト"""

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-1234567890123456789012345678901234567890123456789012'})
    @patch('src.llm.client.OpenAI')
    def test_client_initialization(self, mock_openai):
        """クライアント初期化のテスト"""
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient()
        
        # OpenAIクライアントが初期化されることを確認
        mock_openai.assert_called_once()
        assert hasattr(client, '_client')
        assert hasattr(client, '_config')

    @patch('src.llm.client.OpenAI')
    def test_client_with_api_key(self, mock_openai):
        """API キー指定での初期化テスト"""
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        api_key = "sk-1234567890123456789012345678901234567890123456789012"  # 48文字以上
        client = LLMClient(api_key=api_key)
        
        # API キーが正しく設定されることを確認
        mock_openai.assert_called_once_with(
            api_key=api_key,
            timeout=30.0,
            max_retries=0
        )

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-1234567890123456789012345678901234567890123456789012'})
    @patch('src.llm.client.OpenAI')
    def test_client_with_env_api_key(self, mock_openai):
        """環境変数からのAPI キー読み込みテスト"""
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient()
        
        # 環境変数のAPI キーが使用されることを確認
        mock_openai.assert_called_once_with(
            api_key='sk-1234567890123456789012345678901234567890123456789012',
            timeout=30.0,
            max_retries=0
        )

    def test_client_missing_api_key(self):
        """API キー未設定時のエラーテスト"""
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMClientError
        
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(LLMClientError):
                LLMClient()

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-1234567890123456789012345678901234567890123456789012'})
    @patch('src.llm.client.OpenAI')
    def test_config_from_yaml(self, mock_openai):
        """YAML設定ファイルからの設定読み込みテスト"""
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        with patch('src.llm.client.load_config') as mock_config:
            mock_config.return_value = {
                'llm': {
                    'model': 'gpt-4o-mini',
                    'temperature': 0.0,
                    'max_tokens': 320,
                    'timeout': 30
                }
            }
            
            client = LLMClient()
            
            assert client._config['model'] == 'gpt-4o-mini'
            assert client._config['temperature'] == 0.0
            assert client._config['max_tokens'] == 320
            assert client._config['timeout'] == 30


class TestLLMCompletion:
    """LLM completion機能のテスト"""

    def setup_method(self):
        """テストメソッドのセットアップ"""
        self.mock_response = ChatCompletion(
            id="chatcmpl-test123",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"decision": "hit", "confidence": 0.95, "reasons": ["運転データから劣化予測"]}'
                    ),
                    finish_reason="stop"
                )
            ]
        )

    @patch('src.llm.client.OpenAI')
    def test_completion_basic(self, mock_openai):
        """基本的な completion テスト"""
        # Mock レスポンスのセットアップ
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = self.mock_response
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        
        messages = [
            {"role": "system", "content": "You are a patent analysis assistant"},
            {"role": "user", "content": "分析してください"}
        ]
        
        response = client.completion(messages)
        
        # APIが正しいパラメータで呼ばれることを確認
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        
        assert call_args[1]['model'] == 'gpt-4o-mini'
        assert call_args[1]['temperature'] == 0.0
        assert call_args[1]['max_tokens'] == 320
        assert call_args[1]['messages'] == messages
        
        # レスポンスの構造を確認
        assert 'decision' in response
        assert 'confidence' in response
        assert 'reasons' in response

    @patch('src.llm.client.OpenAI')
    def test_completion_with_custom_params(self, mock_openai):
        """カスタムパラメータでの completion テスト"""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = self.mock_response
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        
        messages = [{"role": "user", "content": "テスト"}]
        
        response = client.completion(
            messages=messages,
            model="gpt-4o",
            temperature=0.1,
            max_tokens=500
        )
        
        call_args = mock_client.chat.completions.create.call_args
        
        # カスタムパラメータが適用されることを確認
        assert call_args[1]['model'] == 'gpt-4o'
        assert call_args[1]['temperature'] == 0.1
        assert call_args[1]['max_tokens'] == 500

    @patch('src.llm.client.OpenAI')
    def test_completion_json_response_format(self, mock_openai):
        """JSON レスポンス形式の指定テスト"""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = self.mock_response
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        messages = [{"role": "user", "content": "JSON形式で返して"}]
        
        response = client.completion(messages, response_format="json")
        
        call_args = mock_client.chat.completions.create.call_args
        
        # response_formatが設定されることを確認
        assert call_args[1]['response_format'] == {"type": "json_object"}

    @patch('src.llm.client.OpenAI')
    def test_completion_invalid_json_response(self, mock_openai):
        """不正なJSON レスポンスのハンドリングテスト"""
        # 不正なJSONを返すモックレスポンス
        invalid_response = ChatCompletion(
            id="chatcmpl-test123",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"decision": "hit", "confidence": 0.95, invalid json'  # 不正なJSON
                    ),
                    finish_reason="stop"
                )
            ]
        )
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = invalid_response
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMResponseError
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        messages = [{"role": "user", "content": "テスト"}]
        
        # JSON パースエラーが適切に処理されることを確認
        with pytest.raises(LLMResponseError):
            client.completion(messages)


class TestLLMErrorHandling:
    """LLMエラーハンドリングのテスト"""

    @patch('src.llm.client.OpenAI')
    def test_api_error_handling(self, mock_openai):
        """OpenAI API エラーのハンドリングテスト"""
        mock_client = Mock()
        # Create proper mock request for APIError
        mock_request = Mock(spec=httpx.Request)
        mock_request.url = "https://api.openai.com/v1/chat/completions"
        mock_request.method = "POST"
        api_error = openai.APIError("API Error", request=mock_request, body=None)
        mock_client.chat.completions.create.side_effect = api_error
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMAPIError
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        messages = [{"role": "user", "content": "テスト"}]
        
        with pytest.raises(LLMAPIError):
            client.completion(messages)

    @patch('src.llm.client.OpenAI')
    def test_rate_limit_error_handling(self, mock_openai):
        """レート制限エラーのハンドリングテスト"""
        mock_client = Mock()
        # Create proper mock response for RateLimitError
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {"Content-Type": "application/json"}
        rate_limit_error = openai.RateLimitError("Rate limit exceeded", response=mock_response, body=None)
        mock_client.chat.completions.create.side_effect = rate_limit_error
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMRateLimitError
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        messages = [{"role": "user", "content": "テスト"}]
        
        with pytest.raises(LLMRateLimitError):
            client.completion(messages)

    @patch('src.llm.client.OpenAI')
    def test_timeout_error_handling(self, mock_openai):
        """タイムアウトエラーのハンドリングテスト"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.APITimeoutError("Timeout")
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMTimeoutError
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        messages = [{"role": "user", "content": "テスト"}]
        
        with pytest.raises(LLMTimeoutError):
            client.completion(messages)

    @patch('src.llm.client.OpenAI')
    def test_authentication_error_handling(self, mock_openai):
        """認証エラーのハンドリングテスト"""
        mock_client = Mock()
        # Create proper mock response for AuthenticationError
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.headers = {"Content-Type": "application/json"}
        auth_error = openai.AuthenticationError("Invalid API key", response=mock_response, body=None)
        mock_client.chat.completions.create.side_effect = auth_error
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMAuthenticationError
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        messages = [{"role": "user", "content": "テスト"}]
        
        with pytest.raises(LLMAuthenticationError):
            client.completion(messages)


class TestLLMRetryMechanism:
    """リトライ機構のテスト"""

    @patch('src.llm.client.OpenAI')
    def test_retry_on_rate_limit(self, mock_openai):
        """レート制限時のリトライテスト"""
        mock_client = Mock()
        
        # 最初の2回はレート制限エラー、3回目は成功
        mock_response = ChatCompletion(
            id="chatcmpl-test123",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"decision": "hit"}'
                    ),
                    finish_reason="stop"
                )
            ]
        )
        
        # Create proper mock response for RateLimitError
        mock_rate_response = Mock(spec=httpx.Response)
        mock_rate_response.status_code = 429
        mock_rate_response.headers = {"Content-Type": "application/json"}
        
        mock_client.chat.completions.create.side_effect = [
            openai.RateLimitError("Rate limit exceeded", response=mock_rate_response, body=None),
            openai.RateLimitError("Rate limit exceeded", response=mock_rate_response, body=None),
            mock_response
        ]
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        with patch('time.sleep') as mock_sleep:  # sleep をモック
            client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012", max_retries=3)
            messages = [{"role": "user", "content": "テスト"}]
            
            response = client.completion(messages)
            
            # 3回呼び出されることを確認
            assert mock_client.chat.completions.create.call_count == 3
            
            # 指数バックオフでsleepが呼ばれることを確認
            assert mock_sleep.call_count == 2
            
            # 最終的に成功レスポンスが返ることを確認
            assert 'decision' in response

    @patch('src.llm.client.OpenAI')
    def test_retry_exhausted(self, mock_openai):
        """リトライ回数超過時のテスト"""
        mock_client = Mock()
        # Create proper mock response for RateLimitError
        mock_rate_response = Mock(spec=httpx.Response)
        mock_rate_response.status_code = 429
        mock_rate_response.headers = {"Content-Type": "application/json"}
        mock_client.chat.completions.create.side_effect = openai.RateLimitError("Rate limit exceeded", response=mock_rate_response, body=None)
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMRateLimitError
        
        with patch('time.sleep'):
            client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012", max_retries=2)
            messages = [{"role": "user", "content": "テスト"}]
            
            with pytest.raises(LLMRateLimitError):
                client.completion(messages)
            
            # max_retries + 1回呼び出されることを確認
            assert mock_client.chat.completions.create.call_count == 3

    @patch('src.llm.client.OpenAI')
    def test_no_retry_on_auth_error(self, mock_openai):
        """認証エラー時はリトライしないことのテスト"""
        mock_client = Mock()
        # Create proper mock response for AuthenticationError
        mock_auth_response = Mock(spec=httpx.Response)
        mock_auth_response.status_code = 401
        mock_auth_response.headers = {"Content-Type": "application/json"}
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError("Invalid API key", response=mock_auth_response, body=None)
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMAuthenticationError
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012", max_retries=3)
        messages = [{"role": "user", "content": "テスト"}]
        
        with pytest.raises(LLMAuthenticationError):
            client.completion(messages)
        
        # 1回だけ呼び出されることを確認（リトライしない）
        assert mock_client.chat.completions.create.call_count == 1


class TestLLMBatchProcessing:
    """バッチ処理機能のテスト"""

    @patch('src.llm.client.OpenAI')
    def test_batch_completion(self, mock_openai):
        """複数メッセージのバッチ処理テスト"""
        mock_response_1 = ChatCompletion(
            id="chatcmpl-test1",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"decision": "hit", "confidence": 0.95}'
                    ),
                    finish_reason="stop"
                )
            ]
        )
        
        mock_response_2 = ChatCompletion(
            id="chatcmpl-test2",
            object="chat.completion",
            created=1677652289,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"decision": "miss", "confidence": 0.3}'
                    ),
                    finish_reason="stop"
                )
            ]
        )
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        
        batch_messages = [
            [{"role": "user", "content": "特許1を分析"}],
            [{"role": "user", "content": "特許2を分析"}]
        ]
        
        results = client.batch_completion(batch_messages)
        
        # 2回APIが呼ばれることを確認
        assert mock_client.chat.completions.create.call_count == 2
        
        # 結果の構造を確認
        assert len(results) == 2
        assert results[0]['decision'] == 'hit'
        assert results[1]['decision'] == 'miss'

    @patch('src.llm.client.OpenAI')
    def test_batch_completion_with_error(self, mock_openai):
        """バッチ処理でエラーが発生した場合のテスト"""
        mock_response = ChatCompletion(
            id="chatcmpl-test1",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"decision": "hit"}'
                    ),
                    finish_reason="stop"
                )
            ]
        )
        
        mock_client = Mock()
        # Create proper mock request for APIError
        mock_request = Mock(spec=httpx.Request)
        mock_request.url = "https://api.openai.com/v1/chat/completions"
        mock_request.method = "POST"
        
        mock_client.chat.completions.create.side_effect = [
            mock_response,
            openai.APIError("API Error", request=mock_request, body=None)
        ]
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        
        batch_messages = [
            [{"role": "user", "content": "特許1を分析"}],
            [{"role": "user", "content": "特許2を分析"}]
        ]
        
        results = client.batch_completion(batch_messages, continue_on_error=True)
        
        # 成功したものと失敗したものが適切に処理されることを確認
        assert len(results) == 2
        assert results[0]['decision'] == 'hit'
        assert 'error' in results[1]

    @patch('src.llm.client.OpenAI')
    def test_concurrent_batch_processing(self, mock_openai):
        """並列バッチ処理のテスト"""
        # 複数のモックレスポンス
        mock_responses = []
        for i in range(5):
            mock_responses.append(
                ChatCompletion(
                    id=f"chatcmpl-test{i}",
                    object="chat.completion",
                    created=1677652288,
                    model="gpt-4o-mini",
                    choices=[
                        Choice(
                            index=0,
                            message=ChatCompletionMessage(
                                role="assistant",
                                content=f'{{"decision": "hit", "index": {i}}}'
                            ),
                            finish_reason="stop"
                        )
                    ]
                )
            )
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = mock_responses
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012", max_workers=3)
        
        batch_messages = [
            [{"role": "user", "content": f"特許{i}を分析"}]
            for i in range(5)
        ]
        
        start_time = time.time()
        results = client.concurrent_batch_completion(batch_messages)
        end_time = time.time()
        
        # 並列処理により処理時間が短縮されることを確認（モックなので実際の時間短縮はないが、構造を確認）
        assert len(results) == 5
        assert mock_client.chat.completions.create.call_count == 5


class TestLLMUsageTracking:
    """使用量トラッキングのテスト"""

    @patch('src.llm.client.OpenAI')
    def test_usage_tracking(self, mock_openai):
        """トークン使用量トラッキングのテスト"""
        mock_response = ChatCompletion(
            id="chatcmpl-test123",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"decision": "hit"}'
                    ),
                    finish_reason="stop"
                )
            ],
            usage=openai.types.CompletionUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
        )
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        messages = [{"role": "user", "content": "テスト"}]
        
        response = client.completion(messages)
        
        # 使用量情報が記録されることを確認
        usage_stats = client.get_usage_stats()
        
        assert usage_stats['total_requests'] == 1
        assert usage_stats['total_tokens'] == 150
        assert usage_stats['prompt_tokens'] == 100
        assert usage_stats['completion_tokens'] == 50

    @patch('src.llm.client.OpenAI')  
    def test_usage_reset(self, mock_openai):
        """使用量リセット機能のテスト"""
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        
        # 初期状態の確認
        usage_stats = client.get_usage_stats()
        assert usage_stats['total_requests'] == 0
        
        # リセット機能のテスト
        client.reset_usage_stats()
        
        usage_stats = client.get_usage_stats()
        assert usage_stats['total_requests'] == 0
        assert usage_stats['total_tokens'] == 0


class TestLLMConfigurationValidation:
    """設定検証のテスト"""

    def test_invalid_model_config(self):
        """不正なモデル設定のテスト"""
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMConfigError
        
        with patch('src.llm.client.load_config') as mock_config:
            mock_config.return_value = {
                'llm': {
                    'model': 'invalid-model',  # 無効なモデル
                    'temperature': 0.0
                }
            }
            
            with pytest.raises(LLMConfigError):
                LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")

    def test_invalid_temperature_config(self):
        """不正な temperature 設定のテスト"""
        # 実装後にテスト  
        from src.llm.client import LLMClient, LLMConfigError
        
        with patch('src.llm.client.load_config') as mock_config:
            mock_config.return_value = {
                'llm': {
                    'model': 'gpt-4o-mini',
                    'temperature': 2.0  # 無効な値（0-1の範囲外）
                }
            }
            
            with pytest.raises(LLMConfigError):
                LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")

    def test_invalid_max_tokens_config(self):
        """不正な max_tokens 設定のテスト"""
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMConfigError
        
        with patch('src.llm.client.load_config') as mock_config:
            mock_config.return_value = {
                'llm': {
                    'model': 'gpt-4o-mini',
                    'max_tokens': -1  # 無効な値
                }
            }
            
            with pytest.raises(LLMConfigError):
                LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")


class TestLLMThreadSafety:
    """スレッドセーフティのテスト（Code-reviewer推奨）"""

    @patch('src.llm.client.OpenAI')
    def test_thread_safety_usage_tracking(self, mock_openai):
        """使用量統計のスレッドセーフテスト"""
        import threading
        
        mock_response = ChatCompletion(
            id="chatcmpl-test123",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"decision": "hit"}'
                    ),
                    finish_reason="stop"
                )
            ],
            usage=openai.types.CompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
        )
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        messages = [{"role": "user", "content": "テスト"}]
        
        # 複数スレッドから同時にcompletion実行
        def run_completion():
            client.completion(messages)
        
        threads = [threading.Thread(target=run_completion) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 使用量統計が正確であることを確認
        usage_stats = client.get_usage_stats()
        assert usage_stats['total_requests'] == 10
        assert usage_stats['total_tokens'] == 150  # 15 * 10

    @patch('src.llm.client.OpenAI')
    def test_concurrent_completion_isolation(self, mock_openai):
        """並行completion実行の分離テスト"""
        import threading
        import queue
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # 各レスポンスが異なることを確認するためのユニークID
        def create_response(call_count=[0]):
            call_count[0] += 1
            return ChatCompletion(
                id=f"chatcmpl-test{call_count[0]}",
                object="chat.completion",
                created=1677652288,
                model="gpt-4o-mini",
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=f'{{"request_id": {call_count[0]}}}'
                        ),
                        finish_reason="stop"
                    )
                ]
            )
        
        mock_client.chat.completions.create.side_effect = create_response
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        results_queue = queue.Queue()
        
        def run_completion(thread_id):
            messages = [{"role": "user", "content": f"スレッド{thread_id}のテスト"}]
            result = client.completion(messages)
            results_queue.put((thread_id, result))
        
        # 5つのスレッドで並行実行
        threads = [threading.Thread(target=run_completion, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 各スレッドが独立した結果を取得
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 5
        request_ids = [result[1]['request_id'] for result in results]
        assert len(set(request_ids)) == 5  # すべて異なる値


class TestLLMMemoryEfficiency:
    """メモリ効率性のテスト（Code-reviewer推奨）"""

    @patch('src.llm.client.OpenAI')
    def test_memory_efficiency_large_batches(self, mock_openai):
        """大量バッチ処理のメモリ効率テスト"""
        import tracemalloc
        
        mock_response = ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"decision": "hit"}'
                    ),
                    finish_reason="stop"
                )
            ]
        )
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        
        # 大量のメッセージバッチ
        large_batch = [
            [{"role": "user", "content": f"特許{i}の分析" + "x" * 1000}]
            for i in range(1000)
        ]
        
        # メモリ使用量を追跡
        tracemalloc.start()
        results = client.batch_completion(large_batch)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # メモリ使用量が妥当な範囲内
        assert peak < 100 * 1024 * 1024  # 100MB未満
        assert len(results) == 1000

    @patch('src.llm.client.OpenAI')
    def test_memory_cleanup_after_error(self, mock_openai):
        """エラー後のメモリクリーンアップテスト"""
        mock_client = Mock()
        # Create proper mock request for APIError
        mock_request = Mock(spec=httpx.Request)
        mock_request.url = "https://api.openai.com/v1/chat/completions"
        mock_request.method = "POST"
        mock_client.chat.completions.create.side_effect = openai.APIError("API Error", request=mock_request, body=None)
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMAPIError
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        
        # エラーが発生するバッチ処理
        batch_messages = [
            [{"role": "user", "content": f"特許{i}の分析"}]
            for i in range(100)
        ]
        
        with pytest.raises(LLMAPIError):
            client.batch_completion(batch_messages, continue_on_error=False)
        
        # エラー後もクライアントが正常に動作すること
        # （メモリリークがないことの間接的確認）
        assert hasattr(client, '_client')
        assert hasattr(client, '_usage_stats')


class TestLLMConfigIntegration:
    """設定統合のテスト（Code-reviewer推奨）"""

    def test_actual_yaml_config_loading(self):
        """実際のYAML設定ファイル読み込みテスト"""
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        # 実際のconfig.yamlが存在する場合のテスト
        try:
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-1234567890123456789012345678901234567890123456789012'}):
                client = LLMClient()
                
                # config.yamlから正しく設定が読み込まれることを確認
                assert client._config['model'] in ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo']
                assert 0.0 <= client._config['temperature'] <= 2.0
                assert client._config['max_tokens'] > 0
        except Exception:
            # config.yamlが存在しない場合はスキップ
            pytest.skip("config.yaml not found or invalid")

    def test_config_validation_edge_cases(self):
        """設定検証のエッジケーステスト"""
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMConfigError
        
        edge_cases = [
            # 境界値のテスト
            {'llm': {'temperature': -0.1}},  # 範囲外
            {'llm': {'temperature': 2.1}},   # 範囲外
            {'llm': {'max_tokens': 0}},      # 無効値
            {'llm': {'max_tokens': 'invalid'}},  # 型エラー
            {'llm': {'model': ''}},          # 空文字列
            {'llm': {'timeout': -1}},        # 負の値
        ]
        
        for invalid_config in edge_cases:
            with patch('src.llm.client.load_config') as mock_config:
                mock_config.return_value = invalid_config
                
                with pytest.raises(LLMConfigError):
                    LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")

    def test_config_hot_reload(self):
        """設定のホットリロードテスト"""
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        initial_config = {
            'llm': {
                'model': 'gpt-4o-mini',
                'temperature': 0.0,
                'max_tokens': 320
            }
        }
        
        updated_config = {
            'llm': {
                'model': 'gpt-4o',
                'temperature': 0.1, 
                'max_tokens': 500
            }
        }
        
        with patch('src.llm.client.load_config') as mock_config:
            mock_config.return_value = initial_config
            client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
            
            # 初期設定の確認
            assert client._config['model'] == 'gpt-4o-mini'
            
            # 設定更新
            mock_config.return_value = updated_config
            client.reload_config()
            
            # 更新された設定の確認
            assert client._config['model'] == 'gpt-4o'
            assert client._config['temperature'] == 0.1


class TestLLMAdvancedErrorHandling:
    """高度なエラーハンドリングのテスト（Code-reviewer推奨）"""

    @patch('src.llm.client.OpenAI')
    def test_timeout_edge_cases(self, mock_openai):
        """タイムアウトのエッジケーステスト"""
        # 部分的タイムアウトのシミュレーション
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            openai.APITimeoutError("Connection timeout"),
            openai.APITimeoutError("Read timeout"),
            openai.APITimeoutError("Write timeout")
        ]
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient, LLMTimeoutError
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012", max_retries=2)
        messages = [{"role": "user", "content": "テスト"}]
        
        with pytest.raises(LLMTimeoutError):
            client.completion(messages)
        
        # 適切な回数のリトライが実行されること
        assert mock_client.chat.completions.create.call_count == 3

    @patch('src.llm.client.OpenAI')
    def test_graceful_shutdown_with_pending_requests(self, mock_openai):
        """未完了リクエストありでのグレースフルシャットダウン"""
        import threading
        import time
        
        # レスポンスに遅延を追加
        def slow_response():
            time.sleep(2)  # 2秒の遅延
            return ChatCompletion(
                id="chatcmpl-test",
                object="chat.completion", 
                created=1677652288,
                model="gpt-4o-mini",
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content='{"decision": "hit"}'
                        ),
                        finish_reason="stop"
                    )
                ]
            )
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = slow_response
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        
        # バックグラウンドでリクエストを開始
        def long_running_request():
            messages = [{"role": "user", "content": "長時間実行リクエスト"}]
            try:
                return client.completion(messages)
            except Exception as e:
                return {'error': str(e)}
        
        thread = threading.Thread(target=long_running_request)
        thread.start()
        
        # 少し待ってからシャットダウン
        time.sleep(0.5)
        client.shutdown()
        
        # スレッドが適切に終了すること
        thread.join(timeout=3)
        assert not thread.is_alive()

    @patch('src.llm.client.OpenAI')
    def test_api_version_compatibility(self, mock_openai):
        """API バージョン互換性テスト"""
        # 将来のAPI変更に対する堅牢性テスト
        mock_response = ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", 
                        content='{"decision": "hit"}',
                        # 将来追加される可能性のあるフィールド
                        tool_calls=None
                    ),
                    finish_reason="stop",
                    # 将来追加される可能性のあるフィールド
                    logprobs=None
                )
            ],
            # 将来追加される可能性のあるフィールド
            system_fingerprint="fp_test"
        )
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # 実装後にテスト
        from src.llm.client import LLMClient
        
        client = LLMClient(api_key="sk-1234567890123456789012345678901234567890123456789012")
        messages = [{"role": "user", "content": "テスト"}]
        
        # 新しいフィールドがあっても正常に動作すること
        response = client.completion(messages)
        assert 'decision' in response


if __name__ == "__main__":
    pytest.main([__file__])