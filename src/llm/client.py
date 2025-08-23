"""
client.py - OpenAI APIクライアント

特許分析用のLLMクライアント機能。Code-reviewerの推奨に基づく
セキュリティ・パフォーマンス・品質重視の実装。
"""

import os
import json
import time
import random
import threading
import re
from typing import Dict, Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import lru_cache
import hashlib
from loguru import logger
import yaml
from pathlib import Path

try:
    import openai
    from openai.types.chat import ChatCompletion
    from openai import OpenAI
except ImportError:
    logger.error("OpenAI library not installed. Run: pip install openai")
    raise

# エラー階層 - Code-reviewer推奨による改善
class LLMClientError(Exception):
    """LLMクライアントのベース例外"""
    pass

class LLMConfigError(LLMClientError):
    """設定関連エラー"""
    pass

class LLMAPIError(LLMClientError):
    """OpenAI API関連エラー"""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error

class LLMRateLimitError(LLMAPIError):
    """レート制限エラー"""
    pass

class LLMTimeoutError(LLMAPIError):
    """タイムアウトエラー"""
    pass

class LLMAuthenticationError(LLMAPIError):
    """認証エラー"""
    pass

class LLMResponseError(LLMAPIError):
    """レスポンス解析エラー"""
    pass

# 定数クラス（Code-reviewer推奨による整理）
class ModelConstants:
    ALLOWED = ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo']
    DEFAULT = 'gpt-4o-mini'

class DefaultConfig:
    TEMPERATURE = 0.0
    MAX_TOKENS = 320
    TIMEOUT = 30.0
    MAX_RETRIES = 3
    MAX_WORKERS = 5

class ValidationConstants:
    MAX_MESSAGE_LENGTH = 100000
    WARN_MESSAGE_LENGTH = 50000  # 警告しきい値
    MIN_API_KEY_LENGTH = 51  # sk- + 48文字
    MAX_BATCH_SIZE = 100
    API_KEY_PATTERN = re.compile(r'^sk-[a-zA-Z0-9_-]{20,}$')  # OpenAI API keyパターン（プロジェクトキー対応）

class RetryConstants:
    BASE_DELAY = 1.0
    MAX_DELAY = 60.0
    JITTER_RANGE = (0.1, 0.9)


def sanitize_error_content(content: str, max_length: int = 50) -> str:
    """
    エラー情報の漏洩防止（Code-reviewer推奨）
    
    Args:
        content: サニタイズ対象のコンテンツ
        max_length: 表示する最大文字数
        
    Returns:
        サニタイズされたコンテンツ
    """
    if len(content) > max_length:
        return f"<{len(content)} chars of content>"
    return content.replace('\n', ' ').replace('\r', ' ')


def get_api_key(api_key: Optional[str] = None) -> str:
    """
    OpenAI API キーを安全に取得・検証（Code-reviewer推奨による改善）
    
    Args:
        api_key: 指定されたAPI キー（オプション）
        
    Returns:
        検証済みのAPI キー
        
    Raises:
        LLMAuthenticationError: API キーが無効な場合
    """
    if api_key:
        # より厳密なAPI キー形式検証
        if not ValidationConstants.API_KEY_PATTERN.match(api_key):
            raise LLMAuthenticationError("Invalid API key format")
        return api_key
    
    # 環境変数から取得
    env_key = os.getenv('OPENAI_API_KEY')
    if env_key:
        if not ValidationConstants.API_KEY_PATTERN.match(env_key):
            raise LLMAuthenticationError("Invalid API key format in environment")
        return env_key
    
    raise LLMAuthenticationError("No valid API key found. Set OPENAI_API_KEY or provide api_key parameter")


def load_config() -> Dict[str, Any]:
    """設定ファイルを読み込む"""
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


def validate_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM設定の検証・サニタイズ（Code-reviewer推奨による改善）
    
    Args:
        config: 設定辞書
        
    Returns:
        検証済み設定辞書
        
    Raises:
        LLMConfigError: 設定が無効な場合
    """
    llm_config = config.get('llm', {})
    
    # モデル検証
    model = llm_config.get('model', ModelConstants.DEFAULT)
    if model not in ModelConstants.ALLOWED:
        raise LLMConfigError(f"Invalid model: {model}. Allowed: {ModelConstants.ALLOWED}")
    
    # 温度検証
    temperature = llm_config.get('temperature', DefaultConfig.TEMPERATURE)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        raise LLMConfigError(f"Invalid temperature: {temperature}. Must be 0-2")
    
    # 最大トークン数検証
    max_tokens = llm_config.get('max_tokens', DefaultConfig.MAX_TOKENS)
    if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 4096:
        raise LLMConfigError(f"Invalid max_tokens: {max_tokens}. Must be 1-4096")
    
    # タイムアウト検証
    timeout = llm_config.get('timeout', DefaultConfig.TIMEOUT)
    if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 300:
        raise LLMConfigError(f"Invalid timeout: {timeout}. Must be 0-300 seconds")
    
    return {
        'model': model,
        'temperature': float(temperature),
        'max_tokens': int(max_tokens),
        'timeout': float(timeout),
        'response_format': llm_config.get('response_format', 'json')
    }


def validate_messages(messages: List[Dict[str, str]], truncate_long: bool = False) -> List[Dict[str, str]]:
    """
    メッセージ構造と内容の検証（Code-reviewer推奨による改善）
    
    Args:
        messages: メッセージのリスト
        truncate_long: 長いメッセージを切り詰めるか
        
    Returns:
        検証・調整されたメッセージリスト
        
    Raises:
        ValueError: メッセージ構造が無効な場合
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Messages must be a non-empty list")
    
    validated_messages = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Message {i} must be a dictionary")
        
        role = msg.get('role')
        if role not in ['system', 'user', 'assistant']:
            raise ValueError(f"Invalid role in message {i}: {role}")
        
        content = msg.get('content', '')
        if not isinstance(content, str):
            raise ValueError(f"Message {i} content must be a string")
        
        # 長いコンテンツの処理
        if len(content) > ValidationConstants.MAX_MESSAGE_LENGTH:
            if truncate_long:
                content = content[:ValidationConstants.MAX_MESSAGE_LENGTH] + "...[TRUNCATED]"
                logger.warning(f"Message {i} truncated from {len(msg.get('content', ''))} to {len(content)} chars")
            else:
                raise ValueError(f"Message {i} content too long: {len(content)} chars")
        elif len(content) > ValidationConstants.WARN_MESSAGE_LENGTH:
            logger.warning(f"Message {i} is large ({len(content)} chars), consider splitting")
        
        validated_messages.append({**msg, 'content': content})
    
    return validated_messages


def exponential_backoff_with_jitter(attempt: int, base_delay: float = RetryConstants.BASE_DELAY) -> float:
    """
    ジッタ付き指数バックオフ遅延計算（Code-reviewer推奨）
    
    Args:
        attempt: 試行回数
        base_delay: ベース遅延時間
        
    Returns:
        計算された遅延時間
    """
    delay = base_delay * (2 ** attempt)
    jitter_min, jitter_max = RetryConstants.JITTER_RANGE
    jitter = random.uniform(jitter_min, jitter_max) * delay
    return min(delay + jitter, RetryConstants.MAX_DELAY)


def should_retry(exception: Exception) -> bool:
    """
    例外がリトライ対象かを判定
    
    Args:
        exception: 発生した例外
        
    Returns:
        リトライすべきかの真偽値
    """
    retryable_exceptions = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError
    )
    return isinstance(exception, retryable_exceptions)


class LLMClient:
    """
    OpenAI APIクライアント
    
    特許分析に特化したLLMクライアント。セキュリティ、パフォーマンス、
    エラーハンドリングを重視した実装。
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        max_retries: int = DefaultConfig.MAX_RETRIES,
        max_workers: int = DefaultConfig.MAX_WORKERS
    ):
        """
        LLMクライアントの初期化
        
        Args:
            api_key: OpenAI API キー（オプション、環境変数から自動取得）
            max_retries: 最大リトライ回数
            max_workers: 並列処理の最大ワーカー数
            
        Raises:
            LLMAuthenticationError: API キーが無効な場合
            LLMConfigError: 設定が無効な場合
        """
        logger.debug("Initializing LLM client")
        
        # API キー取得・検証
        self._api_key = get_api_key(api_key)
        
        # OpenAIクライアント初期化
        self._client = OpenAI(
            api_key=self._api_key,
            timeout=DefaultConfig.TIMEOUT,
            max_retries=0  # 手動でリトライ制御
        )
        
        # 設定読み込み・検証
        config = load_config()
        self._config = validate_llm_config(config)
        
        # リトライ・並列処理設定
        self._max_retries = max_retries
        self._max_workers = max_workers
        
        # 使用量統計（スレッドセーフ）
        self._usage_lock = threading.Lock()
        self._usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'failed_requests': 0
        }
        
        # シャットダウンフラグ
        self._shutdown = False
        
        # セキュリティ強化：API キー漏洩防止（Code-reviewer推奨）
        api_key_hint = f"{'*' * 8}...{self._api_key[-4:]}" if len(self._api_key) > 8 else "hidden"
        logger.info(f"LLM client initialized with model: {self._config['model']}, API key: {api_key_hint}")
    
    def completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        チャット完了を生成
        
        Args:
            messages: メッセージのリスト
            model: 使用するモデル（オプション）
            temperature: 温度パラメータ（オプション）
            max_tokens: 最大トークン数（オプション）
            response_format: レスポンス形式（オプション）
            
        Returns:
            解析されたJSONレスポンス
            
        Raises:
            LLMAuthenticationError: 認証エラー
            LLMRateLimitError: レート制限エラー
            LLMTimeoutError: タイムアウトエラー
            LLMResponseError: レスポンス解析エラー
            LLMAPIError: その他のAPIエラー
        """
        if self._shutdown:
            raise LLMClientError("Client is shutting down")
        
        # 入力検証（Code-reviewer推奨による改善）
        validated_messages = validate_messages(messages)
        
        # パラメータ設定
        params = {
            'model': model or self._config['model'],
            'messages': validated_messages,
            'temperature': temperature if temperature is not None else self._config['temperature'],
            'max_tokens': max_tokens or self._config['max_tokens']
        }
        
        # JSON形式レスポンス設定
        if response_format == 'json' or self._config.get('response_format') == 'json':
            params['response_format'] = {"type": "json_object"}
        
        # リトライ付きで実行
        for attempt in range(self._max_retries + 1):
            try:
                logger.debug(f"Making completion request (attempt {attempt + 1})")
                response = self._client.chat.completions.create(**params)
                
                # 使用量統計を更新
                self._update_usage_stats(response)
                
                # レスポンス処理
                return self._process_response(response)
                
            except openai.AuthenticationError as e:
                # 認証エラーはリトライしない
                logger.error(f"Authentication error: {e}")
                self._increment_failed_requests()
                raise LLMAuthenticationError(f"Authentication failed: {e}")
            
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit (attempt {attempt + 1}): {e}")
                if attempt < self._max_retries:
                    delay = exponential_backoff_with_jitter(attempt)
                    logger.debug(f"Waiting {delay:.2f}s before retry")
                    time.sleep(delay)
                    continue
                else:
                    self._increment_failed_requests()
                    raise LLMRateLimitError(f"Rate limit exceeded after {self._max_retries} retries")
            
            except openai.APITimeoutError as e:
                logger.warning(f"Timeout error (attempt {attempt + 1}): {e}")
                if attempt < self._max_retries and should_retry(e):
                    delay = exponential_backoff_with_jitter(attempt)
                    time.sleep(delay)
                    continue
                else:
                    self._increment_failed_requests()
                    raise LLMTimeoutError(f"Request timeout: {e}")
            
            except LLMResponseError as e:
                # レスポンス解析エラーはリトライしない
                logger.error(f"Response parsing error: {e}")
                self._increment_failed_requests()
                raise e
            
            except Exception as e:
                logger.error(f"Unexpected API error (attempt {attempt + 1}): {e}")
                if attempt < self._max_retries and should_retry(e):
                    delay = exponential_backoff_with_jitter(attempt)
                    time.sleep(delay)
                    continue
                else:
                    self._increment_failed_requests()
                    raise LLMAPIError(f"API error: {e}", e)
    
    def _process_response(self, response: ChatCompletion) -> Dict[str, Any]:
        """
        APIレスポンスを処理
        
        Args:
            response: OpenAI APIレスポンス
            
        Returns:
            解析されたレスポンス
            
        Raises:
            LLMResponseError: レスポンス解析エラー
        """
        try:
            content = response.choices[0].message.content
            if not content:
                raise LLMResponseError("Empty response content")
            
            # JSON形式の場合は解析
            if self._config.get('response_format') == 'json':
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    # セキュリティ強化：エラー情報漏洩防止（Code-reviewer推奨）
                    logger.error(f"JSON parsing error: {e}, content: {sanitize_error_content(content)}")
                    raise LLMResponseError(f"Invalid JSON response: {e}")
            
            return {'content': content}
            
        except (IndexError, AttributeError) as e:
            raise LLMResponseError(f"Invalid response structure: {e}")
    
    def _update_usage_stats(self, response: ChatCompletion) -> None:
        """使用量統計を更新（スレッドセーフ）"""
        with self._usage_lock:
            self._usage_stats['total_requests'] += 1
            
            if hasattr(response, 'usage') and response.usage:
                self._usage_stats['total_tokens'] += response.usage.total_tokens
                self._usage_stats['prompt_tokens'] += response.usage.prompt_tokens
                self._usage_stats['completion_tokens'] += response.usage.completion_tokens
    
    def _increment_failed_requests(self) -> None:
        """失敗リクエスト数を増加（スレッドセーフ）"""
        with self._usage_lock:
            self._usage_stats['failed_requests'] += 1
    
    def get_usage_stats(self) -> Dict[str, int]:
        """
        使用量統計を取得
        
        Returns:
            使用量統計辞書
        """
        with self._usage_lock:
            return self._usage_stats.copy()
    
    def reset_usage_stats(self) -> None:
        """使用量統計をリセット"""
        with self._usage_lock:
            self._usage_stats = {
                'total_requests': 0,
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'failed_requests': 0
            }
            logger.debug("Usage statistics reset")
    
    def batch_completion(
        self, 
        batch_messages: List[List[Dict[str, str]]], 
        continue_on_error: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        複数メッセージのバッチ処理
        
        Args:
            batch_messages: メッセージバッチのリスト
            continue_on_error: エラー発生時に続行するか
            **kwargs: completion()に渡される追加引数
            
        Returns:
            結果のリスト
        """
        if len(batch_messages) > ValidationConstants.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size too large: {len(batch_messages)}")
        
        results = []
        
        for i, messages in enumerate(batch_messages):
            try:
                result = self.completion(messages, **kwargs)
                results.append(result)
            except Exception as e:
                if continue_on_error:
                    logger.warning(f"Batch item {i} failed: {e}")
                    # セキュリティ強化：機密情報の漏洩防止
                    results.append({'error': 'Request failed', 'error_type': type(e).__name__})
                else:
                    logger.error(f"Batch processing failed at item {i}: {e}")
                    raise
        
        logger.info(f"Batch processing completed: {len(results)} items")
        return results
    
    def concurrent_batch_completion(
        self, 
        batch_messages: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        並行バッチ処理
        
        Args:
            batch_messages: メッセージバッチのリスト
            **kwargs: completion()に渡される追加引数
            
        Returns:
            結果のリスト（元の順序を保持）
        """
        if len(batch_messages) > ValidationConstants.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size too large: {len(batch_messages)}")
        
        results = [None] * len(batch_messages)
        
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # 各メッセージの処理を送信
            future_to_index = {
                executor.submit(self._safe_completion, messages, **kwargs): i
                for i, messages in enumerate(batch_messages)
            }
            
            # 結果を回収
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Concurrent batch item {index} failed: {e}")
                    results[index] = {'error': str(e)}
        
        logger.info(f"Concurrent batch processing completed: {len(results)} items")
        return results
    
    def _safe_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """エラーハンドリング付きの安全なcompletion"""
        try:
            return self.completion(messages, **kwargs)
        except Exception as e:
            return {'error': str(e), 'messages': messages}
    
    def reload_config(self) -> None:
        """設定のリロード"""
        logger.debug("Reloading configuration")
        config = load_config()
        self._config = validate_llm_config(config)
        logger.info(f"Configuration reloaded: {self._config}")
    
    def shutdown(self) -> None:
        """グレースフルシャットダウン"""
        logger.info("Initiating LLM client shutdown")
        self._shutdown = True
        
        # 既存のリクエストの完了を待つ
        # （実際のリクエスト追跡は複雑になるため、簡単な実装）
        time.sleep(1)
        
        logger.info("LLM client shutdown completed")
    
    def __del__(self):
        """デストラクタ"""
        try:
            if hasattr(self, '_shutdown') and not self._shutdown:
                self.shutdown()
        except Exception:
            # Windows環境でのハンドルエラーを無視
            pass


if __name__ == "__main__":
    # 簡単なテスト
    try:
        client = LLMClient()
        
        test_messages = [
            {"role": "system", "content": "You are a patent analysis assistant."},
            {"role": "user", "content": "Analyze this patent claim for liquid separation equipment."}
        ]
        
        response = client.completion(test_messages)
        print("Test completion successful")
        print(f"Response: {response}")
        
        # 使用量統計の表示
        stats = client.get_usage_stats()
        print(f"Usage stats: {stats}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        logger.error(f"Test execution failed: {e}")