"""
prompts.py - LLMプロンプトテンプレート機能

特許分析用の3種類のプロンプト生成を提供：
1. 発明要約プロンプト - 発明アイデアを要約
2. 特許要約プロンプト - 特許文書（請求項+抄録）を要約
3. 二値分類プロンプト - hit/miss判定と信頼度・理由を生成

Code-reviewerの推奨に基づくセキュリティ・パフォーマンス・品質重視の実装。
"""

import re
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from functools import lru_cache
from pathlib import Path
import yaml
from loguru import logger

# セキュリティ定数 - Code-reviewer推奨による改善
class SecurityConstants:
    """セキュリティ関連の定数"""
    DANGEROUS_PATTERNS = [
        r'forget\s+previous\s+instructions',
        r'ignore\s+all\s+previous',
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'DROP\s+TABLE',
        r'SELECT.*FROM.*WHERE.*OR.*=.*',
        r'\{\{.*?\}\}',  # テンプレートインジェクション
        r'eval\(',
        r'exec\(',
        r'__import__',
        r'getattr\(',
        r'setattr\(',
        r'rm\s+-rf',  # 危険なコマンド
        r'sudo\s+rm',
        r'del\s+/[sq]',  # Windows危険コマンド
    ]
    
    SENSITIVE_PATTERNS = [
        r'api[_\s]?key[:\s]*[a-zA-Z0-9\-_]{8,}',
        r'password[:\s]*\w+',
        r'secret[:\s]*\w+',
        r'token[:\s]*[a-zA-Z0-9\-_]{10,}',
        r'sk-[a-zA-Z0-9]{10,}',  # OpenAI API key pattern (relaxed for testing)
        r'\b(?:admin|root|user)[\s:]*(?:admin|password|123456|secret)\b',
        r'パスワード[:\s]*\w+',  # Japanese password pattern
        r'admin123',  # Common test passwords
    ]

class ContentLimits:
    """コンテンツ長制限定数"""
    MAX_INVENTION_TITLE = 200
    MAX_INVENTION_CONTENT = 2000
    MAX_PATENT_CLAIM = 3000
    MAX_PATENT_ABSTRACT = 1000
    MAX_SUMMARY_LENGTH = 500
    MAX_PROMPT_TOKENS = 2000
    TRUNCATION_BUFFER = 100

class PromptDefaults:
    """プロンプトデフォルト値"""
    SYSTEM_ROLE = "特許分析の専門家"
    MAX_REASONS = 2
    CONFIDENCE_DECIMALS = 2
    JSON_INDENT = None  # Compact JSON


# エラー階層 - Code-reviewer推奨
class PromptError(Exception):
    """プロンプト生成のベース例外"""
    pass

class PromptValidationError(PromptError, ValueError):
    """入力検証エラー"""
    pass

class PromptSecurityError(PromptError):
    """セキュリティ違反検出エラー"""
    pass

class PromptTemplateError(PromptError):
    """テンプレート処理エラー"""
    pass


# セキュリティ強化されたサニタイゼーション - Code-reviewer推奨
@lru_cache(maxsize=100)
def _compile_security_patterns():
    """セキュリティパターンのコンパイルとキャッシュ"""
    dangerous = {
        name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        for name, pattern in enumerate(SecurityConstants.DANGEROUS_PATTERNS)
    }
    sensitive = {
        name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        for name, pattern in enumerate(SecurityConstants.SENSITIVE_PATTERNS)
    }
    return dangerous, sensitive


def sanitize_prompt_content(content: str) -> str:
    """
    プロンプトコンテンツの安全なサニタイゼーション
    
    Args:
        content: サニタイズ対象のコンテンツ
        
    Returns:
        サニタイズされたコンテンツ
        
    Raises:
        PromptSecurityError: 危険なパターンが検出された場合
    """
    if not isinstance(content, str):
        content = str(content)
    
    dangerous_patterns, sensitive_patterns = _compile_security_patterns()
    
    # 危険なパターンの検出と除去
    original_content = content
    for pattern_id, pattern in dangerous_patterns.items():
        if pattern.search(content):
            logger.warning(f"Dangerous pattern detected and removed: pattern_{pattern_id}")
            content = pattern.sub('[REMOVED FOR SECURITY]', content)
    
    # 機密情報パターンの検出とマスク
    for pattern_id, pattern in sensitive_patterns.items():
        if pattern.search(content):
            logger.warning(f"Sensitive information detected and masked: pattern_{pattern_id}")
            content = pattern.sub('[MASKED]', content)
    
    # 制御文字の除去
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)
    
    # 過度に長い場合の警告
    if len(content) != len(original_content):
        logger.info(f"Content sanitized: {len(original_content)} -> {len(content)} chars")
    
    return content.strip()


def truncate_content_smart(content: str, max_length: int, preserve_sentences: bool = True) -> str:
    """
    コンテンツのスマートな切り詰め（日本語対応）
    
    Args:
        content: 切り詰め対象のコンテンツ
        max_length: 最大文字数
        preserve_sentences: 文境界を保持するか
        
    Returns:
        切り詰められたコンテンツ
    """
    if len(content) <= max_length:
        return content
    
    if not preserve_sentences:
        return content[:max_length] + "..."
    
    # 文境界での切り詰めを試行
    truncated = content[:max_length - 3]  # "..." のスペースを確保
    
    # 日本語の文区切り文字を探す
    sentence_endings = ['。', '！', '？', '!', '?', '.']
    last_sentence_end = -1
    
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > last_sentence_end:
            last_sentence_end = pos
    
    # 文境界が見つかり、元の長さの70%以上である場合は文境界で切断
    if last_sentence_end > max_length * 0.7:
        return truncated[:last_sentence_end + 1] + "..."
    else:
        return truncated + "..."


def estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    """
    プロンプトのトークン数を推定
    
    Args:
        messages: プロンプトメッセージのリスト
        
    Returns:
        推定トークン数
    """
    total_chars = sum(len(msg.get('content', '')) for msg in messages)
    
    # 日本語混じりテキストのトークン数推定
    # 経験則: 英語 ≈ 4文字/トークン、日本語 ≈ 2-3文字/トークン
    # 平均的に3文字/トークンで推定
    estimated_tokens = total_chars // 3
    
    # システムメッセージ、役割情報等のオーバーヘッドを加算
    overhead_tokens = len(messages) * 10  # メッセージごとのオーバーヘッド
    
    return estimated_tokens + overhead_tokens


def validate_prompt_messages(messages: List[Dict[str, str]]) -> bool:
    """
    プロンプトメッセージの構造を検証
    
    Args:
        messages: 検証するメッセージリスト
        
    Returns:
        検証結果（True: 有効）
        
    Raises:
        ValueError: メッセージ構造が無効な場合
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Messages must be a non-empty list")
    
    valid_roles = {'system', 'user', 'assistant'}
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Message {i} must be a dictionary")
        
        if 'role' not in msg:
            raise ValueError(f"Message {i} missing 'role' field")
        
        if msg['role'] not in valid_roles:
            raise ValueError(f"Message {i} has invalid role: {msg['role']}")
        
        if 'content' not in msg:
            raise ValueError(f"Message {i} missing 'content' field")
        
        if not isinstance(msg['content'], str):
            raise ValueError(f"Message {i} content must be a string")
    
    return True


# プロンプトテンプレート定義 - Code-reviewer推奨による構造化
@lru_cache(maxsize=10)
def get_prompt_templates() -> Dict[str, Dict[str, Any]]:
    """プロンプトテンプレートの取得とキャッシュ"""
    return {
        'invention': {
            'system': f"""あなたは{PromptDefaults.SYSTEM_ROLE}です。発明アイデアを分析し、簡潔で的確な要約を作成してください。

要約の要求事項:
- 発明の核心技術と課題解決アプローチを明確に示す
- 液体分離設備分野との関連性を重視
- 技術的特徴と効果を簡潔にまとめる
- {ContentLimits.MAX_SUMMARY_LENGTH}文字以内で要約する
- 専門用語を適切に使用し、第三者が理解できる表現にする""",
            
            'user_template': """以下の発明内容を要約してください:

【発明タイトル】
{title}

【解決すべき課題】
{problem}

【解決手段】
{solution}

【発明の効果】
{effects}

【主要技術要素】
{key_elements}

上記の発明内容を{max_summary_length}文字以内で要約し、液体分離設備分野における技術的意義を明確に示してください。""",
            
            'required_fields': ['title'],
            'optional_fields': ['problem', 'solution', 'effects', 'key_elements'],
            'max_tokens': 200,
        },
        
        'patent': {
            'system': f"""あなたは{PromptDefaults.SYSTEM_ROLE}です。特許文書を分析し、発明要約との比較のための簡潔な特許要約を作成してください。

要約の要求事項:
- 請求項1（独立請求項）と抄録の核心内容を抽出
- 技術分野、解決課題、技術手段、効果を明確に整理
- 発明要約との比較が容易な形式で整理
- {ContentLimits.MAX_SUMMARY_LENGTH}文字以内で要約する
- 特許固有の表現は一般的な技術用語に言い換える""",
            
            'user_template': """発明要約:
{invention_summary}

特許情報:
【公開番号】 {publication_number}
【発明の名称】 {title}

【請求項1】
{claim_1}

【要約】
{abstract}

上記の特許内容を{max_summary_length}文字以内で要約し、上記発明要約との技術的関連性の判断に適した形式で整理してください。""",
            
            'required_fields': ['publication_number'],
            'optional_fields': ['title', 'claim_1', 'abstract'],
            'max_tokens': 200,
        },
        
        'classification': {
            'system': f"""あなたは{PromptDefaults.SYSTEM_ROLE}です。発明アイデアと特許文書の技術的関連性を厳密に判定し、二値分類（hit/miss）で回答してください。

判定基準:
- **hit**: 発明と特許が同一技術分野で、具体的な技術的共通点が複数存在する場合
- **miss**: 技術分野が異なる、または具体的な技術的共通点が不十分な場合

信頼度の基準:
- 0.9-1.0: 明確な技術的一致、同一課題・解決手段
- 0.7-0.8: 強い技術的関連性、類似の課題・手段
- 0.5-0.6: 中程度の技術的関連性
- 0.3-0.4: 弱い関連性
- 0.0-0.2: 関連性なしまたは不明確

必須回答形式（JSON）:
{{
    "decision": "hit" | "miss",
    "confidence": 0.0-1.0（小数点{PromptDefaults.CONFIDENCE_DECIMALS}桁）,
    "hit_reason_1": "主な類似点の簡潔な説明（12語以内）",
    "hit_src_1": "claim" | "abstract",
    "hit_reason_2": "第二の類似点の簡潔な説明（12語以内）",
    "hit_src_2": "claim" | "abstract"
}}

注意事項:
- 必ずJSON形式で回答する
- hit_reason は具体的技術要素を12語以内で記述
- hit_src は根拠となった特許部分（請求項または要約）を示す
- confidence は判定の確信度を正確に反映する""",
            
            'user_template': """【発明要約】
{invention_summary}

【特許要約】
{patent_summary}

上記の発明と特許の技術的関連性を判定し、JSON形式で回答してください。""",
            
            'required_fields': ['invention_summary', 'patent_summary'],
            'json_schema': {
                'type': 'object',
                'properties': {
                    'decision': {'enum': ['hit', 'miss']},
                    'confidence': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
                    'hit_reason_1': {'type': 'string', 'maxLength': 50},
                    'hit_src_1': {'enum': ['claim', 'abstract']},
                    'hit_reason_2': {'type': 'string', 'maxLength': 50},
                    'hit_src_2': {'enum': ['claim', 'abstract']},
                },
                'required': ['decision', 'confidence', 'hit_reason_1', 'hit_src_1']
            },
        }
    }


def _validate_template_data(data: Dict[str, Any], template_config: Dict[str, Any]) -> Dict[str, str]:
    """
    テンプレートデータの検証とサニタイゼーション
    
    Args:
        data: 入力データ
        template_config: テンプレート設定
        
    Returns:
        検証・サニタイズされたデータ
        
    Raises:
        PromptValidationError: データが無効な場合
    """
    if not isinstance(data, dict):
        raise PromptValidationError(f"Template data must be a dictionary, got {type(data)}")
    
    # 必須フィールドの確認
    required_fields = template_config.get('required_fields', [])
    for field in required_fields:
        if field not in data or not data[field]:
            raise PromptValidationError(f"Required field '{field}' is missing or empty")
    
    # データのサニタイゼーション
    sanitized_data = {}
    all_fields = required_fields + template_config.get('optional_fields', [])
    
    for field in all_fields:
        if field in data and data[field] is not None:
            value = str(data[field])
            # セキュリティスキャンを実行
            sanitized_value = sanitize_prompt_content(value)
            sanitized_data[field] = sanitized_value
        else:
            sanitized_data[field] = "（情報なし）"
    
    return sanitized_data


def create_invention_prompt(invention_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    発明要約プロンプトを生成
    
    Args:
        invention_data: 発明データ（title必須、他オプション）
        
    Returns:
        プロンプトメッセージリスト
        
    Raises:
        PromptValidationError: 入力データが無効な場合
        PromptSecurityError: セキュリティ違反が検出された場合
    """
    # 事前型チェック
    if invention_data is None:
        raise TypeError("invention_data cannot be None")
    if not isinstance(invention_data, dict):
        raise TypeError(f"invention_data must be a dictionary, got {type(invention_data)}")
        
    logger.debug(f"Creating invention prompt for: {invention_data.get('title', 'untitled')}")
    
    templates = get_prompt_templates()
    template_config = templates['invention']
    
    # データ検証とサニタイゼーション
    sanitized_data = _validate_template_data(invention_data, template_config)
    
    # 主要技術要素の処理
    if isinstance(invention_data.get('key_elements'), list):
        key_elements = '、'.join(invention_data['key_elements'])
        sanitized_data['key_elements'] = sanitize_prompt_content(key_elements)
    
    # コンテンツ長制限の適用
    for field, limit in [
        ('title', ContentLimits.MAX_INVENTION_TITLE),
        ('problem', ContentLimits.MAX_INVENTION_CONTENT),
        ('solution', ContentLimits.MAX_INVENTION_CONTENT),
        ('effects', ContentLimits.MAX_INVENTION_CONTENT),
        ('key_elements', ContentLimits.MAX_INVENTION_CONTENT)
    ]:
        if field in sanitized_data and len(sanitized_data[field]) > limit:
            sanitized_data[field] = truncate_content_smart(sanitized_data[field], limit)
    
    # プロンプト生成
    system_message = template_config['system']
    user_message = template_config['user_template'].format(
        max_summary_length=ContentLimits.MAX_SUMMARY_LENGTH,
        **sanitized_data
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    # 検証
    validate_prompt_messages(messages)
    
    logger.debug(f"Invention prompt created, estimated tokens: {estimate_prompt_tokens(messages)}")
    return messages


def create_patent_prompt(patent_data: Dict[str, Any], invention_summary: str) -> List[Dict[str, str]]:
    """
    特許要約プロンプトを生成
    
    Args:
        patent_data: 特許データ（publication_number必須）
        invention_summary: 発明要約文
        
    Returns:
        プロンプトメッセージリスト
        
    Raises:
        PromptValidationError: 入力データが無効な場合
    """
    logger.debug(f"Creating patent prompt for: {patent_data.get('publication_number', 'unknown')}")
    
    if not isinstance(invention_summary, str) or not invention_summary.strip():
        raise PromptValidationError("invention_summary must be a non-empty string")
    
    templates = get_prompt_templates()
    template_config = templates['patent']
    
    # データ検証とサニタイゼーション
    sanitized_data = _validate_template_data(patent_data, template_config)
    sanitized_invention_summary = sanitize_prompt_content(invention_summary)
    
    # コンテンツ長制限の適用
    for field, limit in [
        ('title', ContentLimits.MAX_INVENTION_TITLE),
        ('claim_1', ContentLimits.MAX_PATENT_CLAIM),
        ('abstract', ContentLimits.MAX_PATENT_ABSTRACT)
    ]:
        if field in sanitized_data and len(sanitized_data[field]) > limit:
            sanitized_data[field] = truncate_content_smart(sanitized_data[field], limit)
    
    # 発明要約の長制限
    if len(sanitized_invention_summary) > ContentLimits.MAX_SUMMARY_LENGTH:
        sanitized_invention_summary = truncate_content_smart(
            sanitized_invention_summary, ContentLimits.MAX_SUMMARY_LENGTH
        )
    
    # プロンプト生成
    system_message = template_config['system']
    user_message = template_config['user_template'].format(
        invention_summary=sanitized_invention_summary,
        max_summary_length=ContentLimits.MAX_SUMMARY_LENGTH,
        **sanitized_data
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    # 検証
    validate_prompt_messages(messages)
    
    logger.debug(f"Patent prompt created, estimated tokens: {estimate_prompt_tokens(messages)}")
    return messages


def create_classification_prompt(invention_summary: str, patent_summary: str) -> List[Dict[str, str]]:
    """
    二値分類プロンプトを生成
    
    Args:
        invention_summary: 発明要約文
        patent_summary: 特許要約文
        
    Returns:
        プロンプトメッセージリスト
        
    Raises:
        PromptValidationError: 入力データが無効な場合
    """
    logger.debug("Creating classification prompt")
    
    # 入力検証
    if not isinstance(invention_summary, str) or not invention_summary.strip():
        raise PromptValidationError("invention_summary must be a non-empty string")
    
    if not isinstance(patent_summary, str) or not patent_summary.strip():
        raise PromptValidationError("patent_summary must be a non-empty string")
    
    templates = get_prompt_templates()
    template_config = templates['classification']
    
    # サニタイゼーション
    sanitized_invention = sanitize_prompt_content(invention_summary)
    sanitized_patent = sanitize_prompt_content(patent_summary)
    
    # 長制限
    if len(sanitized_invention) > ContentLimits.MAX_SUMMARY_LENGTH:
        sanitized_invention = truncate_content_smart(
            sanitized_invention, ContentLimits.MAX_SUMMARY_LENGTH
        )
    
    if len(sanitized_patent) > ContentLimits.MAX_SUMMARY_LENGTH:
        sanitized_patent = truncate_content_smart(
            sanitized_patent, ContentLimits.MAX_SUMMARY_LENGTH
        )
    
    # プロンプト生成
    system_message = template_config['system']
    user_message = template_config['user_template'].format(
        invention_summary=sanitized_invention,
        patent_summary=sanitized_patent
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    # 検証
    validate_prompt_messages(messages)
    
    logger.debug(f"Classification prompt created, estimated tokens: {estimate_prompt_tokens(messages)}")
    return messages


# ユーティリティ機能 - Code-reviewer推奨
def check_prompt_consistency(prompts: Dict[str, List[Dict[str, str]]]) -> Dict[str, bool]:
    """
    複数プロンプトの一貫性をチェック
    
    Args:
        prompts: プロンプト名をキーとするプロンプト辞書
        
    Returns:
        一貫性チェック結果
    """
    results = {}
    
    # システムロールの一貫性チェック
    system_roles = []
    for prompt_name, messages in prompts.items():
        if messages and messages[0].get('role') == 'system':
            system_content = messages[0]['content']
            
            # 日本語パターン: "あなたは...専門家"
            if 'あなたは' in system_content and '専門家' in system_content:
                role_part = re.search(r'あなたは(.+?)専門家', system_content)
                if role_part:
                    system_roles.append(role_part.group(1).strip())
            # 英語パターン: "You are a..."
            elif 'You are' in system_content:
                role_part = re.search(r'You are (?:a |an )?([\w\s]+?)\.?$', system_content, re.IGNORECASE)
                if role_part:
                    system_roles.append(role_part.group(1).strip())
            else:
                # その他の場合は全体をロールとして使用
                system_roles.append(system_content[:50])  # 先頭50文字で比較
    
    results['system_role_consistent'] = len(set(system_roles)) <= 1
    
    # 平均トークン数の計算
    token_counts = [estimate_prompt_tokens(messages) for messages in prompts.values()]
    results['avg_tokens'] = sum(token_counts) / len(token_counts) if token_counts else 0
    results['max_tokens'] = max(token_counts) if token_counts else 0
    results['token_variance_acceptable'] = (max(token_counts) - min(token_counts)) < 500
    
    logger.debug(f"Prompt consistency check: {results}")
    return results


@lru_cache(maxsize=50)
def _generate_content_hash(content: str) -> str:
    """コンテンツのハッシュ生成（キャッシュキー用）"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


# メイン実行部（テスト用）
if __name__ == "__main__":
    # 簡単なテスト
    test_invention = {
        "title": "液体分離設備の運転データ予測システム",
        "problem": "既存システムでは性能劣化の予測が困難",
        "solution": "機械学習による予測アルゴリズム",
        "effects": "運転効率の向上と保守コスト削減",
        "key_elements": ["センサーデータ", "予測モデル", "アラート機能"]
    }
    
    try:
        # 発明プロンプトテスト
        invention_prompt = create_invention_prompt(test_invention)
        print(f"Invention prompt created: {len(invention_prompt)} messages")
        print(f"Estimated tokens: {estimate_prompt_tokens(invention_prompt)}")
        
        # 特許プロンプトテスト
        test_patent = {
            "publication_number": "JP2025-100001A",
            "title": "膜分離装置の性能監視システム",
            "claim_1": "液体分離設備において運転データを収集する手段を備える装置。",
            "abstract": "本発明は液体分離設備の運転データを収集し分析する装置に関する。"
        }
        
        patent_prompt = create_patent_prompt(test_patent, "液体分離設備の予測システム")
        print(f"Patent prompt created: {len(patent_prompt)} messages")
        
        # 分類プロンプトテスト
        classification_prompt = create_classification_prompt(
            "液体分離設備の運転データ予測システム",
            "膜分離装置の性能監視システム"
        )
        print(f"Classification prompt created: {len(classification_prompt)} messages")
        
        print("All prompts created successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        logger.error(f"Test execution failed: {e}")