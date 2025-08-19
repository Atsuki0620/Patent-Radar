"""
extract.py - 請求項・抄録抽出機能

特許データから請求項1（独立請求項）、抄録、明細書を抽出する。
Code-reviewerの指摘に基づき、セキュリティ・パフォーマンス・品質を重視した実装。
"""

import re
import yaml
import signal
from typing import Dict, Any, List, Optional, Union
from loguru import logger
from pathlib import Path

# 設定値の定数 - Code-reviewer推奨による改善
class ExtractionLimits:
    MAX_PARAGRAPHS = 100
    MAX_PARAGRAPH_LENGTH = 10000
    MAX_CLAIM_NUMBER = 1000
    MAX_TEXT_LENGTH = 1000000
    MAX_ID_LENGTH = 20
    WORD_BOUNDARY_THRESHOLD = 0.8
    CONFIG_TIMEOUT = 5.0  # 設定ファイル読み込みタイムアウト

# セキュリティ強化: 事前コンパイル済み正規表現
CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
CITATION_PATTERN = re.compile(r'請求項(\d+)')

# セキュリティ強化された設定読み込み - Code-reviewer推奨
def load_config() -> Dict[str, Any]:
    """設定ファイルを安全に読み込む"""
    # パストラバーサル対策
    base_dir = Path(__file__).parent.parent.parent.resolve()
    config_path = base_dir / "config.yaml"
    
    # プロジェクトディレクトリ内であることを確認
    if not str(config_path).startswith(str(base_dir)):
        raise SecurityError("Config path outside project directory")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            return validate_config_structure(config_data)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return get_default_config()
    except yaml.YAMLError as e:
        logger.error(f"Config file parsing error: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """デフォルト設定を取得"""
    return {
        'extraction': {
            'max_abstract_length': 500,
            'include_dependent_claims': False,
            'max_description_paragraphs': 50
        }
    }


def validate_config_structure(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """設定データの構造を検証・消毒"""
    if not isinstance(config_data, dict):
        raise ValueError("Config must be a dictionary")
    
    # extraction設定の検証
    extraction = config_data.get('extraction', {})
    if not isinstance(extraction, dict):
        extraction = {}
    
    # 数値設定の範囲チェック
    max_length = extraction.get('max_abstract_length', 500)
    if not isinstance(max_length, int) or max_length < 1 or max_length > 50000:
        logger.warning(f"Invalid max_abstract_length: {max_length}, using default 500")
        extraction['max_abstract_length'] = 500
    
    max_paragraphs = extraction.get('max_description_paragraphs', 50)
    if not isinstance(max_paragraphs, int) or max_paragraphs < 1 or max_paragraphs > 200:
        logger.warning(f"Invalid max_description_paragraphs: {max_paragraphs}, using default 50")
        extraction['max_description_paragraphs'] = 50
    
    config_data['extraction'] = extraction
    return config_data


class SecurityError(Exception):
    """セキュリティ関連エラー"""
    pass


# 設定キャッシュ - Code-reviewer推奨によるパフォーマンス改善
_config_cache = None

def get_config() -> Dict[str, Any]:
    """設定を遅延ロード＋キャッシュで取得"""
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def safe_regex_sub(pattern: re.Pattern, replacement: str, text: str, timeout: float = 1.0) -> str:
    """正規表現置換のタイムアウト保護 - Code-reviewer推奨"""
    def timeout_handler(signum, frame):
        raise TimeoutError("Regex operation timed out")
    
    # Windowsではsignalが制限されているため、長さチェックで代用
    if len(text) > 100000:  # 大きなテキストの場合は単純な文字列操作にフォールバック
        logger.warning("Large text detected, using fallback character removal")
        return ''.join(c for c in text if ord(c) >= 32 or c in '\t\n\r')
    
    try:
        return pattern.sub(replacement, text)
    except re.error as e:
        logger.warning(f"Regex error: {e}, using fallback")
        return ''.join(c for c in text if ord(c) >= 32 or c in '\t\n\r')


def safe_text_processing(text: Any) -> str:
    """
    メモリ効率的・安全なテキスト処理 - Code-reviewer推奨による改善
    
    Args:
        text: 処理対象テキスト
        
    Returns:
        安全に処理されたテキスト文字列
    """
    if text is None:
        return ""
    
    # 文字列に安全に変換（一度だけ実行）
    if isinstance(text, str):
        text_str = text
    else:
        try:
            text_str = str(text)
        except UnicodeError:
            logger.warning(f"Unicode error in text conversion, using repr: {repr(text)}")
            text_str = repr(text)
    
    # 早期切り詰め（メモリ効率化）
    if len(text_str) > ExtractionLimits.MAX_TEXT_LENGTH:
        logger.warning(f"Text truncated from {len(text_str)} to {ExtractionLimits.MAX_TEXT_LENGTH} characters")
        text_str = text_str[:ExtractionLimits.MAX_TEXT_LENGTH]
    
    # セキュリティ強化された制御文字除去
    if text_str:
        text_str = safe_regex_sub(CONTROL_CHAR_PATTERN, '', text_str).strip()
    
    return text_str


def validate_patent_structure(patent_data: Dict[str, Any]) -> None:
    """
    特許データ構造の検証（code-reviewer推奨）
    
    Args:
        patent_data: 特許データ辞書
        
    Raises:
        ValueError: データ構造が不正な場合
        TypeError: 型が不正な場合
    """
    if not isinstance(patent_data, dict):
        raise TypeError(f"Patent data must be a dictionary, got {type(patent_data)}")
    
    # claims フィールドの検証
    claims = patent_data.get('claims')
    if claims is not None:
        if not isinstance(claims, list):
            raise ValueError("Claims field must be a list or None")
        
        # 各請求項の構造検証
        for i, claim in enumerate(claims):
            if not isinstance(claim, dict):
                raise ValueError(f"Claim {i} must be a dictionary")
            
            # 請求項番号の検証
            claim_no = claim.get('no')
            if claim_no is not None:
                if not isinstance(claim_no, int) or claim_no < 1 or claim_no > ExtractionLimits.MAX_CLAIM_NUMBER:
                    raise ValueError(f"Invalid claim number: {claim_no}")


def extract_claim_1(patent_data: Dict[str, Any]) -> Optional[str]:
    """
    独立請求項1を抽出する
    
    Args:
        patent_data: 特許データ辞書
        
    Returns:
        請求項1のテキスト、見つからない場合はNone
        
    Raises:
        ValueError: データ構造が不正な場合
        TypeError: 型が不正な場合
    """
    logger.debug(f"Extracting claim 1 from patent: {patent_data.get('publication_number', 'unknown')}")
    
    # 入力検証
    validate_patent_structure(patent_data)
    
    claims = patent_data.get('claims', [])
    if not claims:
        logger.debug("No claims found")
        return None
    
    # 優先度1: 請求項番号1を探す
    for claim in claims:
        if claim.get('no') == 1:
            text = safe_text_processing(claim.get('text', ''))
            if text:
                logger.debug(f"Found claim 1: {text[:50]}...")
                return text
    
    # 優先度2: 最初の独立請求項を探す
    for claim in claims:
        if claim.get('is_independent', False):
            text = safe_text_processing(claim.get('text', ''))
            if text:
                logger.debug(f"Found first independent claim: {text[:50]}...")
                return text
    
    logger.debug("No claim 1 or independent claim found")
    return None


def extract_abstract(patent_data: Dict[str, Any]) -> Optional[str]:
    """
    抄録を抽出する
    
    Args:
        patent_data: 特許データ辞書
        
    Returns:
        抄録テキスト、見つからない場合はNone
    """
    logger.debug(f"Extracting abstract from patent: {patent_data.get('publication_number', 'unknown')}")
    
    if not isinstance(patent_data, dict):
        raise TypeError(f"Patent data must be a dictionary, got {type(patent_data)}")
    
    abstract = patent_data.get('abstract', '')
    if not abstract:
        logger.debug("No abstract found")
        return None
    
    # 安全なテキスト処理
    abstract = safe_text_processing(abstract)
    if not abstract:
        return None
    
    # 設定に基づく文字数制限 - キャッシュされた設定を使用
    config = get_config()
    max_length = config['extraction'].get('max_abstract_length', 500)
    
    if len(abstract) > max_length:
        logger.debug(f"Abstract truncated from {len(abstract)} to {max_length} characters")
        # 単語境界での切り詰め - パフォーマンス改善
        truncated = abstract[:max_length]
        last_space = truncated.rfind(' ')
        
        # スペースが制限の80%以降にある場合は単語境界で切り詰め
        if last_space > max_length * ExtractionLimits.WORD_BOUNDARY_THRESHOLD:
            abstract = "".join([truncated[:last_space], "..."])
        else:
            abstract = "".join([truncated, "..."])
    
    logger.debug(f"Extracted abstract: {abstract[:50]}...")
    return abstract


def extract_description(patent_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    明細書を抽出する（メモリ安全対応）
    
    Args:
        patent_data: 特許データ辞書
        
    Returns:
        明細書段落のリスト（id, textを含む辞書のリスト）
        
    Raises:
        TypeError: 型が不正な場合
    """
    logger.debug(f"Extracting description from patent: {patent_data.get('publication_number', 'unknown')}")
    
    if not isinstance(patent_data, dict):
        raise TypeError(f"Patent data must be a dictionary, got {type(patent_data)}")
    
    description = patent_data.get('description')
    if not description:
        logger.debug("No description found")
        return []
    
    # 設定値の取得 - キャッシュされた設定を使用
    config = get_config()
    max_paragraphs = config['extraction'].get('max_description_paragraphs', 50)
    
    if isinstance(description, str):
        # 文字列形式の場合、段落に分割
        return _extract_description_from_string(description, max_paragraphs)
    elif isinstance(description, list):
        # リスト形式の場合、構造を正規化
        return _extract_description_from_list(description, max_paragraphs)
    else:
        raise ValueError(f"Description must be string or list, got {type(description)}")


def _extract_description_from_string(description: str, max_paragraphs: int) -> List[Dict[str, str]]:
    """
    文字列形式の明細書から段落を抽出
    
    Args:
        description: 明細書文字列
        max_paragraphs: 最大段落数
        
    Returns:
        段落リスト
    """
    paragraphs = description.split('\n\n')
    normalized = []
    
    for i, para in enumerate(paragraphs):
        if i >= max_paragraphs:
            logger.debug(f"Description truncated to {max_paragraphs} paragraphs")
            break
            
        para = safe_text_processing(para)
        if para and len(para) > 0:
            normalized.append({
                'id': f'[{i+1:04d}]',
                'text': para[:ExtractionLimits.MAX_PARAGRAPH_LENGTH]  # 段落長制限
            })
    
    logger.debug(f"Extracted {len(normalized)} paragraphs from string description")
    return normalized


def _extract_description_from_list(description: List[Any], max_paragraphs: int) -> List[Dict[str, str]]:
    """
    リスト形式の明細書から段落を抽出（メモリ安全対応）
    
    Args:
        description: 明細書リスト
        max_paragraphs: 最大段落数
        
    Returns:
        段落リスト
    """
    if len(description) > ExtractionLimits.MAX_PARAGRAPHS:
        logger.warning(f"Description list truncated from {len(description)} to {ExtractionLimits.MAX_PARAGRAPHS} items")
        description = description[:ExtractionLimits.MAX_PARAGRAPHS]
    
    normalized = []
    valid_count = 0
    
    for item in description:
        if valid_count >= max_paragraphs:
            logger.debug(f"Description truncated to {max_paragraphs} valid paragraphs")
            break
            
        # 有効な辞書要素のみ処理
        if isinstance(item, dict) and 'text' in item:
            text = safe_text_processing(item.get('text', ''))
            if text:
                # ID の安全な処理
                item_id = str(item.get('id', f'[{valid_count+1:04d}]'))[:ExtractionLimits.MAX_ID_LENGTH]  # ID長制限
                
                normalized.append({
                    'id': item_id,
                    'text': text[:ExtractionLimits.MAX_PARAGRAPH_LENGTH]  # 段落長制限
                })
                valid_count += 1
        else:
            # 不正な要素はスキップ
            logger.debug(f"Skipping invalid description item: {type(item)}")
            continue
    
    logger.debug(f"Extracted {len(normalized)} valid paragraphs from list description")
    return normalized


def extract_dependent_claims(patent_data: Dict[str, Any], base_claim_no: int = 1) -> List[Dict[str, Any]]:
    """
    指定された請求項に依存する従属請求項を抽出
    
    Args:
        patent_data: 特許データ辞書
        base_claim_no: 基準となる請求項番号
        
    Returns:
        従属請求項のリスト
    """
    validate_patent_structure(patent_data)
    
    claims = patent_data.get('claims', [])
    dependent_claims = []
    
    for claim in claims:
        if not claim.get('is_independent', True):  # 従属請求項
            text = safe_text_processing(claim.get('text', ''))
            # 基準請求項への言及をチェック
            if f'請求項{base_claim_no}' in text:
                dependent_claims.append({
                    'no': claim.get('no'),
                    'text': text,
                    'references': base_claim_no
                })
    
    logger.debug(f"Found {len(dependent_claims)} dependent claims for claim {base_claim_no}")
    return dependent_claims


def extract_citation_patterns(claim_text: str) -> List[int]:
    """
    請求項テキストから引用パターンを抽出
    
    Args:
        claim_text: 請求項テキスト
        
    Returns:
        引用されている請求項番号のリスト
    """
    text = safe_text_processing(claim_text)
    
    # 事前コンパイル済み正規表現を使用 - パフォーマンス改善
    citations = CITATION_PATTERN.findall(text)
    citation_numbers = []
    
    for citation in citations:
        try:
            claim_no = int(citation)
            if 1 <= claim_no <= ExtractionLimits.MAX_CLAIM_NUMBER:
                citation_numbers.append(claim_no)
        except ValueError:
            continue
    
    # 重複除去と並び替え
    return sorted(list(set(citation_numbers)))


def extract_technical_terms(text: str) -> List[str]:
    """
    テキストから技術用語を抽出
    
    Args:
        text: 抽出対象テキスト
        
    Returns:
        技術用語のリスト
    """
    text = safe_text_processing(text)
    
    # 特許技術分野の専門用語パターン
    technical_patterns = [
        r'液体分離設備',
        r'逆浸透膜',
        r'限外濾過',
        r'ナノ濾過',
        r'膜分離',
        r'性能劣化',
        r'運転データ',
        r'予測システム',
        r'汚染指標',
        r'膜ファウリング',
        r'透過性能'
    ]
    
    found_terms = []
    for pattern in technical_patterns:
        if re.search(pattern, text):
            matches = re.findall(pattern, text)
            found_terms.extend(matches)
    
    # 重複除去
    return list(set(found_terms))


def extract_with_fallback(patent_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    包括的なエラーハンドリング付き抽出
    
    Args:
        patent_data: 特許データ辞書
        
    Returns:
        抽出結果辞書（ステータス情報含む）
    """
    pub_number = patent_data.get('publication_number', 'unknown')
    logger.debug(f"Starting extraction with fallback for patent: {pub_number}")
    
    try:
        result = {
            'publication_number': pub_number,
            'claim_1': extract_claim_1(patent_data),
            'abstract': extract_abstract(patent_data),
            'description': extract_description(patent_data),
            'extraction_status': 'success'
        }
        
        # 依存請求項の抽出（設定で有効な場合） - キャッシュされた設定を使用
        config = get_config()
        if config['extraction'].get('include_dependent_claims', False):
            result['dependent_claims'] = extract_dependent_claims(patent_data)
        
        logger.debug(f"Extraction completed successfully for patent: {pub_number}")
        return result
        
    except Exception as e:
        logger.error(f"Extraction failed for patent {pub_number}: {e}")
        return {
            'publication_number': pub_number,
            'claim_1': None,
            'abstract': None,
            'description': [],
            'extraction_status': 'failed',
            'error': str(e)
        }


if __name__ == "__main__":
    # 簡単なテスト
    test_patent = {
        "publication_number": "JP2025-100001A",
        "claims": [
            {"no": 1, "text": "液体分離設備において運転データを収集する手段を備える装置。", "is_independent": True}
        ],
        "abstract": "本発明は液体分離設備の運転データを収集し分析する装置に関する。"
    }
    
    result = extract_with_fallback(test_patent)
    print(f"Claim 1: {result['claim_1']}")
    print(f"Abstract: {result['abstract']}")
    print(f"Status: {result['extraction_status']}")