"""
normalize.py - JSONデータ正規化機能

特許JSONデータと発明アイデアの正規化処理を行う。
全角・半角、大小文字、単位、空白文字などの統一を図る。
"""

import re
import json
from typing import Dict, Any, Union, List, Optional
from datetime import datetime
from loguru import logger


def normalize_patent_json(patent_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    特許JSONデータを正規化する
    
    Args:
        patent_data: 特許データ辞書
        
    Returns:
        正規化された特許データ辞書
        
    Raises:
        ValueError: 必須フィールドが欠落している場合
        ValueError: 日付形式が無効な場合
    """
    logger.debug(f"Normalizing patent: {patent_data.get('publication_number', 'unknown')}")
    
    # 必須フィールドの確認
    required_fields = ['publication_number', 'title', 'assignee', 'pub_date', 'claims', 'abstract']
    missing_fields = [field for field in required_fields if field not in patent_data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # 正規化されたデータを作成
    normalized = {}
    
    # 文字列フィールドの正規化
    string_fields = ['publication_number', 'title', 'assignee', 'abstract']
    for field in string_fields:
        if field in patent_data:
            normalized[field] = clean_text(patent_data[field])
    
    # 日付の正規化
    normalized['pub_date'] = normalize_date(patent_data['pub_date'])
    
    # 請求項の正規化
    normalized['claims'] = normalize_claims(patent_data['claims'])
    
    # 任意フィールドの処理
    optional_fields = ['description', 'ipc', 'cpc', 'legal_status', 'citations_forward', 'url_hint']
    for field in optional_fields:
        if field in patent_data:
            if field in ['ipc', 'cpc']:
                # 分類コードはリストとして正規化
                normalized[field] = normalize_classification_codes(patent_data[field])
            elif field == 'citations_forward':
                # 被引用数は整数として正規化
                normalized[field] = normalize_integer(patent_data[field])
            elif field == 'description':
                # 説明文は段落のリストとして正規化
                normalized[field] = normalize_description(patent_data[field])
            else:
                # その他の文字列フィールド
                normalized[field] = clean_text(patent_data[field])
    
    logger.debug(f"Patent normalized: {normalized['publication_number']}")
    return normalized


def normalize_invention_idea(idea_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    発明アイデアを正規化する
    
    Args:
        idea_data: 発明アイデア（文字列またはJSON）
        
    Returns:
        正規化された発明アイデア辞書
    """
    logger.debug("Normalizing invention idea")
    
    if isinstance(idea_data, str):
        # テキスト形式の場合、基本的な構造化を試行
        normalized = {
            'title': extract_title_from_text(idea_data),
            'content': clean_text(idea_data)
        }
    elif isinstance(idea_data, dict):
        # JSON形式の場合、各フィールドを正規化
        normalized = {}
        
        # 標準フィールドの正規化
        standard_fields = ['title', 'problem', 'solution', 'effects', 'constraints']
        for field in standard_fields:
            if field in idea_data:
                normalized[field] = clean_text(idea_data[field])
        
        # リスト形式のフィールド
        if 'key_elements' in idea_data:
            normalized['key_elements'] = [
                clean_text(element) for element in idea_data['key_elements']
                if isinstance(element, str)
            ]
    else:
        raise ValueError(f"Unsupported idea_data type: {type(idea_data)}")
    
    logger.debug("Invention idea normalized")
    return normalized


def clean_text(text: str) -> str:
    """
    テキストの基本的な正規化を行う
    
    Args:
        text: 正規化対象テキスト
        
    Returns:
        正規化されたテキスト
    """
    if not isinstance(text, str):
        return str(text)
    
    # 空白文字の正規化
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 全角英数字を半角に変換
    text = normalize_alphanumeric(text)
    
    return text


def normalize_alphanumeric(text: str) -> str:
    """
    全角英数字を半角に変換する
    
    Args:
        text: 変換対象テキスト
        
    Returns:
        半角に変換されたテキスト
    """
    # 全角英数字の変換テーブル
    full_to_half = str.maketrans(
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
        'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        '０１２３４５６７８９',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'abcdefghijklmnopqrstuvwxyz'
        '0123456789'
    )
    
    return text.translate(full_to_half)


def normalize_date(date_str: str) -> str:
    """
    日付文字列を統一形式（YYYY-MM-DD）に正規化する
    
    Args:
        date_str: 日付文字列
        
    Returns:
        正規化された日付文字列（YYYY-MM-DD）
        
    Raises:
        ValueError: 日付形式が無効な場合
    """
    if not isinstance(date_str, str):
        date_str = str(date_str)
    
    # 既に正しい形式の場合
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            pass
    
    # その他の形式を試行
    date_formats = [
        '%Y/%m/%d',
        '%Y.%m.%d',
        '%Y年%m月%d日',
        '%m/%d/%Y',
        '%d/%m/%Y'
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    raise ValueError(f"Invalid date format: {date_str}")


def normalize_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    請求項リストを正規化する
    
    Args:
        claims: 請求項のリスト
        
    Returns:
        正規化された請求項リスト
    """
    if not isinstance(claims, list):
        raise ValueError("Claims must be a list")
    
    normalized_claims = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
            
        normalized_claim = {}
        
        # 請求項番号
        if 'no' in claim:
            normalized_claim['no'] = normalize_integer(claim['no'])
        
        # 請求項テキスト
        if 'text' in claim:
            normalized_claim['text'] = clean_text(claim['text'])
        
        # 独立/従属フラグ
        if 'is_independent' in claim:
            normalized_claim['is_independent'] = bool(claim['is_independent'])
        
        if normalized_claim:  # 空でない場合のみ追加
            normalized_claims.append(normalized_claim)
    
    return normalized_claims


def normalize_classification_codes(codes: Union[str, List[str]]) -> List[str]:
    """
    分類コード（IPC/CPC）を正規化する
    
    Args:
        codes: 分類コード（文字列またはリスト）
        
    Returns:
        正規化された分類コードのリスト
    """
    if isinstance(codes, str):
        # カンマ区切りの文字列をリストに変換
        codes = [code.strip() for code in codes.split(',')]
    elif not isinstance(codes, list):
        codes = [str(codes)]
    
    # 各コードを正規化
    normalized = []
    for code in codes:
        if isinstance(code, str) and code.strip():
            normalized.append(code.strip().upper())
    
    return normalized


def normalize_description(description: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """
    明細書を正規化する
    
    Args:
        description: 明細書（文字列またはid付き段落のリスト）
        
    Returns:
        正規化された明細書段落のリスト
    """
    if isinstance(description, str):
        # 文字列の場合、段落に分割
        paragraphs = description.split('\n\n')
        normalized = []
        for i, para in enumerate(paragraphs):
            para = clean_text(para)
            if para:
                normalized.append({
                    'id': f'[{i+1:04d}]',
                    'text': para
                })
        return normalized
    
    elif isinstance(description, list):
        # リストの場合、各要素を正規化
        normalized = []
        for item in description:
            if isinstance(item, dict) and 'text' in item:
                normalized_item = {
                    'id': item.get('id', f'[{len(normalized)+1:04d}]'),
                    'text': clean_text(item['text'])
                }
                if normalized_item['text']:
                    normalized.append(normalized_item)
        return normalized
    
    else:
        raise ValueError(f"Unsupported description type: {type(description)}")


def normalize_integer(value: Any) -> int:
    """
    整数値を正規化する
    
    Args:
        value: 整数に変換する値
        
    Returns:
        正規化された整数
        
    Raises:
        ValueError: 整数に変換できない場合
    """
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            # カンマ区切りの数字を処理
            cleaned = value.replace(',', '').strip()
            return int(cleaned)
    elif isinstance(value, float):
        return int(value)
    else:
        raise ValueError(f"Cannot convert to integer: {value}")


def extract_title_from_text(text: str) -> str:
    """
    テキストからタイトルを抽出する
    
    Args:
        text: 抽出対象テキスト
        
    Returns:
        抽出されたタイトル
    """
    # 最初の行または最初の文をタイトルとして扱う
    lines = text.strip().split('\n')
    if lines:
        first_line = clean_text(lines[0])
        # 長すぎる場合は制限
        if len(first_line) > 100:
            first_line = first_line[:100] + '...'
        return first_line
    
    return "無題"


if __name__ == "__main__":
    # 簡単なテスト
    test_patent = {
        "publication_number": "JP2025-100001A",
        "title": "液体分離設備の予測保全システム",
        "assignee": "アクアテック株式会社",
        "pub_date": "2025-01-15",
        "claims": [
            {"no": 1, "text": "液体分離設備において運転データを収集する", "is_independent": True}
        ],
        "abstract": "本発明は液体分離設備の運転データを分析する"
    }
    
    result = normalize_patent_json(test_patent)
    print(json.dumps(result, indent=2, ensure_ascii=False))