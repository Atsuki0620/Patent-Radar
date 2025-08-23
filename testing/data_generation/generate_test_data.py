"""
50件のテストデータ生成スクリプト
多様な液体分離関連特許を生成
"""

import json
import random
from datetime import datetime, timedelta

def generate_patents():
    """50件の特許データを生成"""
    
    patents = []
    
    # カテゴリ定義
    categories = {
        'high_hit': {  # 明確なHIT候補（10件）
            'titles': [
                "AIによる膜分離プロセスの自動最適化システム",
                "新規ポリマー材料を用いた高効率分離膜",
                "機械学習による分離効率予測システム",
                "ナノ構造制御による超高性能分離膜",
                "IoTセンサと連携した膜性能リアルタイム監視装置",
                "エネルギー回収機能付き膜分離装置",
                "自己修復機能を有する新規分離膜材料",
                "量子コンピュータによる分離条件最適化",
                "バイオミメティック分離膜の製造方法",
                "ハイブリッド型AI制御分離システム"
            ],
            'keywords': ['AI', '新規ポリマー', '最適化', 'エネルギー削減', '効率向上']
        },
        'medium_hit': {  # 境界線上（15件）
            'titles': [
                "膜分離装置の洗浄方法",
                "分離プロセスのデータ収集システム",
                "複合膜の製造方法",
                "分離装置の圧力制御機構",
                "膜モジュールの配置最適化",
                "分離性能評価装置",
                "膜の劣化診断方法",
                "分離プロセスのシミュレーション方法",
                "膜分離と蒸留の組み合わせシステム",
                "分離装置の省スペース設計",
                "膜の再生処理方法",
                "分離効率モニタリング装置",
                "膜材料の改質方法",
                "分離装置の自動運転システム",
                "膜の目詰まり防止機構"
            ],
            'keywords': ['膜', '分離', '制御', 'モニタリング']
        },
        'no_hit': {  # 明確なMISS（25件）
            'titles': [
                "半導体製造装置",
                "電池の充放電制御システム",
                "画像認識による品質検査装置",
                "ロボットアームの制御方法",
                "無線通信システム",
                "データベース管理システム",
                "自動車の運転支援装置",
                "建築物の耐震構造",
                "食品の包装方法",
                "医療用診断装置",
                "農業用散水システム",
                "風力発電装置",
                "3Dプリンタの制御方法",
                "暗号化通信システム",
                "音声認識装置",
                "顔認証システム",
                "ドローンの飛行制御",
                "VRディスプレイ装置",
                "量子暗号通信",
                "ブロックチェーン管理システム",
                "遺伝子解析装置",
                "気象予測システム",
                "交通管制システム",
                "eコマースプラットフォーム",
                "SNS解析システム"
            ],
            'keywords': []
        }
    }
    
    # 出願人リスト
    assignees = [
        "技術開発株式会社", "環境ソリューション株式会社", "先端材料研究所",
        "グローバルテック株式会社", "イノベーション工業", "サステナブル技術開発",
        "ナノテクノロジー研究所", "AIシステム株式会社", "バイオ技術研究所",
        "エコロジー開発株式会社"
    ]
    
    # 日付生成用
    base_date = datetime(2024, 1, 1)
    
    pub_number = 100001
    
    # 高HIT候補生成（10件）
    for i, title in enumerate(categories['high_hit']['titles']):
        patent = {
            "publication_number": f"JP2025-{pub_number:06d}A",
            "title": title,
            "assignee": random.choice(assignees),
            "pub_date": (base_date + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
            "claims": [
                {
                    "no": 1,
                    "text": f"{title}において、新規ポリマー材料とAI制御を組み合わせ、エネルギー効率を最適化することを特徴とする装置。",
                    "is_independent": True
                },
                {
                    "no": 2,
                    "text": "請求項1において、リアルタイムモニタリング機能を備えることを特徴とする装置。",
                    "is_independent": False
                }
            ],
            "abstract": f"本発明は{title}に関し、新規材料の採用とAI技術により、従来比50%の効率向上と30%のエネルギー削減を実現する。",
            "cpc": ["B01D61/00", "G05B13/02"],
            "expected_hit": True,
            "category": "high_hit"
        }
        patents.append(patent)
        pub_number += 1
    
    # 境界線上生成（15件）
    for i, title in enumerate(categories['medium_hit']['titles']):
        patent = {
            "publication_number": f"JP2025-{pub_number:06d}A",
            "title": title,
            "assignee": random.choice(assignees),
            "pub_date": (base_date + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
            "claims": [
                {
                    "no": 1,
                    "text": f"{title}に関する技術であって、膜分離プロセスの改良を特徴とする。",
                    "is_independent": True
                }
            ],
            "abstract": f"本発明は{title}に関し、既存技術の改良により性能向上を図る。",
            "cpc": ["B01D63/00"],
            "expected_hit": False,  # 境界線上だが基本的にMISS想定
            "category": "medium_hit"
        }
        patents.append(patent)
        pub_number += 1
    
    # 明確なMISS生成（25件）
    for i, title in enumerate(categories['no_hit']['titles']):
        patent = {
            "publication_number": f"JP2025-{pub_number:06d}A",
            "title": title,
            "assignee": random.choice(assignees),
            "pub_date": (base_date + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
            "claims": [
                {
                    "no": 1,
                    "text": f"{title}に関する技術。",
                    "is_independent": True
                }
            ],
            "abstract": f"本発明は{title}の分野における新技術を提供する。",
            "cpc": ["G06F1/00"],
            "expected_hit": False,
            "category": "no_hit"
        }
        patents.append(patent)
        pub_number += 1
    
    # ランダムに並び替え（実際のデータセットをシミュレート）
    random.shuffle(patents)
    
    return patents

def main():
    """メイン処理"""
    print("50件のテストデータを生成中...")
    
    patents = generate_patents()
    
    # JSONL形式で保存
    output_path = "test_data/production_patents_50.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for patent in patents:
            # expected_hitとcategoryは出力から除外（内部検証用）
            output_patent = {k: v for k, v in patent.items() 
                           if k not in ['expected_hit', 'category']}
            f.write(json.dumps(output_patent, ensure_ascii=False) + '\n')
    
    print(f"テストデータを生成しました: {output_path}")
    
    # 統計情報
    high_hit = sum(1 for p in patents if p.get('category') == 'high_hit')
    medium_hit = sum(1 for p in patents if p.get('category') == 'medium_hit')
    no_hit = sum(1 for p in patents if p.get('category') == 'no_hit')
    
    print(f"\nデータ構成:")
    print(f"  高HIT候補: {high_hit}件")
    print(f"  境界線上: {medium_hit}件")
    print(f"  明確なMISS: {no_hit}件")
    print(f"  合計: {len(patents)}件")

if __name__ == "__main__":
    main()