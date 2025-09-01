#!/usr/bin/env python3
"""
汎用バイナリ分類器評価システム設定
Binary Classifier Evaluation System Configuration
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """モデル設定"""
    name: str
    description: str
    task_type: str = "binary_classification"  # "binary_classification", "multi_classification", etc.
    positive_class_name: str = "HIT"  # 正例クラス名（例：HIT, SPAM, FRAUD, etc.）
    negative_class_name: str = "MISS"  # 負例クラス名（例：MISS, HAM, NORMAL, etc.）
    prediction_threshold: float = 0.5  # デフォルト予測閾値
    performance_target: float = 0.8  # 目標性能（F1スコア）


@dataclass 
class BusinessConfig:
    """ビジネス設定"""
    false_negative_cost: float = 2000000  # 見逃しコスト（円）
    false_positive_cost: float = 20000   # 誤検出1件あたりの工数コスト（円）
    currency: str = "JPY"  # 通貨単位
    cost_unit_name: str = "円"  # コスト単位名


@dataclass
class VisualizationConfig:
    """可視化設定"""
    dpi: int = 300
    figsize_width: int = 10
    figsize_height: int = 8
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: str = "husl"
    font_size_title: int = 14
    font_size_label: int = 12


@dataclass
class ReportConfig:
    """レポート設定"""
    title_template: str = "{model_name} 評価レポート"
    language: str = "ja"  # "ja", "en"
    include_business_metrics: bool = True
    include_technical_details: bool = True
    graph_format: str = "png"  # "png", "svg", "jpg"


@dataclass
class EvaluationConfig:
    """全体評価設定"""
    model: ModelConfig
    business: BusinessConfig
    visualization: VisualizationConfig
    report: ReportConfig
    threshold_resolution: int = 1000  # 閾値最適化の解像度
    random_seed: int = 42
    
    
class ConfigManager:
    """設定管理クラス"""
    
    DEFAULT_CONFIGS = {
        "patent_radar": {
            "model": {
                "name": "PatentRadar",
                "description": "特許衝突検出システム",
                "positive_class_name": "HIT",
                "negative_class_name": "MISS",
                "prediction_threshold": 0.5,
                "performance_target": 0.8
            },
            "business": {
                "false_negative_cost": 2000000,
                "false_positive_cost": 20000,
                "currency": "JPY", 
                "cost_unit_name": "円"
            },
            "visualization": {
                "dpi": 300,
                "figsize_width": 10,
                "figsize_height": 8,
                "style": "seaborn-v0_8-whitegrid",
                "color_palette": "husl",
                "font_size_title": 14,
                "font_size_label": 12
            },
            "report": {
                "title_template": "{model_name} 評価レポート",
                "language": "ja",
                "include_business_metrics": True,
                "include_technical_details": True,
                "graph_format": "png"
            }
        },
        
        "spam_filter": {
            "model": {
                "name": "SpamFilter", 
                "description": "スパムメール検出システム",
                "positive_class_name": "SPAM",
                "negative_class_name": "HAM", 
                "prediction_threshold": 0.5,
                "performance_target": 0.9
            },
            "business": {
                "false_negative_cost": 10000,  # スパム見逃しコスト
                "false_positive_cost": 50000,  # 重要メール誤検出コスト
                "currency": "JPY",
                "cost_unit_name": "円"
            },
            "visualization": {
                "dpi": 300,
                "figsize_width": 10,
                "figsize_height": 8,
                "style": "seaborn-v0_8-whitegrid", 
                "color_palette": "husl",
                "font_size_title": 14,
                "font_size_label": 12
            },
            "report": {
                "title_template": "{model_name} Performance Evaluation Report",
                "language": "en",
                "include_business_metrics": True,
                "include_technical_details": True,
                "graph_format": "png"
            }
        },
        
        "fraud_detection": {
            "model": {
                "name": "FraudDetector",
                "description": "詐欺検出システム", 
                "positive_class_name": "FRAUD",
                "negative_class_name": "NORMAL",
                "prediction_threshold": 0.3,  # 詐欺検出は保守的な閾値
                "performance_target": 0.85
            },
            "business": {
                "false_negative_cost": 5000000,  # 詐欺見逃し大損失
                "false_positive_cost": 3000,     # 誤検出調査コスト
                "currency": "JPY",
                "cost_unit_name": "円" 
            },
            "visualization": {
                "dpi": 300,
                "figsize_width": 12,
                "figsize_height": 8,
                "style": "seaborn-v0_8-whitegrid",
                "color_palette": "husl", 
                "font_size_title": 16,
                "font_size_label": 12
            },
            "report": {
                "title_template": "{model_name} 詐欺検出性能レポート", 
                "language": "ja",
                "include_business_metrics": True,
                "include_technical_details": True,
                "graph_format": "png"
            }
        }
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        設定管理初期化
        
        Args:
            config_path: カスタム設定ファイルパス
        """
        self.config_path = Path(config_path) if config_path else None
        self.custom_configs = {}
        
        if self.config_path and self.config_path.exists():
            self.load_config_file()
    
    def load_config_file(self):
        """設定ファイルを読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.custom_configs = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config file {self.config_path}: {e}")
    
    def save_config_file(self, config_name: str, config_dict: Dict):
        """設定をファイルに保存"""
        if not self.config_path:
            self.config_path = Path("analysis/evaluation_system/custom_configs.json")
        
        self.custom_configs[config_name] = config_dict
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.custom_configs, f, ensure_ascii=False, indent=2)
    
    def get_config(self, config_name: str) -> EvaluationConfig:
        """
        設定を取得
        
        Args:
            config_name: 設定名（"patent_radar", "spam_filter", "fraud_detection", etc.）
            
        Returns:
            EvaluationConfig: 評価設定オブジェクト
        """
        # カスタム設定を優先
        config_dict = self.custom_configs.get(config_name) or self.DEFAULT_CONFIGS.get(config_name)
        
        if not config_dict:
            # デフォルト設定がない場合は汎用設定を生成
            config_dict = self._create_generic_config(config_name)
        
        return self._dict_to_config(config_dict)
    
    def _create_generic_config(self, model_name: str) -> Dict:
        """汎用設定を生成"""
        return {
            "model": {
                "name": model_name,
                "description": f"{model_name} Binary Classification System",
                "positive_class_name": "POSITIVE",
                "negative_class_name": "NEGATIVE",
                "prediction_threshold": 0.5,
                "performance_target": 0.8
            },
            "business": {
                "false_negative_cost": 100000,
                "false_positive_cost": 10000,
                "currency": "JPY",
                "cost_unit_name": "円"
            },
            "visualization": {
                "dpi": 300,
                "figsize_width": 10,
                "figsize_height": 8,
                "style": "seaborn-v0_8-whitegrid",
                "color_palette": "husl",
                "font_size_title": 14,
                "font_size_label": 12
            },
            "report": {
                "title_template": "{model_name} Evaluation Report",
                "language": "en",
                "include_business_metrics": True,
                "include_technical_details": True,
                "graph_format": "png"
            }
        }
    
    def _dict_to_config(self, config_dict: Dict) -> EvaluationConfig:
        """辞書から設定オブジェクトに変換"""
        return EvaluationConfig(
            model=ModelConfig(**config_dict["model"]),
            business=BusinessConfig(**config_dict["business"]), 
            visualization=VisualizationConfig(**config_dict["visualization"]),
            report=ReportConfig(**config_dict["report"]),
            threshold_resolution=config_dict.get("threshold_resolution", 1000),
            random_seed=config_dict.get("random_seed", 42)
        )
    
    def list_available_configs(self) -> List[str]:
        """利用可能な設定名一覧を取得"""
        configs = set(self.DEFAULT_CONFIGS.keys())
        configs.update(self.custom_configs.keys())
        return sorted(list(configs))
    
    def create_custom_config(self, 
                           config_name: str,
                           model_name: str,
                           description: str,
                           positive_class_name: str = "POSITIVE",
                           negative_class_name: str = "NEGATIVE",
                           **kwargs) -> EvaluationConfig:
        """
        カスタム設定を作成
        
        Args:
            config_name: 設定名
            model_name: モデル名
            description: モデル説明
            positive_class_name: 正例クラス名
            negative_class_name: 負例クラス名
            **kwargs: その他の設定項目
            
        Returns:
            EvaluationConfig: 作成された設定
        """
        config_dict = self._create_generic_config(model_name)
        
        # モデル設定をカスタマイズ
        config_dict["model"]["name"] = model_name
        config_dict["model"]["description"] = description
        config_dict["model"]["positive_class_name"] = positive_class_name
        config_dict["model"]["negative_class_name"] = negative_class_name
        
        # その他のカスタム設定を適用
        for key, value in kwargs.items():
            if "." in key:
                # ネストされた設定（例：business.false_negative_cost）
                section, param = key.split(".", 1)
                if section in config_dict:
                    config_dict[section][param] = value
            else:
                # トップレベル設定
                config_dict[key] = value
        
        # ファイルに保存
        self.save_config_file(config_name, config_dict)
        
        return self._dict_to_config(config_dict)


# 使用例とヘルパー関数
def get_evaluation_config(config_name: str = "patent_radar", 
                         config_path: Optional[str] = None) -> EvaluationConfig:
    """
    評価設定を取得するヘルパー関数
    
    Args:
        config_name: 設定名
        config_path: カスタム設定ファイルパス
        
    Returns:
        EvaluationConfig: 評価設定
    """
    manager = ConfigManager(config_path)
    return manager.get_config(config_name)


def create_custom_evaluation_config(
    config_name: str,
    model_name: str, 
    description: str,
    positive_class_name: str = "POSITIVE",
    negative_class_name: str = "NEGATIVE",
    config_path: Optional[str] = None,
    **kwargs
) -> EvaluationConfig:
    """
    カスタム評価設定を作成するヘルパー関数
    
    Args:
        config_name: 設定名
        model_name: モデル名
        description: モデル説明
        positive_class_name: 正例クラス名
        negative_class_name: 負例クラス名
        config_path: カスタム設定ファイルパス
        **kwargs: その他の設定項目
        
    Returns:
        EvaluationConfig: 作成された設定
    """
    manager = ConfigManager(config_path)
    return manager.create_custom_config(
        config_name, model_name, description,
        positive_class_name, negative_class_name,
        **kwargs
    )


if __name__ == "__main__":
    # 設定システムのテスト
    print("=== 汎用バイナリ分類器評価システム設定テスト ===")
    
    # 設定管理初期化
    manager = ConfigManager()
    
    # 利用可能な設定一覧
    print(f"利用可能な設定: {manager.list_available_configs()}")
    
    # Patent-Radar設定を取得
    patent_config = manager.get_config("patent_radar")
    print(f"Patent-Radar設定: {patent_config.model.name}")
    print(f"  - 正例クラス: {patent_config.model.positive_class_name}")
    print(f"  - 負例クラス: {patent_config.model.negative_class_name}")
    print(f"  - 見逃しコスト: {patent_config.business.false_negative_cost:,}{patent_config.business.cost_unit_name}")
    
    # カスタム設定を作成
    custom_config = manager.create_custom_config(
        "custom_classifier",
        "MyClassifier", 
        "カスタム分類器テスト",
        positive_class_name="GOOD",
        negative_class_name="BAD",
        **{
            "business.false_negative_cost": 500000,
            "business.false_positive_cost": 5000,
            "model.performance_target": 0.85
        }
    )
    
    print(f"カスタム設定作成: {custom_config.model.name}")
    print(f"  - 正例クラス: {custom_config.model.positive_class_name}")
    print(f"  - 負例クラス: {custom_config.model.negative_class_name}")
    print(f"  - 目標性能: {custom_config.model.performance_target}")