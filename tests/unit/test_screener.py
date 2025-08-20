"""
Tests for screener module - 特許スクリーニング統合機能のテスト

特許分析システム全体の統合テスト。全コンポーネント（正規化、抽出、
LLM処理、ソート、出力）の統合実行をテストする。

Phase A: Code-reviewerによるテスト設計レビューのためのテストケース。
"""

import pytest
import tempfile
import json
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import yaml
import time


class TestPatentScreener:
    """特許スクリーニング統合機能の基本テスト"""

    def test_screener_initialization(self):
        """スクリーナー初期化のテスト"""
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        
        # 必要なコンポーネントが初期化されることを確認
        assert hasattr(screener, '_config')
        assert hasattr(screener, '_extract_function')
        assert hasattr(screener, '_pipeline')
        assert hasattr(screener, '_sorter')
        assert hasattr(screener, '_export_csv_function')
        assert hasattr(screener, '_export_jsonl_function')
        assert hasattr(screener, '_stats')

    def test_screener_with_custom_config(self):
        """カスタム設定でのスクリーナー初期化テスト"""
        from src.core.screener import PatentScreener
        
        custom_config = {
            'llm': {
                'model': 'gpt-4o-mini',
                'temperature': 0.0,
                'max_tokens': 320
            },
            'processing': {
                'batch_size': 5,
                'max_workers': 3
            },
            'ranking': {
                'method': 'llm_only',
                'tiebreaker': 'pub_number'
            }
        }
        
        screener = PatentScreener(config=custom_config)
        
        # カスタム設定が各コンポーネントに適用されることを確認
        assert screener._config['llm']['model'] == 'gpt-4o-mini'
        assert screener._config['processing']['batch_size'] == 5
        assert screener._config['ranking']['method'] == 'llm_only'

    def test_screener_config_validation(self):
        """スクリーナー設定検証のテスト"""
        from src.core.screener import PatentScreener
        
        # 無効な設定でのエラーハンドリング
        invalid_configs = [
            {'llm': {'model': 'invalid_model'}},  # 無効なモデル
            {'processing': {'batch_size': -1}},   # 無効なバッチサイズ
            {'ranking': {'method': 'invalid'}},   # 無効なランキング方法
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, Exception)):
                PatentScreener(config=invalid_config)

    def test_screener_statistics_initialization(self):
        """スクリーナー統計情報の初期化テスト"""
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        stats = screener.get_processing_stats()
        
        # 統計情報の構造確認
        expected_keys = [
            'total_patents_processed', 'successful_extractions', 'successful_classifications',
            'processing_time', 'average_time_per_patent', 'extraction_errors', 'llm_errors',
            'sorting_time', 'export_time', 'total_hits', 'total_misses'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))


class TestFullPipelineIntegration:
    """完全パイプライン統合テスト"""

    def test_end_to_end_processing_small_dataset(self):
        """小規模データセットでのエンドツーエンド処理テスト"""
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        
        # テスト用の発明アイデア
        invention_idea = {
            "title": "膜分離装置の性能向上技術",
            "problem": "従来の膜分離装置では分離効率が低く、エネルギー消費が大きい",
            "solution": "新しい膜材料と流路設計により分離性能を向上させる",
            "effects": "分離効率50%向上、エネルギー消費30%削減",
            "key_elements": ["新規膜材料", "最適化流路", "圧力制御システム"]
        }
        
        # テスト用の特許データ（最小構成）
        patents_data = [
            {
                "publication_number": "JP2025-100001A",
                "title": "高効率膜分離装置",
                "assignee": "技術株式会社",
                "pub_date": "2025-01-15",
                "abstract": "新しい膜材料を用いることで分離効率を大幅に向上させる膜分離装置。",
                "claims": [
                    {
                        "no": 1,
                        "text": "新規ポリマー膜を用いた液体分離装置であって、流路が最適化されていることを特徴とする装置。",
                        "is_independent": True
                    }
                ],
                "url_hint": "https://example.com/patent1"
            },
            {
                "publication_number": "JP2025-100002A",
                "title": "水処理システム",
                "assignee": "水処理工業株式会社", 
                "pub_date": "2025-02-01",
                "abstract": "従来のろ過技術を改良した水処理システム。",
                "claims": [
                    {
                        "no": 1,
                        "text": "砂ろ過と活性炭を組み合わせた水処理装置。",
                        "is_independent": True
                    }
                ],
                "url_hint": "https://example.com/patent2"
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 入力ファイル作成
            invention_file = Path(temp_dir) / "invention.json"
            patents_file = Path(temp_dir) / "patents.jsonl"
            
            with open(invention_file, 'w', encoding='utf-8') as f:
                json.dump(invention_idea, f, ensure_ascii=False, indent=2)
            
            with open(patents_file, 'w', encoding='utf-8') as f:
                for patent in patents_data:
                    f.write(json.dumps(patent, ensure_ascii=False) + '\n')
            
            # 出力ファイルパス
            output_csv = Path(temp_dir) / "results.csv"
            output_jsonl = Path(temp_dir) / "details.jsonl"
            
            # LLMクライアントのモック
            with patch('src.core.screener.LLMClient') as mock_client_class:
                # モックの設定
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                # LLMレスポンスのモック
                def mock_completion(messages, **kwargs):
                    content = messages[-1]['content']
                    if "発明アイデアを要約" in content:
                        return {
                            "summary": "膜分離装置の性能向上技術に関する発明",
                            "key_points": ["新規膜材料", "流路最適化", "エネルギー効率"]
                        }
                    elif "特許を要約" in content:
                        if "JP2025-100001A" in content:
                            return {
                                "summary": "高効率膜分離装置：新規ポリマー膜と最適化流路",
                                "key_points": ["ポリマー膜", "流路設計", "分離効率"]
                            }
                        else:
                            return {
                                "summary": "水処理システム：砂ろ過と活性炭の組み合わせ",
                                "key_points": ["砂ろ過", "活性炭", "水処理"]
                            }
                    else:  # 分類リクエスト
                        if "JP2025-100001A" in content:
                            return {
                                "decision": "hit",
                                "confidence": 0.85,
                                "reasoning": [
                                    {"reason": "膜材料技術の共通性", "source": "claim", "quote": "新規ポリマー膜"},
                                    {"reason": "分離効率向上の目的一致", "source": "abstract", "quote": "分離効率を大幅に向上"}
                                ]
                            }
                        else:
                            return {
                                "decision": "miss",
                                "confidence": 0.25,
                                "reasoning": [
                                    {"reason": "技術分野の相違", "source": "claim", "quote": "砂ろ過"}
                                ]
                            }
                
                mock_client.completion.side_effect = mock_completion
                mock_client.get_usage_stats.return_value = {
                    'total_requests': 5,
                    'total_tokens': 1500,
                    'failed_requests': 0
                }
                
                # エンドツーエンド処理実行
                results = screener.analyze(
                    invention=str(invention_file),
                    patents=str(patents_file),
                    output_csv=str(output_csv),
                    output_jsonl=str(output_jsonl)
                )
                
                # 結果の検証
                assert isinstance(results, dict)
                assert 'summary' in results
                assert 'patents_processed' in results
                assert 'total_hits' in results
                assert 'total_misses' in results
                assert 'processing_time' in results
                
                # 処理結果の確認
                assert results['patents_processed'] == 2
                assert results['total_hits'] == 1
                assert results['total_misses'] == 1
                
                # 出力ファイルの確認
                assert output_csv.exists()
                assert output_jsonl.exists()
                
                # CSV内容の確認
                with open(output_csv, 'r', encoding='utf-8') as f:
                    csv_reader = csv.DictReader(f)
                    rows = list(csv_reader)
                    
                assert len(rows) == 2
                assert rows[0]['decision'] == 'hit'  # 最高信頼度が最初
                assert rows[0]['LLM_confidence'] == '0.85'
                assert rows[1]['decision'] == 'miss'
                assert rows[1]['LLM_confidence'] == '0.25'
                
                # JSONL内容の確認
                with open(output_jsonl, 'r', encoding='utf-8') as f:
                    jsonl_data = [json.loads(line) for line in f]
                    
                assert len(jsonl_data) == 2
                hit_patent = next(p for p in jsonl_data if p['decision'] == 'hit')
                assert hit_patent['rank'] == 1
                assert hit_patent['confidence'] == 0.85

    def test_error_handling_during_processing(self):
        """処理中のエラーハンドリングテスト"""
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        
        # 不正な入力ファイルでのエラーハンドリング
        with tempfile.TemporaryDirectory() as temp_dir:
            # 存在しないファイル
            with pytest.raises((FileNotFoundError, Exception)):
                screener.analyze(
                    invention="nonexistent.json",
                    patents="nonexistent.jsonl"
                )
            
            # 不正なJSON形式ファイル
            invalid_file = Path(temp_dir) / "invalid.json"
            with open(invalid_file, 'w') as f:
                f.write("invalid json content")
            
            with pytest.raises((json.JSONDecodeError, ValueError, Exception)):
                screener.analyze(
                    invention=str(invalid_file),
                    patents=str(invalid_file)
                )

    def test_partial_processing_with_errors(self):
        """一部エラーありでの部分処理テスト"""
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        
        # 正常な発明アイデア
        invention_idea = {
            "title": "テスト発明",
            "problem": "テスト問題",
            "solution": "テスト解決策"
        }
        
        # 一部不正なデータを含む特許リスト
        mixed_patents_data = [
            # 正常なデータ
            {
                "publication_number": "JP2025-100001A",
                "title": "正常特許",
                "assignee": "正常会社",
                "pub_date": "2025-01-15",
                "abstract": "正常な抄録",
                "claims": [{"no": 1, "text": "正常なクレーム", "is_independent": True}]
            },
            # 不正なデータ（フィールド欠損）
            {
                "publication_number": "JP2025-100002A",
                "title": "不正特許",
                # assignee, pub_date, abstract, claims が欠損
            },
            # 正常なデータ
            {
                "publication_number": "JP2025-100003A",
                "title": "正常特許2",
                "assignee": "正常会社2",
                "pub_date": "2025-01-16",
                "abstract": "正常な抄録2",
                "claims": [{"no": 1, "text": "正常なクレーム2", "is_independent": True}]
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            invention_file = Path(temp_dir) / "invention.json"
            patents_file = Path(temp_dir) / "patents.jsonl"
            output_csv = Path(temp_dir) / "results.csv"
            output_jsonl = Path(temp_dir) / "details.jsonl"
            
            with open(invention_file, 'w', encoding='utf-8') as f:
                json.dump(invention_idea, f, ensure_ascii=False)
            
            with open(patents_file, 'w', encoding='utf-8') as f:
                for patent in mixed_patents_data:
                    f.write(json.dumps(patent, ensure_ascii=False) + '\n')
            
            with patch('src.core.screener.LLMClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                # 成功レスポンス
                def mock_completion(messages, **kwargs):
                    if "発明アイデアを要約" in messages[-1]['content']:
                        return {"summary": "テスト発明の要約"}
                    elif "特許を要約" in messages[-1]['content']:
                        return {"summary": "特許の要約"}
                    else:
                        return {
                            "decision": "hit",
                            "confidence": 0.75,
                            "reasoning": [{"reason": "テスト理由", "source": "claim"}]
                        }
                
                mock_client.completion.side_effect = mock_completion
                mock_client.get_usage_stats.return_value = {'total_requests': 3, 'failed_requests': 0}
                
                # 部分処理の実行（エラーがあっても続行）
                results = screener.analyze(
                    invention=str(invention_file),
                    patents=str(patents_file),
                    output_csv=str(output_csv),
                    output_jsonl=str(output_jsonl),
                    continue_on_error=True
                )
                
                # 正常に処理された特許のみが結果に含まれることを確認
                assert results['patents_processed'] >= 2  # 正常な2件は処理される
                
                # 統計にエラー数が記録されることを確認
                stats = screener.get_processing_stats()
                assert stats.get('extraction_errors', 0) > 0


class TestPerformanceAndScaling:
    """性能・スケーリングテスト"""

    def test_large_dataset_processing(self):
        """大規模データセット処理テスト"""
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        
        invention_idea = {"title": "テスト発明", "problem": "問題", "solution": "解決策"}
        
        # 100件の特許データ生成
        large_patents_data = []
        for i in range(100):
            large_patents_data.append({
                "publication_number": f"JP2025-{i:06d}A",
                "title": f"テスト特許{i}",
                "assignee": f"会社{i}",
                "pub_date": "2025-01-15",
                "abstract": f"テスト抄録{i}",
                "claims": [{"no": 1, "text": f"テストクレーム{i}", "is_independent": True}]
            })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            invention_file = Path(temp_dir) / "invention.json"
            patents_file = Path(temp_dir) / "patents.jsonl"
            output_csv = Path(temp_dir) / "results.csv"
            
            with open(invention_file, 'w', encoding='utf-8') as f:
                json.dump(invention_idea, f, ensure_ascii=False)
            
            with open(patents_file, 'w', encoding='utf-8') as f:
                for patent in large_patents_data:
                    f.write(json.dumps(patent, ensure_ascii=False) + '\n')
            
            with patch('src.core.screener.LLMClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                def mock_completion(messages, **kwargs):
                    if "発明アイデアを要約" in messages[-1]['content']:
                        return {"summary": "発明要約"}
                    elif "特許を要約" in messages[-1]['content']:
                        return {"summary": "特許要約"}
                    else:
                        return {
                            "decision": "hit" if hash(messages[-1]['content']) % 2 else "miss",
                            "confidence": 0.5 + (hash(messages[-1]['content']) % 50) / 100,
                            "reasoning": [{"reason": "テスト", "source": "claim"}]
                        }
                
                mock_client.completion.side_effect = mock_completion
                mock_client.get_usage_stats.return_value = {'total_requests': 201, 'failed_requests': 0}
                
                # 処理時間測定
                start_time = time.time()
                results = screener.analyze(
                    invention=str(invention_file),
                    patents=str(patents_file),
                    output_csv=str(output_csv)
                )
                processing_time = time.time() - start_time
                
                # 性能要件の確認
                assert processing_time < 30.0  # 30秒以内
                assert results['patents_processed'] == 100
                
                # 1特許あたりの処理時間
                avg_time = processing_time / 100
                assert avg_time < 0.3  # 1件あたり300ms以内

    def test_concurrent_processing_safety(self):
        """並行処理の安全性テスト"""
        import threading
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        results_list = []
        errors_list = []
        
        def process_worker(worker_id):
            try:
                invention = {"title": f"発明{worker_id}", "problem": "問題", "solution": "解決策"}
                patents = [{
                    "publication_number": f"JP2025-W{worker_id}-001A",
                    "title": f"特許{worker_id}",
                    "assignee": "会社",
                    "pub_date": "2025-01-15",
                    "abstract": "抄録",
                    "claims": [{"no": 1, "text": "クレーム", "is_independent": True}]
                }]
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    inv_file = Path(temp_dir) / f"inv_{worker_id}.json"
                    pat_file = Path(temp_dir) / f"pat_{worker_id}.jsonl"
                    out_csv = Path(temp_dir) / f"out_{worker_id}.csv"
                    
                    with open(inv_file, 'w', encoding='utf-8') as f:
                        json.dump(invention, f, ensure_ascii=False)
                    
                    with open(pat_file, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(patents[0], ensure_ascii=False) + '\n')
                    
                    with patch('src.core.screener.LLMClient') as mock_client_class:
                        mock_client = Mock()
                        mock_client_class.return_value = mock_client
                        
                        mock_client.completion.return_value = {
                            "summary": "テスト",
                            "decision": "hit",
                            "confidence": 0.8,
                            "reasoning": [{"reason": "テスト", "source": "claim"}]
                        }
                        mock_client.get_usage_stats.return_value = {'total_requests': 1, 'failed_requests': 0}
                        
                        result = screener.analyze(
                            invention=str(inv_file),
                            patents=str(pat_file),
                            output_csv=str(out_csv)
                        )
                        results_list.append((worker_id, result))
                        
            except Exception as e:
                errors_list.append((worker_id, e))
        
        # 5つのワーカーで並行処理
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 完了待機
        for thread in threads:
            thread.join()
        
        # エラーが発生しないことを確認
        assert len(errors_list) == 0, f"Concurrent processing errors: {errors_list}"
        assert len(results_list) == 5
        
        # 各ワーカーが正常に処理されたことを確認
        for worker_id, result in results_list:
            assert result['patents_processed'] == 1

    def test_memory_efficiency_large_dataset(self):
        """大規模データセットでのメモリ効率テスト"""
        import gc
        import psutil
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        
        # メモリ使用量測定
        gc.collect()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 500件の大きなデータセット
        large_invention = {
            "title": "大規模テスト発明",
            "problem": "テスト問題" * 100,  # 長いテキスト
            "solution": "テスト解決策" * 100,
            "effects": "テスト効果" * 100
        }
        
        large_patents_data = []
        for i in range(500):
            large_patents_data.append({
                "publication_number": f"JP2025-{i:06d}A",
                "title": f"大規模テスト特許{i}",
                "assignee": f"大手企業株式会社{i}",
                "pub_date": "2025-01-15",
                "abstract": "非常に長い抄録テキスト" * 50,  # メモリ使用量増加
                "claims": [
                    {
                        "no": 1,
                        "text": "非常に長いクレームテキスト" * 100,
                        "is_independent": True
                    }
                ]
            })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            invention_file = Path(temp_dir) / "large_invention.json"
            patents_file = Path(temp_dir) / "large_patents.jsonl"
            output_csv = Path(temp_dir) / "large_results.csv"
            
            with open(invention_file, 'w', encoding='utf-8') as f:
                json.dump(large_invention, f, ensure_ascii=False)
            
            with open(patents_file, 'w', encoding='utf-8') as f:
                for patent in large_patents_data:
                    f.write(json.dumps(patent, ensure_ascii=False) + '\n')
            
            with patch('src.core.screener.LLMClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                def mock_completion(messages, **kwargs):
                    return {
                        "summary": "要約",
                        "decision": "hit",
                        "confidence": 0.7,
                        "reasoning": [{"reason": "理由", "source": "claim"}]
                    }
                
                mock_client.completion.side_effect = mock_completion
                mock_client.get_usage_stats.return_value = {'total_requests': 1001, 'failed_requests': 0}
                
                # 大規模データ処理実行
                results = screener.analyze(
                    invention=str(invention_file),
                    patents=str(patents_file),
                    output_csv=str(output_csv)
                )
                
                peak_memory = process.memory_info().rss
                memory_increase = peak_memory - initial_memory
                
                # メモリ使用量が合理的な範囲内であることを確認
                assert memory_increase < 1024 * 1024 * 1024  # 1GB未満の増加
                assert results['patents_processed'] == 500
                
                # クリーンアップ
                del large_patents_data, large_invention
                gc.collect()


class TestConfigurationAndCustomization:
    """設定・カスタマイズテスト"""

    def test_custom_output_paths(self):
        """カスタム出力パス設定テスト"""
        from src.core.screener import PatentScreener
        
        custom_config = {
            'io': {
                'out_csv': 'custom/path/results.csv',
                'out_jsonl': 'custom/path/details.jsonl'
            }
        }
        
        screener = PatentScreener(config=custom_config)
        
        # 設定が正しく適用されることを確認
        assert screener._config['io']['out_csv'] == 'custom/path/results.csv'
        assert screener._config['io']['out_jsonl'] == 'custom/path/details.jsonl'

    def test_processing_parameters_customization(self):
        """処理パラメータカスタマイズテスト"""
        from src.core.screener import PatentScreener
        
        custom_config = {
            'processing': {
                'batch_size': 20,
                'max_workers': 10
            },
            'llm': {
                'temperature': 0.1,
                'max_tokens': 500
            }
        }
        
        screener = PatentScreener(config=custom_config)
        
        # 設定確認
        assert screener._config['processing']['batch_size'] == 20
        assert screener._config['processing']['max_workers'] == 10
        assert screener._config['llm']['temperature'] == 0.1
        assert screener._config['llm']['max_tokens'] == 500

    def test_extraction_parameters_customization(self):
        """抽出パラメータカスタマイズテスト"""
        from src.core.screener import PatentScreener
        
        custom_config = {
            'extraction': {
                'max_abstract_length': 1000,
                'include_dependent_claims': True
            }
        }
        
        screener = PatentScreener(config=custom_config)
        
        # 抽出設定が適用されることを確認
        assert screener._config['extraction']['max_abstract_length'] == 1000
        assert screener._config['extraction']['include_dependent_claims'] is True

    def test_ranking_parameters_customization(self):
        """ランキングパラメータカスタマイズテスト"""
        from src.core.screener import PatentScreener
        
        custom_config = {
            'ranking': {
                'method': 'llm_only',
                'tiebreaker': 'pub_date'
            }
        }
        
        screener = PatentScreener(config=custom_config)
        
        # ランキング設定が適用されることを確認
        assert screener._config['ranking']['method'] == 'llm_only'
        assert screener._config['ranking']['tiebreaker'] == 'pub_date'


class TestMonitoringAndLogging:
    """監視・ログテスト"""

    def test_processing_statistics_collection(self):
        """処理統計収集テスト"""
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        
        # 初期統計の確認
        initial_stats = screener.get_processing_stats()
        assert initial_stats['total_patents_processed'] == 0
        assert initial_stats['total_hits'] == 0
        assert initial_stats['total_misses'] == 0
        
        # 簡単な処理を実行
        invention = {"title": "テスト", "problem": "問題", "solution": "解決策"}
        patents = [{
            "publication_number": "JP2025-100001A",
            "title": "テスト特許",
            "assignee": "テスト会社",
            "pub_date": "2025-01-15",
            "abstract": "テスト抄録",
            "claims": [{"no": 1, "text": "テストクレーム", "is_independent": True}]
        }]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            inv_file = Path(temp_dir) / "invention.json"
            pat_file = Path(temp_dir) / "patents.jsonl"
            out_csv = Path(temp_dir) / "results.csv"
            
            with open(inv_file, 'w', encoding='utf-8') as f:
                json.dump(invention, f, ensure_ascii=False)
            
            with open(pat_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(patents[0], ensure_ascii=False) + '\n')
            
            with patch('src.core.screener.LLMClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                mock_client.completion.return_value = {
                    "summary": "テスト要約",
                    "decision": "hit", 
                    "confidence": 0.9,
                    "reasoning": [{"reason": "テスト理由", "source": "claim"}]
                }
                mock_client.get_usage_stats.return_value = {'total_requests': 3, 'failed_requests': 0}
                
                screener.analyze(
                    invention=str(inv_file),
                    patents=str(pat_file),
                    output_csv=str(out_csv)
                )
                
                # 処理後統計の確認
                final_stats = screener.get_processing_stats()
                assert final_stats['total_patents_processed'] == 1
                assert final_stats['total_hits'] == 1
                assert final_stats['total_misses'] == 0
                assert final_stats['processing_time'] > 0

    def test_detailed_component_statistics(self):
        """詳細コンポーネント統計テスト"""
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        
        # コンポーネント別統計の確認
        component_stats = screener.get_component_stats()
        
        # 各コンポーネントの統計が含まれることを確認
        assert 'pipeline' in component_stats
        assert 'sorter' in component_stats
        
        # 統計構造の確認
        for component, stats in component_stats.items():
            assert isinstance(stats, dict)

    def test_error_tracking_and_reporting(self):
        """エラー追跡・報告テスト"""
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        
        # エラー統計の初期確認
        initial_stats = screener.get_processing_stats()
        assert initial_stats['extraction_errors'] == 0
        assert initial_stats['llm_errors'] == 0
        
        # エラー発生シナリオ（LLMエラー）
        invention = {"title": "テスト", "problem": "問題", "solution": "解決策"}
        patents = [{
            "publication_number": "JP2025-100001A", 
            "title": "テスト特許",
            "assignee": "テスト会社",
            "pub_date": "2025-01-15",
            "abstract": "テスト抄録",
            "claims": [{"no": 1, "text": "テストクレーム", "is_independent": True}]
        }]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            inv_file = Path(temp_dir) / "invention.json"
            pat_file = Path(temp_dir) / "patents.jsonl"
            out_csv = Path(temp_dir) / "results.csv"
            
            with open(inv_file, 'w', encoding='utf-8') as f:
                json.dump(invention, f, ensure_ascii=False)
            
            with open(pat_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(patents[0], ensure_ascii=False) + '\n')
            
            with patch('src.core.screener.LLMClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                # LLMエラーを発生させる
                mock_client.completion.side_effect = Exception("LLM Error")
                mock_client.get_usage_stats.return_value = {'total_requests': 0, 'failed_requests': 1}
                
                try:
                    screener.analyze(
                        invention=str(inv_file),
                        patents=str(pat_file),
                        output_csv=str(out_csv),
                        continue_on_error=True
                    )
                except:
                    pass  # エラーは期待される
                
                # エラー統計が更新されることを確認
                final_stats = screener.get_processing_stats()
                assert final_stats['llm_errors'] > 0


class TestOutputFormatValidation:
    """出力フォーマット検証テスト"""

    def test_csv_output_format_compliance(self):
        """CSV出力フォーマット準拠テスト"""
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        
        invention = {"title": "テスト発明", "problem": "問題", "solution": "解決策"}
        patents = [
            {
                "publication_number": "JP2025-100001A",
                "title": "テスト特許1", 
                "assignee": "会社1",
                "pub_date": "2025-01-15",
                "abstract": "抄録1",
                "claims": [{"no": 1, "text": "クレーム1", "is_independent": True}],
                "url_hint": "https://example.com/1"
            },
            {
                "publication_number": "JP2025-100002A",
                "title": "テスト特許2",
                "assignee": "会社2", 
                "pub_date": "2025-01-16",
                "abstract": "抄録2",
                "claims": [{"no": 1, "text": "クレーム2", "is_independent": True}],
                "url_hint": "https://example.com/2"
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            inv_file = Path(temp_dir) / "invention.json"
            pat_file = Path(temp_dir) / "patents.jsonl"
            out_csv = Path(temp_dir) / "results.csv"
            
            with open(inv_file, 'w', encoding='utf-8') as f:
                json.dump(invention, f, ensure_ascii=False)
            
            with open(pat_file, 'w', encoding='utf-8') as f:
                for patent in patents:
                    f.write(json.dumps(patent, ensure_ascii=False) + '\n')
            
            with patch('src.core.screener.LLMClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                def mock_completion(messages, **kwargs):
                    if "発明アイデアを要約" in messages[-1]['content']:
                        return {"summary": "発明要約"}
                    elif "特許を要約" in messages[-1]['content']:
                        return {"summary": "特許要約"}
                    else:
                        pub_num = "JP2025-100001A" if "JP2025-100001A" in messages[-1]['content'] else "JP2025-100002A"
                        return {
                            "decision": "hit" if pub_num == "JP2025-100001A" else "miss",
                            "confidence": 0.85 if pub_num == "JP2025-100001A" else 0.25,
                            "reasoning": [
                                {"reason": f"理由1_{pub_num}", "source": "claim", "quote": "引用1"},
                                {"reason": f"理由2_{pub_num}", "source": "abstract", "quote": "引用2"}
                            ]
                        }
                
                mock_client.completion.side_effect = mock_completion
                mock_client.get_usage_stats.return_value = {'total_requests': 5, 'failed_requests': 0}
                
                screener.analyze(
                    invention=str(inv_file),
                    patents=str(pat_file),
                    output_csv=str(out_csv)
                )
                
                # CSV出力の検証
                with open(out_csv, 'r', encoding='utf-8') as f:
                    csv_reader = csv.DictReader(f)
                    rows = list(csv_reader)
                
                # 必要なカラムがすべて存在することを確認
                required_columns = [
                    'rank', 'pub_number', 'title', 'assignee', 'pub_date',
                    'decision', 'LLM_confidence', 'hit_reason_1', 'hit_src_1',
                    'hit_reason_2', 'hit_src_2', 'url_hint'
                ]
                
                for column in required_columns:
                    assert column in csv_reader.fieldnames
                
                # データの順序確認（信頼度降順）
                assert len(rows) == 2
                assert rows[0]['decision'] == 'hit'
                assert rows[0]['LLM_confidence'] == '0.85'
                assert rows[1]['decision'] == 'miss' 
                assert rows[1]['LLM_confidence'] == '0.25'
                
                # ランクの確認
                assert rows[0]['rank'] == '1'
                assert rows[1]['rank'] == '2'

    def test_jsonl_output_format_compliance(self):
        """JSONL出力フォーマット準拠テスト"""
        from src.core.screener import PatentScreener
        
        screener = PatentScreener()
        
        invention = {"title": "テスト発明", "problem": "問題", "solution": "解決策"}
        patent = {
            "publication_number": "JP2025-100001A",
            "title": "テスト特許",
            "assignee": "テスト会社",
            "pub_date": "2025-01-15", 
            "abstract": "テスト抄録",
            "claims": [{"no": 1, "text": "テストクレーム", "is_independent": True}],
            "url_hint": "https://example.com/1"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            inv_file = Path(temp_dir) / "invention.json"
            pat_file = Path(temp_dir) / "patents.jsonl"
            out_jsonl = Path(temp_dir) / "details.jsonl"
            
            with open(inv_file, 'w', encoding='utf-8') as f:
                json.dump(invention, f, ensure_ascii=False)
            
            with open(pat_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(patent, ensure_ascii=False) + '\n')
            
            with patch('src.core.screener.LLMClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                def mock_completion(messages, **kwargs):
                    if "発明アイデアを要約" in messages[-1]['content']:
                        return {"summary": "発明要約", "key_points": ["要素1", "要素2"]}
                    elif "特許を要約" in messages[-1]['content']:
                        return {"summary": "特許要約", "key_points": ["特徴1", "特徴2"]}
                    else:
                        return {
                            "decision": "hit",
                            "confidence": 0.85,
                            "reasoning": [
                                {"reason": "技術的関連性", "source": "claim", "quote": "テストクレーム"},
                                {"reason": "目的の一致", "source": "abstract", "quote": "テスト抄録"}
                            ]
                        }
                
                mock_client.completion.side_effect = mock_completion
                mock_client.get_usage_stats.return_value = {'total_requests': 3, 'failed_requests': 0}
                
                screener.analyze(
                    invention=str(inv_file),
                    patents=str(pat_file),
                    output_jsonl=str(out_jsonl)
                )
                
                # JSONL出力の検証
                with open(out_jsonl, 'r', encoding='utf-8') as f:
                    jsonl_data = [json.loads(line) for line in f]
                
                assert len(jsonl_data) == 1
                result = jsonl_data[0]
                
                # 必要なフィールドの存在確認
                required_fields = [
                    'rank', 'publication_number', 'decision', 'confidence',
                    'hit_reason_1', 'hit_src_1', 'hit_reason_2', 'hit_src_2',
                    'original_data', 'processing_time', 'llm_summary'
                ]
                
                for field in required_fields:
                    assert field in result
                
                # データの内容確認
                assert result['decision'] == 'hit'
                assert result['confidence'] == 0.85
                assert result['rank'] == 1
                assert result['publication_number'] == 'JP2025-100001A'
                assert isinstance(result['original_data'], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])