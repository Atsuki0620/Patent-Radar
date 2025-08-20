"""
Tests for sorter module - 特許ランキングソート機能のテスト

特許分析結果をLLM_confidenceでソートし、安定した順序を保証する機能をテストする：
1. LLM_confidenceによる降順ソート
2. pub_numberによるタイブレーカー（安定ソート）
3. 大規模データセットでの性能
4. メモリ効率的ソート

Phase A: Code-reviewerによるテスト設計レビューのためのテストケース。
"""

import pytest
import random
import time
import copy
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import psutil
import os
import threading


class TestPatentSorter:
    """特許ソート機能の基本テスト"""

    def test_sorter_initialization(self):
        """ソーター初期化のテスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        # デフォルト設定が適用されることを確認
        assert hasattr(sorter, '_config')
        assert hasattr(sorter, '_sort_stats')
        assert sorter._config['method'] == 'llm_only'
        assert sorter._config['tiebreaker'] == 'pub_number'

    def test_sorter_with_custom_config(self):
        """カスタム設定でのソーター初期化テスト"""
        from src.ranking.sorter import PatentSorter
        
        custom_config = {
            'ranking': {
                'method': 'llm_only',
                'tiebreaker': 'pub_number',
                'stable_sort': True
            }
        }
        
        sorter = PatentSorter(config=custom_config)
        
        # カスタム設定が適用されることを確認
        assert sorter._config['method'] == 'llm_only'
        assert sorter._config['tiebreaker'] == 'pub_number'
        assert sorter._config['stable_sort'] is True

    def test_sorter_config_validation(self):
        """ソーター設定検証のテスト"""
        from src.ranking.sorter import PatentSorter
        
        # 無効な設定でのエラーハンドリング
        invalid_configs = [
            {'ranking': {'method': 'invalid_method'}},  # 無効なソート方法
            {'ranking': {'tiebreaker': 'invalid_field'}},  # 無効なタイブレーカー
            {'ranking': {'stable_sort': 'not_boolean'}},  # 無効な型
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, Exception)):
                PatentSorter(config=invalid_config)

    def test_sort_statistics_initialization(self):
        """ソート統計情報の初期化テスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        stats = sorter.get_sort_stats()
        
        # 統計情報の構造確認
        expected_keys = [
            'total_sorted', 'sort_time', 'average_sort_time',
            'largest_dataset_size', 'stability_verified'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float, bool))


class TestLLMConfidenceSort:
    """LLM信頼度によるソートのテスト"""

    def test_basic_confidence_sort(self):
        """基本的な信頼度ソートテスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        # テストデータ（異なる信頼度）
        patent_results = [
            {
                'publication_number': 'JP2025-100001A',
                'decision': 'hit',
                'confidence': 0.85,
                'title': '高信頼度特許1'
            },
            {
                'publication_number': 'JP2025-100002A',
                'decision': 'hit',
                'confidence': 0.95,
                'title': '最高信頼度特許'
            },
            {
                'publication_number': 'JP2025-100003A',
                'decision': 'miss',
                'confidence': 0.25,
                'title': '低信頼度特許'
            },
            {
                'publication_number': 'JP2025-100004A',
                'decision': 'hit',
                'confidence': 0.75,
                'title': '中程度信頼度特許'
            }
        ]
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # 信頼度降順ソートの確認
        assert len(sorted_results) == 4
        confidences = [r['confidence'] for r in sorted_results]
        assert confidences == [0.95, 0.85, 0.75, 0.25]
        
        # ランク付けの確認
        assert sorted_results[0]['rank'] == 1
        assert sorted_results[1]['rank'] == 2
        assert sorted_results[2]['rank'] == 3
        assert sorted_results[3]['rank'] == 4

    def test_confidence_tiebreaker_sort(self):
        """信頼度同値時のタイブレーカーテスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        # 同じ信頼度の特許データ
        patent_results = [
            {
                'publication_number': 'JP2025-100003A',  # アルファベット順で後
                'decision': 'hit',
                'confidence': 0.85,
                'title': '同信頼度特許3'
            },
            {
                'publication_number': 'JP2025-100001A',  # アルファベット順で最初
                'decision': 'hit',
                'confidence': 0.85,
                'title': '同信頼度特許1'
            },
            {
                'publication_number': 'JP2025-100002A',  # アルファベット順で中間
                'decision': 'hit',
                'confidence': 0.85,
                'title': '同信頼度特許2'
            }
        ]
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # タイブレーカー（pub_number昇順）の確認
        pub_numbers = [r['publication_number'] for r in sorted_results]
        assert pub_numbers == ['JP2025-100001A', 'JP2025-100002A', 'JP2025-100003A']
        
        # すべて同じランクになることを確認（同点）
        for result in sorted_results:
            assert result['rank'] == 1

    def test_mixed_confidence_and_tiebreaker(self):
        """信頼度とタイブレーカーの混合ソートテスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        patent_results = [
            {'publication_number': 'JP2025-100005A', 'confidence': 0.75, 'decision': 'hit'},
            {'publication_number': 'JP2025-100001A', 'confidence': 0.85, 'decision': 'hit'},
            {'publication_number': 'JP2025-100003A', 'confidence': 0.85, 'decision': 'hit'},
            {'publication_number': 'JP2025-100002A', 'confidence': 0.85, 'decision': 'hit'},
            {'publication_number': 'JP2025-100004A', 'confidence': 0.65, 'decision': 'hit'},
        ]
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # 期待される順序：信頼度0.85の3つ（pub_number昇順）→ 0.75 → 0.65
        expected_pub_numbers = [
            'JP2025-100001A',  # 0.85, rank 1
            'JP2025-100002A',  # 0.85, rank 1 
            'JP2025-100003A',  # 0.85, rank 1
            'JP2025-100005A',  # 0.75, rank 4
            'JP2025-100004A'   # 0.65, rank 5
        ]
        
        actual_pub_numbers = [r['publication_number'] for r in sorted_results]
        assert actual_pub_numbers == expected_pub_numbers
        
        # ランクの確認
        expected_ranks = [1, 1, 1, 4, 5]
        actual_ranks = [r['rank'] for r in sorted_results]
        assert actual_ranks == expected_ranks

    def test_hit_miss_mixed_sort(self):
        """hit/miss混在データのソートテスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        patent_results = [
            {'publication_number': 'JP2025-100001A', 'confidence': 0.85, 'decision': 'hit'},
            {'publication_number': 'JP2025-100002A', 'confidence': 0.95, 'decision': 'miss'},  # missでも高信頼度
            {'publication_number': 'JP2025-100003A', 'confidence': 0.75, 'decision': 'hit'},
            {'publication_number': 'JP2025-100004A', 'confidence': 0.25, 'decision': 'miss'},
        ]
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # 信頼度のみでソート（hit/missは区別しない）
        confidences = [r['confidence'] for r in sorted_results]
        assert confidences == [0.95, 0.85, 0.75, 0.25]
        
        # hit/missが混在していることを確認
        decisions = [r['decision'] for r in sorted_results]
        assert 'hit' in decisions and 'miss' in decisions


class TestSortStability:
    """ソート安定性のテスト"""

    def test_sort_stability_verification(self):
        """ソートの安定性検証テスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        # 元の順序を持つテストデータ
        patent_results = [
            {'publication_number': 'JP2025-100001A', 'confidence': 0.85, 'original_index': 0},
            {'publication_number': 'JP2025-100002A', 'confidence': 0.75, 'original_index': 1},
            {'publication_number': 'JP2025-100003A', 'confidence': 0.85, 'original_index': 2},
            {'publication_number': 'JP2025-100004A', 'confidence': 0.75, 'original_index': 3},
        ]
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # 同じ信頼度内でpub_numberでソートされることを確認
        confidence_85_results = [r for r in sorted_results if r['confidence'] == 0.85]
        confidence_75_results = [r for r in sorted_results if r['confidence'] == 0.75]
        
        # 信頼度0.85グループ内でのpub_number順序確認
        pub_numbers_85 = [r['publication_number'] for r in confidence_85_results]
        assert pub_numbers_85 == ['JP2025-100001A', 'JP2025-100003A']
        
        # 信頼度0.75グループ内でのpub_number順序確認
        pub_numbers_75 = [r['publication_number'] for r in confidence_75_results]
        assert pub_numbers_75 == ['JP2025-100002A', 'JP2025-100004A']

    def test_multiple_sort_consistency(self):
        """複数回ソートの一貫性テスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        # ランダムなテストデータ
        patent_results = []
        for i in range(50):
            patent_results.append({
                'publication_number': f'JP2025-{i:06d}A',
                'confidence': random.choice([0.95, 0.85, 0.75, 0.65, 0.55]),
                'decision': random.choice(['hit', 'miss'])
            })
        
        # 複数回ソートして一貫性を確認
        first_sort = sorter.sort_patents(copy.deepcopy(patent_results))
        second_sort = sorter.sort_patents(copy.deepcopy(patent_results))
        third_sort = sorter.sort_patents(copy.deepcopy(patent_results))
        
        # 結果の一貫性確認
        first_order = [r['publication_number'] for r in first_sort]
        second_order = [r['publication_number'] for r in second_sort]
        third_order = [r['publication_number'] for r in third_sort]
        
        assert first_order == second_order == third_order

    def test_sort_stability_large_dataset(self):
        """大規模データセットでの安定性テスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        # 1000件のテストデータ
        patent_results = []
        confidence_values = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25]
        
        for i in range(1000):
            patent_results.append({
                'publication_number': f'JP2025-{i:06d}A',
                'confidence': random.choice(confidence_values),
                'decision': random.choice(['hit', 'miss']),
                'title': f'特許タイトル{i}'
            })
        
        start_time = time.time()
        sorted_results = sorter.sort_patents(patent_results)
        sort_time = time.time() - start_time
        
        # 大規模データセットでも合理的な時間で完了することを確認
        assert sort_time < 5.0  # 5秒以内
        assert len(sorted_results) == 1000
        
        # ソート順序の検証（信頼度降順）
        confidences = [r['confidence'] for r in sorted_results]
        assert confidences == sorted(confidences, reverse=True)


class TestSortPerformance:
    """ソート性能のテスト"""

    def test_sort_performance_scaling(self):
        """ソート性能スケーリングテスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        dataset_sizes = [10, 100, 500, 1000]
        sort_times = []
        
        for size in dataset_sizes:
            # テストデータ生成
            patent_results = []
            for i in range(size):
                patent_results.append({
                    'publication_number': f'JP2025-{i:06d}A',
                    'confidence': random.random(),
                    'decision': random.choice(['hit', 'miss'])
                })
            
            # ソート時間測定
            start_time = time.time()
            sorted_results = sorter.sort_patents(patent_results)
            sort_time = time.time() - start_time
            sort_times.append(sort_time)
            
            # 結果の妥当性確認
            assert len(sorted_results) == size
        
        # パフォーマンスが合理的に向上することを確認
        # O(n log n)の性能特性を期待
        assert all(t < 10.0 for t in sort_times)  # すべて10秒以内

    def test_sort_memory_efficiency(self):
        """ソートメモリ効率性テスト"""
        import gc
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        # メモリ使用量測定
        gc.collect()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 大きなデータセットでソート
        large_dataset = []
        for i in range(5000):
            large_dataset.append({
                'publication_number': f'JP2025-{i:06d}A',
                'confidence': random.random(),
                'decision': random.choice(['hit', 'miss']),
                'title': '非常に長いタイトル' * 100,  # メモリ使用量を増加
                'abstract': '非常に長い抄録' * 200,
                'claim_1': '非常に長いクレーム' * 150
            })
        
        sorted_results = sorter.sort_patents(large_dataset)
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # メモリ使用量が合理的な範囲内であることを確認
        assert memory_increase < 500 * 1024 * 1024  # 500MB未満の増加
        assert len(sorted_results) == 5000
        
        # クリーンアップ
        del large_dataset, sorted_results
        gc.collect()

    def test_concurrent_sort_safety(self):
        """並行ソートの安全性テスト"""
        import threading
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        results = []
        errors = []
        
        def sort_worker(worker_id, patent_count):
            try:
                # ワーカー固有のテストデータ
                patent_results = []
                for i in range(patent_count):
                    patent_results.append({
                        'publication_number': f'JP2025-W{worker_id}-{i:04d}A',
                        'confidence': random.random(),
                        'decision': random.choice(['hit', 'miss'])
                    })
                
                sorted_result = sorter.sort_patents(patent_results)
                results.append((worker_id, len(sorted_result)))
                
            except Exception as e:
                errors.append((worker_id, e))
        
        # 5つのワーカーで並行ソート実行
        threads = []
        for i in range(5):
            thread = threading.Thread(target=sort_worker, args=(i, 100))
            threads.append(thread)
            thread.start()
        
        # すべてのスレッドの完了を待機
        for thread in threads:
            thread.join()
        
        # エラーが発生しないことを確認
        assert len(errors) == 0, f"Concurrent sort errors: {errors}"
        assert len(results) == 5
        
        # すべてのワーカーが正しく処理されたことを確認
        for worker_id, result_count in results:
            assert result_count == 100


class TestSortValidation:
    """ソート入力検証のテスト"""

    def test_invalid_input_handling(self):
        """無効入力のハンドリングテスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        # 無効な入力パターン
        invalid_inputs = [
            None,                           # None
            [],                            # 空リスト
            "not a list",                  # 文字列
            [None, None],                  # None要素
            [{"invalid": "no_confidence"}],  # 必須フィールド欠損
        ]
        
        for invalid_input in invalid_inputs:
            try:
                result = sorter.sort_patents(invalid_input)
                
                if invalid_input == []:
                    # 空リストは有効な入力として処理
                    assert result == []
                elif invalid_input == [{"invalid": "no_confidence"}]:
                    # 欠損フィールドは適切にハンドリング
                    assert len(result) == 1
                    assert 'confidence' not in result[0] or result[0]['confidence'] is None
                else:
                    # 他の無効入力はエラーまたは空結果を期待
                    assert result == [] or result is None
                    
            except (ValueError, TypeError, AttributeError):
                # エラーが発生することも期待される動作
                pass

    def test_missing_confidence_handling(self):
        """信頼度欠損データのハンドリングテスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        # 信頼度が欠損または不正なデータ
        patent_results = [
            {
                'publication_number': 'JP2025-100001A',
                'confidence': 0.85,
                'decision': 'hit'
            },
            {
                'publication_number': 'JP2025-100002A',
                # confidence フィールドなし
                'decision': 'hit'
            },
            {
                'publication_number': 'JP2025-100003A',
                'confidence': None,  # None値
                'decision': 'miss'
            },
            {
                'publication_number': 'JP2025-100004A',
                'confidence': "invalid",  # 文字列
                'decision': 'hit'
            }
        ]
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # 有効な信頼度を持つデータが正しく処理されることを確認
        assert len(sorted_results) >= 1  # 少なくとも1つは有効
        
        # 無効データの処理方法を確認
        valid_results = [r for r in sorted_results if isinstance(r.get('confidence'), (int, float))]
        assert len(valid_results) >= 1

    def test_publication_number_validation(self):
        """特許番号検証のテスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        patent_results = [
            {
                'publication_number': 'JP2025-100001A',
                'confidence': 0.85,
                'decision': 'hit'
            },
            {
                'publication_number': '',  # 空文字列
                'confidence': 0.75,
                'decision': 'hit'
            },
            {
                # publication_number フィールドなし
                'confidence': 0.65,
                'decision': 'miss'
            },
            {
                'publication_number': None,  # None値
                'confidence': 0.55,
                'decision': 'hit'
            }
        ]
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # 結果が返されることを確認（エラーにならない）
        assert isinstance(sorted_results, list)
        assert len(sorted_results) > 0


class TestSortConfiguration:
    """ソート設定・カスタマイズテスト"""

    def test_custom_tiebreaker_field(self):
        """カスタムタイブレーカーフィールドのテスト"""
        from src.ranking.sorter import PatentSorter
        
        # pub_date をタイブレーカーとする設定
        custom_config = {
            'ranking': {
                'method': 'llm_only',
                'tiebreaker': 'pub_date'
            }
        }
        
        sorter = PatentSorter(config=custom_config)
        
        patent_results = [
            {
                'publication_number': 'JP2025-100003A',
                'pub_date': '2025-03-01',
                'confidence': 0.85,
                'decision': 'hit'
            },
            {
                'publication_number': 'JP2025-100001A', 
                'pub_date': '2025-01-01',
                'confidence': 0.85,
                'decision': 'hit'
            },
            {
                'publication_number': 'JP2025-100002A',
                'pub_date': '2025-02-01',
                'confidence': 0.85,
                'decision': 'hit'
            }
        ]
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # pub_date順（昇順）でタイブレーカーされることを確認
        pub_dates = [r['pub_date'] for r in sorted_results]
        assert pub_dates == ['2025-01-01', '2025-02-01', '2025-03-01']

    def test_sort_method_configuration(self):
        """ソート方法設定のテスト"""
        from src.ranking.sorter import PatentSorter
        
        # 将来の拡張性のためのテスト
        llm_only_config = {
            'ranking': {
                'method': 'llm_only',
                'tiebreaker': 'pub_number'
            }
        }
        
        sorter = PatentSorter(config=llm_only_config)
        
        patent_results = [
            {'publication_number': 'JP001', 'confidence': 0.75, 'decision': 'hit'},
            {'publication_number': 'JP002', 'confidence': 0.85, 'decision': 'hit'},
        ]
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # LLM信頼度のみでソートされることを確認
        confidences = [r['confidence'] for r in sorted_results]
        assert confidences == [0.85, 0.75]

    def test_stable_sort_flag(self):
        """安定ソートフラグのテスト"""
        from src.ranking.sorter import PatentSorter
        
        stable_config = {
            'ranking': {
                'method': 'llm_only',
                'tiebreaker': 'pub_number',
                'stable_sort': True
            }
        }
        
        sorter = PatentSorter(config=stable_config)
        
        # 安定ソートが有効であることを設定で確認
        assert sorter._config['stable_sort'] is True
        
        # 実際のソート動作の確認は他のテストでカバー
        patent_results = [
            {'publication_number': 'JP001', 'confidence': 0.85, 'decision': 'hit'},
            {'publication_number': 'JP002', 'confidence': 0.85, 'decision': 'hit'},
        ]
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # 安定ソートが実行されることを確認
        pub_numbers = [r['publication_number'] for r in sorted_results]
        assert pub_numbers == ['JP001', 'JP002']  # 元の相対順序または昇順


class TestSortIntegration:
    """ソート統合機能のテスト"""

    def test_ranking_with_statistics_collection(self):
        """統計収集付きランキングテスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        patent_results = []
        for i in range(100):
            patent_results.append({
                'publication_number': f'JP2025-{i:06d}A',
                'confidence': random.random(),
                'decision': random.choice(['hit', 'miss'])
            })
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # 統計情報の確認
        stats = sorter.get_sort_stats()
        
        assert stats['total_sorted'] >= 100
        assert stats['sort_time'] > 0
        assert stats['largest_dataset_size'] >= 100

    def test_integration_with_pipeline_output(self):
        """パイプライン出力との統合テスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        # パイプラインからの典型的な出力フォーマット
        pipeline_output = [
            {
                'publication_number': 'JP2025-100001A',
                'decision': 'hit',
                'confidence': 0.85,
                'hit_reason_1': '技術分野一致',
                'hit_src_1': 'claim',
                'hit_reason_2': '解決手段類似',
                'hit_src_2': 'abstract',
                'processing_time': 2.5,
                'original_data': {'title': '特許タイトル1'}
            },
            {
                'publication_number': 'JP2025-100002A',
                'decision': 'miss',
                'confidence': 0.25,
                'hit_reason_1': '技術分野相違',
                'hit_src_1': 'claim',
                'hit_reason_2': '',
                'hit_src_2': '',
                'processing_time': 2.1,
                'original_data': {'title': '特許タイトル2'}
            }
        ]
        
        sorted_results = sorter.sort_patents(pipeline_output)
        
        # パイプライン出力の構造が保持されることを確認
        assert all('hit_reason_1' in r for r in sorted_results)
        assert all('processing_time' in r for r in sorted_results)
        assert all('original_data' in r for r in sorted_results)
        
        # ランクが正しく付与されることを確認
        assert all('rank' in r for r in sorted_results)
        
        # 信頼度順ソートの確認
        confidences = [r['confidence'] for r in sorted_results]
        assert confidences == [0.85, 0.25]

    def test_export_ready_format(self):
        """エクスポート対応フォーマットテスト"""
        from src.ranking.sorter import PatentSorter
        
        sorter = PatentSorter()
        
        patent_results = [
            {
                'publication_number': 'JP2025-100001A',
                'title': '膜分離装置の監視システム',
                'assignee': '技術株式会社',
                'pub_date': '2025-01-15',
                'decision': 'hit',
                'confidence': 0.85,
                'hit_reason_1': '技術分野一致',
                'hit_src_1': 'claim',
                'hit_reason_2': '解決手段類似',
                'hit_src_2': 'abstract',
                'url_hint': 'https://example.com/patent1'
            },
            {
                'publication_number': 'JP2025-100002A',
                'title': '液体処理システム',
                'assignee': '化学工業株式会社',
                'pub_date': '2025-02-01',
                'decision': 'hit',
                'confidence': 0.92,
                'hit_reason_1': '同一技術要素',
                'hit_src_1': 'claim',
                'hit_reason_2': '効果一致',
                'hit_src_2': 'abstract',
                'url_hint': 'https://example.com/patent2'
            }
        ]
        
        sorted_results = sorter.sort_patents(patent_results)
        
        # エクスポートに必要なフィールドがすべて保持されることを確認
        required_fields = [
            'rank', 'publication_number', 'title', 'assignee', 'pub_date',
            'decision', 'confidence', 'hit_reason_1', 'hit_src_1',
            'hit_reason_2', 'hit_src_2', 'url_hint'
        ]
        
        for result in sorted_results:
            for field in required_fields:
                assert field in result
        
        # 正しいランクとソート順序の確認
        assert sorted_results[0]['confidence'] == 0.92
        assert sorted_results[0]['rank'] == 1
        assert sorted_results[1]['confidence'] == 0.85
        assert sorted_results[1]['rank'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])