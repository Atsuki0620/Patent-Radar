"""
Tests for pipeline module - LLM処理パイプライン機能のテスト

特許分析のLLM処理パイプラインをテストする：
1. 発明要約バッチ処理
2. 特許要約バッチ処理  
3. 二値分類バッチ処理
4. パイプライン統合処理

Phase A: Code-reviewerによるテスト設計レビューのためのテストケース。
"""

import pytest
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path


class TestLLMPipeline:
    """LLM処理パイプラインの基本機能テスト"""

    @patch('src.llm.client.LLMClient')
    def test_pipeline_initialization(self, mock_client_class):
        """パイプライン初期化のテスト"""
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        # 必要なコンポーネントが初期化されることを確認
        assert hasattr(pipeline, '_client')
        assert hasattr(pipeline, '_config')
        assert hasattr(pipeline, '_stats')
        mock_client_class.assert_called_once()

    @patch('src.llm.client.LLMClient')
    def test_pipeline_with_custom_config(self, mock_client_class):
        """カスタム設定でのパイプライン初期化テスト"""
        from src.llm.pipeline import LLMPipeline
        
        custom_config = {
            'processing': {
                'batch_size': 5,
                'max_workers': 3,
                'timeout': 60
            },
            'llm': {
                'model': 'gpt-4o',
                'temperature': 0.0
            }
        }
        
        pipeline = LLMPipeline(config=custom_config)
        
        # カスタム設定が適用されることを確認
        assert pipeline._config['processing']['batch_size'] == 5
        assert pipeline._config['processing']['max_workers'] == 3

    def test_pipeline_config_validation(self):
        """パイプライン設定検証のテスト"""
        from src.llm.pipeline import LLMPipeline
        
        # 無効な設定でのエラーハンドリング
        invalid_configs = [
            {'processing': {'batch_size': 0}},  # 無効なバッチサイズ
            {'processing': {'max_workers': -1}},  # 無効なワーカー数
            {'processing': {'timeout': -5}},  # 無効なタイムアウト
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, Exception)):  # PipelineConfigErrorも含める
                LLMPipeline(config=invalid_config)

    def test_pipeline_stats_initialization(self):
        """パイプライン統計情報の初期化テスト"""
        from src.llm.pipeline import LLMPipeline
        
        pipeline = LLMPipeline()
        stats = pipeline.get_stats()
        
        # 初期統計情報の構造確認
        expected_keys = [
            'total_processed', 'successful_processed', 'failed_processed',
            'invention_summaries', 'patent_summaries', 'classifications',
            'total_tokens_used', 'processing_time', 'average_processing_time'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))


class TestInventionBatchProcessing:
    """発明要約バッチ処理のテスト"""
    
    def setup_method(self):
        """各テストメソッド実行前の初期化"""
        # グローバル状態をクリア
        import gc
        gc.collect()
    
    def teardown_method(self):
        """各テストメソッド実行後のクリーンアップ"""
        # グローバル状態をクリア
        import gc
        gc.collect()

    @patch('src.llm.client.LLMClient')
    def test_invention_batch_processing_basic(self, mock_client_class):
        """基本的な発明バッチ処理テスト"""
        from src.llm.pipeline import LLMPipeline
        
        # モッククライアントの設定
        mock_client = Mock()
        mock_client.reset_mock()  # 明示的にモックをリセット
        
        # 基本テスト用のside_effect
        def basic_batch_side_effect(*args, **kwargs):
            return [
                {'content': '液体分離設備の運転データ予測システム'},
                {'content': '膜分離装置の性能監視システム'}
            ]
        
        mock_client.batch_completion.side_effect = basic_batch_side_effect
        mock_client.batch_completion.return_value = None  # side_effectを使用するためreturn_valueはクリア
        mock_client_class.reset_mock()
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        inventions = [
            {
                'title': '運転データ予測システム',
                'problem': '性能劣化の予測困難',
                'solution': '機械学習アルゴリズム'
            },
            {
                'title': '性能監視システム',
                'problem': 'リアルタイム監視不足',
                'solution': 'IoTセンサー活用'
            }
        ]
        
        results = pipeline.process_invention_batch(inventions)
        
        # 結果の検証
        assert len(results) == 2
        assert all('summary' in result for result in results)
        assert all('processing_time' in result for result in results)
        mock_client.batch_completion.assert_called_once()

    @patch('src.llm.client.LLMClient')
    def test_invention_batch_with_errors(self, mock_client_class):
        """エラーを含む発明バッチ処理テスト"""
        from src.llm.pipeline import LLMPipeline
        
        # エラーを含むモック応答
        mock_client = Mock()
        mock_client.reset_mock()  # 明示的にモックをリセット
        
        # 固定のレスポンス配列を返すside_effectを設定
        def error_batch_side_effect(*args, **kwargs):
            return [
                {'content': '有効な要約結果'},
                {'error': 'API error', 'error_type': 'LLMAPIError'}
            ]
        
        mock_client.batch_completion.side_effect = error_batch_side_effect
        mock_client.batch_completion.return_value = None  # side_effectを使用するためreturn_valueはクリア
        mock_client_class.reset_mock()
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        inventions = [
            {'title': '正常な発明'},
            {'title': 'エラー発明'}
        ]
        
        results = pipeline.process_invention_batch(inventions, continue_on_error=True)
        
        # エラーハンドリングの確認
        assert len(results) == 2
        assert 'summary' in results[0]
        assert 'error' in results[1]
        assert results[1]['status'] == 'failed'

    @patch('src.llm.client.LLMClient')
    def test_invention_batch_large_dataset(self, mock_client_class):
        """大規模データセットでの発明バッチ処理テスト"""
        from src.llm.pipeline import LLMPipeline
        
        # バッチごとの適切な応答をモック (バッチサイズは10)
        mock_client = Mock()
        mock_client.reset_mock()  # 明示的にモックをリセット
        
        # バッチサイズ10で処理されるため、各バッチで10個のレスポンスを返すようにする
        def batch_side_effect(*args, **kwargs):
            prompts = args[0]  # 第一引数がプロンプトのリスト
            return [{'content': f'要約結果{i}'} for i in range(len(prompts))]
        
        mock_client.batch_completion.side_effect = batch_side_effect
        mock_client.batch_completion.return_value = None  # side_effectを使用するためreturn_valueはクリア
        mock_client_class.reset_mock()
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        # 100件の発明データ
        inventions = [{'title': f'発明{i}'} for i in range(100)]
        
        start_time = time.time()
        results = pipeline.process_invention_batch(inventions)
        processing_time = time.time() - start_time
        
        # 性能とスケーラビリティの確認
        assert len(results) == 100
        assert processing_time < 30  # 30秒以内での処理
        assert all('summary' in result for result in results)

    @patch('src.llm.client.LLMClient')
    def test_invention_batch_memory_efficiency(self, mock_client_class):
        """発明バッチ処理のメモリ効率性テスト"""
        import gc
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.reset_mock()  # 明示的にモックをリセット
        
        # メモリ効率テスト用のside_effect
        def memory_batch_side_effect(*args, **kwargs):
            prompts = args[0]
            return [{'content': '要約'}] * len(prompts)
        
        mock_client.batch_completion.side_effect = memory_batch_side_effect
        mock_client.batch_completion.return_value = None  # side_effectを使用するためreturn_valueはクリア
        mock_client_class.reset_mock()
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        # メモリ使用量測定
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # 大きな発明データで処理
        large_inventions = [
            {'title': '大きな発明' * 1000, 'problem': '大きな問題' * 1000}
            for _ in range(50)
        ]
        
        results = pipeline.process_invention_batch(large_inventions)
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # メモリリークがないことを確認
        object_increase = final_objects - initial_objects
        assert object_increase < 1000  # 合理的な増加範囲


class TestPatentBatchProcessing:
    """特許要約バッチ処理のテスト"""

    @patch('src.llm.client.LLMClient')
    def test_patent_batch_processing_basic(self, mock_client_class):
        """基本的な特許バッチ処理テスト"""
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.batch_completion.return_value = [
            {'content': '膜分離装置の運転監視システム'},
            {'content': '液体処理設備の性能管理装置'}
        ]
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        patents = [
            {
                'publication_number': 'JP2025-100001A',
                'title': '膜分離装置',
                'claim_1': '運転データを収集する手段を備えた装置',
                'abstract': '膜分離装置の監視システム'
            },
            {
                'publication_number': 'JP2025-100002A',
                'title': '液体処理設備',
                'claim_1': '性能データを管理する手段',
                'abstract': '液体処理設備の管理装置'
            }
        ]
        
        invention_summary = '液体分離設備の予測システム'
        
        results = pipeline.process_patent_batch(patents, invention_summary)
        
        # 結果の検証
        assert len(results) == 2
        assert all('summary' in result for result in results)
        assert all('publication_number' in result for result in results)
        mock_client.batch_completion.assert_called_once()

    @patch('src.llm.client.LLMClient')
    def test_patent_batch_with_missing_fields(self, mock_client_class):
        """フィールド欠損のある特許バッチ処理テスト"""
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.batch_completion.return_value = [
            {'content': '不完全な特許の要約'},
        ]
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        incomplete_patents = [
            {
                'publication_number': 'JP2025-INCOMPLETE',
                'title': '不完全特許',
                'claim_1': None,  # 欠損
                'abstract': ''    # 空文字
            }
        ]
        
        invention_summary = 'テスト発明'
        
        results = pipeline.process_patent_batch(incomplete_patents, invention_summary)
        
        # 不完全データのハンドリング確認
        assert len(results) == 1
        assert results[0]['status'] in ['success', 'partial']
        assert 'summary' in results[0]

    @patch('src.llm.client.LLMClient') 
    def test_patent_batch_concurrent_processing(self, mock_client_class):
        """特許バッチの並行処理テスト"""
        from src.llm.pipeline import LLMPipeline
        
        # 並行処理対応のモッククライアント
        mock_client = Mock()
        mock_client.concurrent_batch_completion.return_value = [
            {'content': f'並行処理要約{i}'} for i in range(20)
        ]
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        patents = [
            {'publication_number': f'JP2025-{i:06d}A', 'title': f'特許{i}'}
            for i in range(20)
        ]
        
        invention_summary = '並行処理テスト発明'
        
        start_time = time.time()
        results = pipeline.process_patent_batch(
            patents, invention_summary, concurrent=True
        )
        processing_time = time.time() - start_time
        
        # 並行処理の効率性確認
        assert len(results) == 20
        assert processing_time < 10  # 並行処理により高速化
        mock_client.concurrent_batch_completion.assert_called_once()


class TestClassificationBatchProcessing:
    """二値分類バッチ処理のテスト"""

    @patch('src.llm.client.LLMClient')
    def test_classification_batch_processing_basic(self, mock_client_class):
        """基本的な分類バッチ処理テスト"""
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.batch_completion.return_value = [
            {
                'content': json.dumps({
                    'decision': 'hit',
                    'confidence': 0.85,
                    'hit_reason_1': '同一技術分野',
                    'hit_src_1': 'claim',
                    'hit_reason_2': '類似解決手段',
                    'hit_src_2': 'abstract'
                })
            },
            {
                'content': json.dumps({
                    'decision': 'miss',
                    'confidence': 0.25,
                    'hit_reason_1': '技術分野相違',
                    'hit_src_1': 'claim',
                    'hit_reason_2': '',
                    'hit_src_2': ''
                })
            }
        ]
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        classification_pairs = [
            {
                'invention_summary': '液体分離設備の予測システム',
                'patent_summary': '膜分離装置の監視システム'
            },
            {
                'invention_summary': '液体分離設備の予測システム',
                'patent_summary': '自動車エンジンの制御システム'
            }
        ]
        
        results = pipeline.process_classification_batch(classification_pairs)
        
        # 結果の検証
        assert len(results) == 2
        assert results[0]['decision'] == 'hit'
        assert results[0]['confidence'] == 0.85
        assert results[1]['decision'] == 'miss'
        assert results[1]['confidence'] == 0.25

    @patch('src.llm.client.LLMClient')
    def test_classification_batch_json_validation(self, mock_client_class):
        """分類結果のJSON検証テスト"""
        from src.llm.pipeline import LLMPipeline
        
        # 無効なJSONを含むモック応答
        mock_client = Mock()
        mock_client.batch_completion.return_value = [
            {'content': json.dumps({'decision': 'hit', 'confidence': 0.9})},  # 有効
            {'content': 'invalid json response'},  # 無効なJSON
            {'content': json.dumps({'decision': 'invalid', 'confidence': 1.5})}  # 無効な値
        ]
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        classification_pairs = [
            {'invention_summary': 'テスト1', 'patent_summary': 'テスト1'},
            {'invention_summary': 'テスト2', 'patent_summary': 'テスト2'},
            {'invention_summary': 'テスト3', 'patent_summary': 'テスト3'}
        ]
        
        results = pipeline.process_classification_batch(
            classification_pairs, validate_json=True
        )
        
        # JSON検証結果の確認
        assert len(results) == 3
        assert results[0]['status'] == 'success'
        assert results[1]['status'] == 'failed'
        assert results[2]['status'] == 'failed'
        assert 'validation_error' in results[1]

    @patch('src.llm.client.LLMClient')
    def test_classification_confidence_distribution(self, mock_client_class):
        """分類信頼度分布のテスト"""
        from src.llm.pipeline import LLMPipeline
        
        # 様々な信頼度のモック応答
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        mock_responses = [
            {
                'content': json.dumps({
                    'decision': 'hit' if conf > 0.5 else 'miss',
                    'confidence': conf,
                    'hit_reason_1': '理由',
                    'hit_src_1': 'claim'
                })
            }
            for conf in confidences
        ]
        
        mock_client = Mock()
        mock_client.batch_completion.return_value = mock_responses
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        pairs = [
            {'invention_summary': f'発明{i}', 'patent_summary': f'特許{i}'}
            for i in range(len(confidences))
        ]
        
        results = pipeline.process_classification_batch(pairs)
        
        # 信頼度分布の確認
        hit_count = sum(1 for r in results if r['decision'] == 'hit')
        miss_count = sum(1 for r in results if r['decision'] == 'miss')
        
        assert hit_count == 4  # 信頼度 > 0.5
        assert miss_count == 3  # 信頼度 <= 0.5
        
        # 信頼度の範囲確認
        for result in results:
            assert 0.0 <= result['confidence'] <= 1.0


class TestPipelineIntegration:
    """パイプライン統合処理のテスト"""

    @patch('src.llm.client.LLMClient')
    def test_full_pipeline_processing(self, mock_client_class):
        """完全なパイプライン処理テスト"""
        from src.llm.pipeline import LLMPipeline
        
        # 段階的なモック応答を設定
        mock_client = Mock()
        
        def mock_completion(messages, **kwargs):
            content = messages[0]['content']
            if '発明内容を要約' in content:
                return {'content': '液体分離設備の運転予測システム'}
            elif '特許情報' in content:
                return {'content': '膜分離装置の監視システム'}
            elif '関連性を判定' in content:
                return {'content': json.dumps({
                    'decision': 'hit',
                    'confidence': 0.85,
                    'hit_reason_1': '技術分野一致',
                    'hit_src_1': 'claim'
                })}
        
        mock_client.completion = mock_completion
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        # 統合処理データ
        invention_data = {
            'title': '運転データ予測システム',
            'problem': '性能劣化予測困難',
            'solution': '機械学習活用'
        }
        
        patent_data = [
            {
                'publication_number': 'JP2025-100001A',
                'title': '膜分離監視装置',
                'claim_1': '運転データ収集手段を備えた装置',
                'abstract': '膜分離装置の監視システム'
            }
        ]
        
        results = pipeline.process_full_pipeline(invention_data, patent_data)
        
        # 統合結果の検証
        assert 'invention_summary' in results
        assert 'patent_results' in results
        assert len(results['patent_results']) == 1
        assert results['patent_results'][0]['decision'] == 'hit'
        assert results['patent_results'][0]['confidence'] == 0.85

    @patch('src.llm.client.LLMClient')
    def test_pipeline_with_progress_tracking(self, mock_client_class):
        """進捗追跡付きパイプライン処理テスト"""
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.completion.return_value = {'content': 'テスト応答'}
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        progress_updates = []
        
        def progress_callback(stage, current, total):
            progress_updates.append({
                'stage': stage,
                'current': current,
                'total': total,
                'percentage': (current / total) * 100
            })
        
        invention_data = {'title': 'テスト発明'}
        patent_data = [{'publication_number': f'JP{i}'} for i in range(5)]
        
        results = pipeline.process_full_pipeline(
            invention_data, patent_data, progress_callback=progress_callback
        )
        
        # 進捗追跡の確認
        assert len(progress_updates) > 0
        stages = [update['stage'] for update in progress_updates]
        assert 'invention_summary' in stages
        assert 'patent_summaries' in stages
        assert 'classification' in stages

    @patch('src.llm.client.LLMClient')
    def test_pipeline_error_recovery(self, mock_client_class):
        """パイプラインエラー回復テスト"""
        from src.llm.pipeline import LLMPipeline
        
        # 部分的なエラーを含むモッククライアント
        mock_client = Mock()
        
        def mock_completion_with_errors(messages, **kwargs):
            content = messages[0]['content']
            if 'エラー特許' in content:
                raise Exception("API エラー")
            return {'content': 'テスト応答'}
        
        mock_client.completion = mock_completion_with_errors
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        invention_data = {'title': 'テスト発明'}
        patent_data = [
            {'publication_number': 'JP-NORMAL', 'title': '正常特許'},
            {'publication_number': 'JP-ERROR', 'title': 'エラー特許'},
            {'publication_number': 'JP-NORMAL2', 'title': '正常特許2'}
        ]
        
        results = pipeline.process_full_pipeline(
            invention_data, patent_data, continue_on_error=True
        )
        
        # エラー回復の確認
        assert 'invention_summary' in results
        assert len(results['patent_results']) == 3
        
        # 正常処理とエラー処理の分離確認
        successful = [r for r in results['patent_results'] if r.get('status') == 'success']
        failed = [r for r in results['patent_results'] if r.get('status') == 'failed']
        
        assert len(successful) == 2
        assert len(failed) == 1


class TestPipelinePerformance:
    """パイプライン性能・スケーラビリティテスト"""

    @patch('src.llm.client.LLMClient')
    def test_pipeline_batch_optimization(self, mock_client_class):
        """パイプラインバッチ最適化テスト"""
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.batch_completion.return_value = [
            {'content': f'最適化テスト{i}'} for i in range(50)
        ]
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        # 大量データでの処理時間測定
        large_dataset = [{'title': f'特許{i}'} for i in range(50)]
        
        start_time = time.time()
        results = pipeline.process_patent_batch(large_dataset, '発明要約')
        processing_time = time.time() - start_time
        
        # 性能要件の確認
        assert len(results) == 50
        assert processing_time < 60  # 1分以内での処理
        
        # バッチ最適化が機能していることを確認
        call_count = mock_client.batch_completion.call_count
        assert call_count <= 10  # 効率的なバッチング

    @patch('src.llm.client.LLMClient')
    def test_pipeline_memory_management(self, mock_client_class):
        """パイプラインメモリ管理テスト"""
        import psutil
        import os
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.batch_completion.return_value = [
            {'content': '要約'} for _ in range(100)
        ]
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        # メモリ使用量測定
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 大量データ処理
        large_dataset = [
            {'title': '大きなタイトル' * 100, 'abstract': '大きな抄録' * 100}
            for _ in range(100)
        ]
        
        results = pipeline.process_patent_batch(large_dataset, '発明要約')
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # メモリ使用量が合理的な範囲内であることを確認
        assert memory_increase < 100 * 1024 * 1024  # 100MB未満の増加

    @patch('src.llm.client.LLMClient')
    def test_pipeline_concurrent_processing_limits(self, mock_client_class):
        """パイプライン並行処理制限テスト"""
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.concurrent_batch_completion.return_value = [
            {'content': f'並行{i}'} for i in range(20)
        ]
        mock_client_class.return_value = mock_client
        
        # ワーカー数制限設定
        config = {
            'processing': {
                'max_workers': 3,  # 制限された並行度
                'batch_size': 5
            }
        }
        
        pipeline = LLMPipeline(config=config)
        
        dataset = [{'title': f'特許{i}'} for i in range(20)]
        
        results = pipeline.process_patent_batch(
            dataset, '発明要約', concurrent=True
        )
        
        # 並行処理制限が守られることを確認
        assert len(results) == 20
        
        # 実際のワーカー数が制限を超えないことを確認（実装依存）
        # これは実装時にThreadPoolExecutorのmax_workersで制御される


class TestPipelineConfiguration:
    """パイプライン設定・カスタマイズテスト"""

    def test_pipeline_config_loading(self):
        """パイプライン設定読み込みテスト"""
        from src.llm.pipeline import LLMPipeline, load_pipeline_config
        
        # テスト用設定ファイル作成
        test_config = {
            'processing': {
                'batch_size': 8,
                'max_workers': 4,
                'timeout': 120
            },
            'llm': {
                'model': 'gpt-4o',
                'max_tokens': 400
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            loaded_config = load_pipeline_config(config_path)
            
            # 設定が正しく読み込まれることを確認
            assert loaded_config['processing']['batch_size'] == 8
            assert loaded_config['llm']['model'] == 'gpt-4o'
        finally:
            Path(config_path).unlink()  # クリーンアップ

    def test_pipeline_custom_prompts(self):
        """パイプラインカスタムプロンプトテスト"""
        from src.llm.pipeline import LLMPipeline
        
        custom_prompts = {
            'invention_system': 'カスタム発明分析システム',
            'patent_system': 'カスタム特許分析システム',
            'classification_system': 'カスタム分類システム'
        }
        
        pipeline = LLMPipeline(custom_prompts=custom_prompts)
        
        # カスタムプロンプトが適用されることを確認
        assert hasattr(pipeline, '_custom_prompts')
        assert pipeline._custom_prompts['invention_system'] == 'カスタム発明分析システム'

    def test_pipeline_plugin_system(self):
        """パイプラインプラグインシステムテスト"""
        from src.llm.pipeline import LLMPipeline
        
        # カスタムプロセッサプラグイン
        def custom_preprocessor(data):
            """データ前処理プラグイン"""
            return {**data, 'processed': True}
        
        def custom_postprocessor(results):
            """結果後処理プラグイン"""
            return [{'enhanced': True, **result} for result in results]
        
        plugins = {
            'preprocessor': custom_preprocessor,
            'postprocessor': custom_postprocessor
        }
        
        pipeline = LLMPipeline(plugins=plugins)
        
        # プラグインが登録されることを確認
        assert hasattr(pipeline, '_plugins')
        assert 'preprocessor' in pipeline._plugins
        assert 'postprocessor' in pipeline._plugins


class TestPipelineEdgeCases:
    """パイプラインエッジケース・境界値テスト"""

    def test_pipeline_config_edge_cases(self):
        """設定パラメータの境界値テスト"""
        from src.llm.pipeline import LLMPipeline
        
        # 境界値設定のテスト
        edge_configs = [
            {'processing': {'batch_size': 1}},      # 最小バッチサイズ
            {'processing': {'batch_size': 100}},    # 最大バッチサイズ
            {'processing': {'max_workers': 1}},     # 最小ワーカー数
            {'processing': {'max_workers': 20}},    # 最大ワーカー数
            {'processing': {'timeout': 1}},         # 最小タイムアウト
            {'processing': {'timeout': 600}},       # 最大タイムアウト
        ]
        
        for config in edge_configs:
            pipeline = LLMPipeline(config=config)
            
            # 境界値設定が適用されることを確認
            if 'batch_size' in config['processing']:
                assert pipeline._config['processing']['batch_size'] == config['processing']['batch_size']
            if 'max_workers' in config['processing']:
                assert pipeline._config['processing']['max_workers'] == config['processing']['max_workers']

    @patch('src.llm.client.LLMClient')
    def test_pipeline_resource_exhaustion_handling(self, mock_client_class):
        """リソース制約下での動作テスト"""
        import threading
        from src.llm.pipeline import LLMPipeline
        
        # メモリ不足をシミュレート
        mock_client = Mock()
        mock_client.batch_completion.side_effect = MemoryError("Insufficient memory")
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        large_dataset = [{'title': 'メモリテスト' * 1000} for _ in range(100)]
        
        # リソース不足エラーが適切に処理されることを確認
        with pytest.raises(MemoryError):
            pipeline.process_invention_batch(large_dataset)
        
        # 統計情報にエラーが記録されることを確認
        stats = pipeline.get_stats()
        assert stats['failed_processed'] > 0

    @patch('src.llm.client.LLMClient')
    def test_pipeline_malformed_data_handling(self, mock_client_class):
        """不正形式データの処理テスト"""
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.batch_completion.return_value = [{'content': 'テスト応答'}]
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        # 様々な不正データ
        malformed_data = [
            None,                           # None値
            {},                            # 空辞書
            {'title': None},               # None値フィールド
            {'title': ''},                 # 空文字列
            {'title': '正常', 'extra': object()},  # シリアライズ不可オブジェクト
            {'title': 'テスト' * 10000},    # 異常に長いデータ
        ]
        
        results = pipeline.process_invention_batch(malformed_data, continue_on_error=True)
        
        # 不正データが適切に処理またはスキップされることを確認
        assert len(results) == len(malformed_data)
        
        # 一部は成功、一部は失敗になることを確認
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'failed']
        
        assert len(successful) > 0  # 正常なデータは処理される
        assert len(failed) > 0      # 不正なデータはエラーになる

    @patch('src.llm.client.LLMClient')
    def test_progress_tracking_accuracy(self, mock_client_class):
        """進捗追跡の精度テスト"""
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.completion.return_value = {'content': 'テスト'}
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        progress_updates = []
        
        def precise_progress_callback(stage, current, total):
            percentage = (current / total) * 100
            progress_updates.append({
                'stage': stage,
                'current': current,
                'total': total,
                'percentage': percentage
            })
        
        test_data = [{'title': f'データ{i}'} for i in range(10)]
        
        results = pipeline.process_invention_batch(
            test_data, progress_callback=precise_progress_callback
        )
        
        # 進捗の数学的正確性を確認
        for update in progress_updates:
            expected_percentage = (update['current'] / update['total']) * 100
            assert abs(update['percentage'] - expected_percentage) < 0.01  # 浮動小数点誤差許容
            assert 0 <= update['percentage'] <= 100
            assert update['current'] <= update['total']

    @patch('src.llm.client.LLMClient')
    def test_pipeline_graceful_shutdown(self, mock_client_class):
        """処理中の正常シャットダウンテスト"""
        import threading
        import time
        from src.llm.pipeline import LLMPipeline
        
        # 処理を遅延させるモッククライアント
        mock_client = Mock()
        
        def slow_completion(messages, **kwargs):
            time.sleep(0.1)  # 遅延シミュレート
            return {'content': '遅延応答'}
        
        mock_client.completion = slow_completion
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        test_data = [{'title': f'長時間処理{i}'} for i in range(50)]
        
        # 別スレッドで処理開始
        result_holder = {}
        
        def run_processing():
            try:
                result_holder['results'] = pipeline.process_invention_batch(test_data)
            except Exception as e:
                result_holder['error'] = e
        
        processing_thread = threading.Thread(target=run_processing)
        processing_thread.start()
        
        # 少し待ってからシャットダウン要求
        time.sleep(0.2)
        pipeline.shutdown()
        
        # スレッドの完了を待つ
        processing_thread.join(timeout=5)
        
        # 正常にシャットダウンされたことを確認
        assert not processing_thread.is_alive()
        
        # 部分的な結果または適切なエラーが返されることを確認
        assert 'results' in result_holder or 'error' in result_holder


class TestPipelineMonitoring:
    """パイプライン監視・統計テスト"""

    @patch('src.llm.client.LLMClient')
    def test_pipeline_statistics_collection(self, mock_client_class):
        """パイプライン統計収集テスト"""
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.batch_completion.return_value = [
            {'content': '統計テスト結果'}
        ]
        mock_client.get_usage_stats.return_value = {
            'total_tokens': 1000,
            'total_requests': 5
        }
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline()
        
        # 処理実行
        test_data = [{'title': '統計テスト'}]
        results = pipeline.process_invention_batch(test_data)
        
        # 統計情報の確認
        stats = pipeline.get_detailed_stats()
        
        expected_stats = [
            'total_processed', 'successful_processed', 'failed_processed',
            'processing_time', 'token_usage', 'error_rates'
        ]
        
        for stat in expected_stats:
            assert stat in stats

    @patch('src.llm.client.LLMClient')
    def test_pipeline_real_time_monitoring(self, mock_client_class):
        """パイプラインリアルタイム監視テスト"""
        from src.llm.pipeline import LLMPipeline
        
        mock_client = Mock()
        mock_client.batch_completion.return_value = [
            {'content': '監視テスト'}
        ]
        mock_client_class.return_value = mock_client
        
        pipeline = LLMPipeline(enable_monitoring=True)
        
        monitoring_events = []
        
        def monitor_callback(event):
            monitoring_events.append(event)
        
        pipeline.set_monitor_callback(monitor_callback)
        
        # 監視対象処理の実行
        test_data = [{'title': '監視テスト'}]
        results = pipeline.process_invention_batch(test_data)
        
        # 監視イベントの確認
        assert len(monitoring_events) > 0
        
        event_types = [event['type'] for event in monitoring_events]
        assert 'batch_start' in event_types
        assert 'batch_complete' in event_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])