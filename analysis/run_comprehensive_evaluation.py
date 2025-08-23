#!/usr/bin/env python3
"""
注目特許仕分けくん 包括的評価実行スクリプト
すべての分析・レポート生成を一括実行
"""

import sys
from pathlib import Path
from datetime import datetime
from calculate_metrics import main as calculate_metrics
from generate_html_report import main as generate_html_report
from generate_markdown_report import main as generate_markdown_report


def main():
    """包括的評価の実行"""
    
    print("=" * 60)
    print("注目特許仕分けくん 包括的性能評価")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. 性能指標計算
        print("1. 性能指標計算を実行中...")
        print("-" * 40)
        metrics_result = calculate_metrics()
        print("✓ 性能指標計算完了")
        print()
        
        # 2. HTMLレポート生成
        print("2. HTMLレポート生成を実行中...")
        print("-" * 40)
        html_path = generate_html_report()
        print("✓ HTMLレポート生成完了")
        print()
        
        # 3. Markdownレポート生成  
        print("3. Markdownレポート生成を実行中...")
        print("-" * 40)
        md_path = generate_markdown_report()
        print("✓ Markdownレポート生成完了")
        print()
        
        # 4. 結果サマリー表示
        print("=" * 60)
        print("評価結果サマリー")
        print("=" * 60)
        
        # 主要指標表示
        if metrics_result:
            metrics = metrics_result.get('metrics', {})
            success = metrics_result.get('success_criteria_met', {})
            
            print(f"📊 主要性能指標:")
            print(f"   総合精度:     {metrics.get('accuracy', 0):.1%}")
            print(f"   HIT検出精度:  {metrics.get('precision', 0):.1%}")
            print(f"   HIT検出再現率: {metrics.get('recall', 0):.1%}")
            print(f"   F1スコア:     {metrics.get('f1_score', 0):.3f}")
            print(f"   ROC AUC:     {metrics.get('roc_auc', 0):.3f}")
            print()
            
            print(f"✅ 成功基準達成状況:")
            print(f"   総合精度:     {'達成' if success.get('accuracy') else '未達'}")
            print(f"   HIT再現率:    {'達成' if success.get('recall') else '未達'}")
            print(f"   HIT精度:      {'達成' if success.get('precision') else '未達'}")
            print(f"   総合判定:     {'成功' if success.get('overall') else '要改善'}")
            print()
        
        # 生成ファイル情報
        print(f"📁 生成ファイル:")
        print(f"   HTMLレポート: {Path(html_path).name}")
        print(f"   Markdownレポート: {Path(md_path).name}")
        print(f"   メトリクス: analysis/metrics_summary.json")
        print()
        
        # 推奨事項
        print(f"💡 主要推奨事項:")
        if metrics_result and metrics_result.get('success_criteria_met', {}).get('overall'):
            print("   - システムは実用水準に到達")
            print("   - 本格運用の準備を推奨") 
            print("   - 偽陰性2件の詳細分析実施")
            print("   - 段階的展開計画の策定")
        else:
            print("   - 成功基準未達のため改善が必要")
            print("   - 特に再現率と精度の向上に注力")
            print("   - プロンプトエンジニアリングの見直し")
        print()
        
        print("=" * 60)
        print("✅ すべての評価処理が正常に完了しました")
        print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        return {
            'html_report': html_path,
            'markdown_report': md_path,
            'metrics_summary': metrics_result
        }
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print(f"詳細: {type(e).__name__}")
        return None


if __name__ == "__main__":
    result = main()
    
    if result:
        print(f"\\n📋 3ポイント要約:")
        print(f"1. システム性能: 総合精度{result['metrics_summary']['metrics']['accuracy']:.1%}を達成")
        print(f"2. 成功基準: {'クリア' if result['metrics_summary']['success_criteria_met']['overall'] else '要改善'}")
        print(f"3. 次のステップ: {'本格運用準備' if result['metrics_summary']['success_criteria_met']['overall'] else 'システム改善'}")
        
        print(f"\\n📄 生成レポート:")
        print(f"HTML: {result['html_report']}")  
        print(f"Markdown: {result['markdown_report']}")
        
        sys.exit(0)
    else:
        sys.exit(1)