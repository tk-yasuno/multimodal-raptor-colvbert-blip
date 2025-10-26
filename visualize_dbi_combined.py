"""
特定のRAPTOR Treeを可視化するスクリプト
DBI戦略とCombined戦略のツリーを可視化
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize_raptor_tree import RAPTORTreeVisualizer
from pathlib import Path

# 可視化対象ファイル
tree_files = [
    "scaling_test_tree_2000chunks_20251025_211253.pkl",  # DBI戦略
    "scaling_test_tree_2000chunks_20251026_082623.pkl",  # Combined戦略
]

results_dir = Path(r"C:\Users\yasun\LangChain\learning-langchain\multimodal-raptor-colvbert-blip\data\encoder_comparison_46pdfs\results")
output_dir = Path(r"C:\Users\yasun\LangChain\learning-langchain\multimodal-raptor-colvbert-blip\data\encoder_comparison_46pdfs\raptor_trees")

print("=" * 80)
print("DBI & Combined戦略のツリー可視化")
print("=" * 80)

for tree_file in tree_files:
    tree_path = results_dir / tree_file
    
    if not tree_path.exists():
        print(f"⚠️  ファイルが見つかりません: {tree_file}")
        continue
    
    print(f"\n[処理中] {tree_file}")
    print("-" * 80)
    
    try:
        # 可視化
        visualizer = RAPTORTreeVisualizer(str(tree_path))
        visualizer.load_tree()
        visualizer.build_graph()
        
        # 出力ファイル名（拡張子を除去）
        base_name = tree_path.stem
        
        # ツリー可視化
        tree_output = output_dir / f"{base_name}_tree.png"
        visualizer.visualize_tree(str(tree_output))
        print(f"✅ ツリー保存: {tree_output}")
        
        # 統計グラフ
        stats_output = output_dir / f"{base_name}_stats.png"
        visualizer.create_statistics_plot(str(stats_output))
        print(f"✅ 統計保存: {stats_output}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("✅ 可視化完了")
print("=" * 80)
