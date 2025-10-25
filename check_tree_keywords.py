"""
生成されたツリーのノードとキーワードを確認するスクリプト
"""
import pickle
import sys
from pathlib import Path

# UTF-8出力を強制
sys.stdout.reconfigure(encoding='utf-8')

pkl_file = r"C:\Users\yasun\LangChain\learning-langchain\multimodal-raptor-colvbert-blip\data\encoder_comparison_46pdfs\results\scaling_test_tree_250chunks_20251025_144137.pkl"

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

tree = data['tree']

def show_cluster_info(tree_dict, depth=0, cluster_path=""):
    """クラスタ情報を表示"""
    if not isinstance(tree_dict, dict):
        return
    
    if 'clusters' in tree_dict and tree_dict['clusters']:
        clusters = tree_dict['clusters']
        
        for cluster_id, cluster_data in clusters.items():
            current_path = f"{cluster_path}C{cluster_id}"
            
            # サマリーテキスト取得
            summary_text = ""
            if 'summary' in cluster_data:
                summary = cluster_data['summary']
                if hasattr(summary, 'page_content'):
                    summary_text = summary.page_content
            
            print(f"\n{'  ' * depth}[Depth {depth}] {current_path}")
            print(f"{'  ' * depth}サマリー: {summary_text[:150]}...")
            
            # 子ツリーを再帰的に表示
            if 'children' in cluster_data and cluster_data['children']:
                show_cluster_info(cluster_data['children'], depth + 1, current_path + "-")

print("=" * 80)
print("250チャンク RAPTOR Tree - クラスタサマリー")
print("=" * 80)

show_cluster_info(tree)

print("\n" + "=" * 80)
