"""
pklファイルの構造を調べるデバッグスクリプト
"""
import pickle
from pathlib import Path

pkl_file = r"C:\Users\yasun\LangChain\learning-langchain\multimodal-raptor-colvbert-blip\data\encoder_comparison_46pdfs\results\scaling_test_tree_250chunks_20251025_144137.pkl"

print(f"ファイル調査: {Path(pkl_file).name}")
print("=" * 80)

with open(pkl_file, 'rb') as f:
    tree = pickle.load(f)

print(f"型: {type(tree)}")
print(f"クラス名: {tree.__class__.__name__ if hasattr(tree, '__class__') else 'N/A'}")
print()

# 属性リスト
print("属性一覧:")
attrs = [attr for attr in dir(tree) if not attr.startswith('_')]
for attr in attrs[:30]:  # 最初の30個
    print(f"  - {attr}")
print()

# 重要そうな属性を調査
important_attrs = ['all_nodes', 'tree', 'root', 'nodes', 'retrieval_tree', 'layer_to_nodes']
print("重要な属性の検査:")
for attr in important_attrs:
    if hasattr(tree, attr):
        val = getattr(tree, attr)
        print(f"  {attr}: {type(val)}")
        if isinstance(val, (list, dict)):
            print(f"    長さ/キー数: {len(val)}")
            if isinstance(val, list) and len(val) > 0:
                print(f"    最初の要素型: {type(val[0])}")
            if isinstance(val, dict) and len(val) > 0:
                print(f"    キー: {list(val.keys())[:5]}")
    else:
        print(f"  {attr}: なし")

print("\n" + "=" * 80)

# 辞書の場合、キーを表示
if isinstance(tree, dict):
    print("辞書のキー:")
    for key in tree.keys():
        print(f"  - {key}: {type(tree[key])}")
        if isinstance(tree[key], (list, dict)):
            print(f"    長さ: {len(tree[key])}")
    
    # treeキーの詳細
    if 'tree' in tree:
        print("\ntree詳細:")
        t = tree['tree']
        print(f"  型: {type(t)}")
        print(f"  クラス名: {t.__class__.__name__ if hasattr(t, '__class__') else 'N/A'}")
        
        # 辞書構造を詳しく調査
        if isinstance(t, dict):
            print(f"  キー: {list(t.keys())}")
            
            if 'clusters' in t:
                clusters = t['clusters']
                print(f"\n  clusters: {type(clusters)}, 長さ={len(clusters)}")
                for cid, cdata in list(clusters.items())[:2]:  # 最初の2クラスタ
                    print(f"    Cluster {cid}:")
                    print(f"      キー: {list(cdata.keys())}")
                    if 'summary' in cdata:
                        print(f"      summary: {type(cdata['summary'])}")
                    if 'documents' in cdata:
                        print(f"      documents: {len(cdata['documents'])}個")
                    if 'children' in cdata:
                        child = cdata['children']
                        print(f"      children: {type(child)}")
                        if isinstance(child, dict):
                            print(f"        children keys: {list(child.keys())}")
                            if 'clusters' in child:
                                print(f"        children clusters: {len(child['clusters'])}個")
