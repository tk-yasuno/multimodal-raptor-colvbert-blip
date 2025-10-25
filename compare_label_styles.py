"""
ラベルスタイルの変更を確認するスクリプト
"""
import pickle
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# UTF-8出力を強制
sys.stdout.reconfigure(encoding='utf-8')

pkl_file = r"C:\Users\yasun\LangChain\learning-langchain\multimodal-raptor-colvbert-blip\data\encoder_comparison_46pdfs\results\scaling_test_tree_250chunks_20251025_144137.pkl"

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

tree = data['tree']

# サマリーを収集
all_summaries = []
node_info = []

def collect_summaries(tree_dict, depth=0, cluster_path=""):
    """サマリーを収集"""
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
            
            all_summaries.append(summary_text if summary_text else "")
            node_info.append({
                'path': current_path,
                'depth': depth,
                'cluster_id': cluster_id,
                'text': summary_text[:80] if summary_text else ""
            })
            
            # 子ツリーを再帰的に処理
            if 'children' in cluster_data and cluster_data['children']:
                collect_summaries(cluster_data['children'], depth + 1, current_path + "-")

collect_summaries(tree)

print("=" * 80)
print("ラベルスタイル比較")
print("=" * 80)
print(f"総ノード数: {len(all_summaries)}\n")

# TF-IDFで重要キーワードを抽出
stop_words = {'こと', 'もの', 'ため', 'よう', 'ところ', 'これ', 'それ', 
             'あれ', 'この', 'その', 'あの', 'など', 'ほか', '以上',
             'について', 'における', 'に関する', 'による', 'によって',
             'として', 'とする', 'である', 'です', 'ます', 'した'}

vectorizer = TfidfVectorizer(
    token_pattern=r'[一-龥ぁ-んァ-ヶー]{2,}|[a-zA-Z]{3,}',
    max_features=1000,
    stop_words=None
)

tfidf_matrix = vectorizer.fit_transform(all_summaries)
feature_names = vectorizer.get_feature_names_out()

# 最初の5ノードで比較
print("最初の5ノードのラベル比較:\n")
for i, info in enumerate(node_info[:5]):
    tfidf_scores = tfidf_matrix[i].toarray()[0]
    top_indices = tfidf_scores.argsort()[::-1]
    
    # 上位キーワード
    keywords = []
    for idx in top_indices:
        if len(keywords) >= 3:
            break
        word = feature_names[idx]
        if word not in stop_words and tfidf_scores[idx] > 0:
            keywords.append(word)
    
    # 旧スタイル（カンマ区切り）
    old_style = ', '.join(keywords[:2]) if len(keywords) >= 2 else keywords[0] if keywords else f"C{info['cluster_id']}"
    
    # 新スタイル（改行区切り）
    if len(keywords) >= 2:
        new_style = f"{keywords[0]}\\n{keywords[1]}"
    elif keywords:
        new_style = keywords[0]
    else:
        new_style = f"C{info['cluster_id']}"
    
    print(f"[{info['path']}] (Depth {info['depth']})")
    print(f"  旧: {old_style}")
    print(f"  新: {new_style}")
    print(f"  サマリー: {info['text']}...")
    print()

print("=" * 80)
print("変更点:")
print("✅ カンマ区切り「災害, 復興」→ 改行区切り「災害\\n復興」")
print("✅ ノードサイズを少し大きく (リーフ: 300→500, 内部: 800-1200→1000-1300)")
print("✅ bboxパディングを縮小 (0.3→0.2)")
print("✅ フォントサイズ閾値を調整 (20ノード以下で8pt)")
print("=" * 80)
print("\n結果: よりコンパクトで読みやすいラベル表示")
