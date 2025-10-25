"""
ColModernVBERT (SigLIP) vs ColVBERT (BLIP) 大規模性能比較スクリプト
RAPTOR Tree構築を含む包括的評価 - 46 PDF文書版（チャンク分割版）

既存の画像ファイルを使用して:
1. テキストを500トークンチャンクに分割
2. 全チャンクでRAPTOR階層ツリーを構築
3. エンコーダー性能を比較
4. GPU使用量とTree構造を評価
"""

import os
import sys
import time
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
import subprocess

# 日本語フォント設定
matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visual_raptor_colbert import VisualRAPTORColBERT, VisualDocument

print("=" * 80)
print("ColModernVBERT (SigLIP) vs ColVBERT (BLIP) 大規模性能比較")
print("RAPTOR Tree構築 + 階層的検索評価（チャンク分割版）")
print("実際のPDFデータ使用 (46文書、2378ページ → チャンク分割)")
print("=" * 80)

# GPU情報取得関数
def get_gpu_memory_usage():
    """nvidia-smiを使用してGPUメモリ使用量を取得"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                values = lines[0].split(',')
                return {
                    'memory_used_mb': float(values[0].strip()),
                    'memory_total_mb': float(values[1].strip()),
                    'gpu_utilization': float(values[2].strip())
                }
    except Exception as e:
        print(f"  ⚠️ GPU情報取得エラー: {e}")
    return {'memory_used_mb': 0, 'memory_total_mb': 0, 'gpu_utilization': 0}

def count_tree_nodes(tree):
    """ツリーの統計情報を計算"""
    if not tree or not isinstance(tree, dict):
        return {'num_leaf_nodes': 0, 'num_internal_nodes': 0, 'total_nodes': 0, 'max_depth': 0}
    
    def count_recursive(node, depth=0):
        """再帰的にノードをカウント (leaf_count, internal_count, max_depth)"""
        if not node or not isinstance(node, dict):
            return (0, 0, depth)
        
        # クラスタがない場合はリーフノード
        clusters = node.get('clusters', {})
        if not clusters:
            return (1, 0, depth)
        
        # 内部ノードとして1つカウント
        total_leaf = 0
        total_internal = 1  # このノード自体
        max_child_depth = depth
        
        # 各クラスタの子ノードを再帰的にカウント
        for cluster_id, cluster_data in clusters.items():
            if isinstance(cluster_data, dict) and 'children' in cluster_data:
                leaf, internal, child_depth = count_recursive(cluster_data['children'], depth + 1)
                total_leaf += leaf
                total_internal += internal
                max_child_depth = max(max_child_depth, child_depth)
        
        return (total_leaf, total_internal, max_child_depth)
    
    leaf_count, internal_count, max_depth = count_recursive(tree, 0)
    
    return {
        'num_leaf_nodes': leaf_count,
        'num_internal_nodes': internal_count,
        'total_nodes': leaf_count + internal_count,
        'max_depth': max_depth
    }

# ステップ1: ディレクトリ準備
print("\n[ステップ 1/10] ディレクトリ準備...")
output_dir = Path("data/encoder_comparison_46pdfs")
results_dir = output_dir / "results"
trees_dir = output_dir / "raptor_trees"
images_dir = output_dir / "images"

for dir_path in [results_dir, trees_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ {dir_path} 準備完了")

# ステップ2: 既存の画像ファイルとテキストキャッシュを読み込み
print("\n[ステップ 2/10] 既存の画像ファイルとテキストキャッシュを読み込み中...")

if not images_dir.exists():
    print(f"❌ 画像ディレクトリが見つかりません: {images_dir}")
    sys.exit(1)

image_files = sorted(glob(str(images_dir / "*.png")))
print(f"✅ {len(image_files)}枚の画像を発見")

# テキストキャッシュを読み込み
cache_file = output_dir / "pdf_text_cache.json"
pdf_text_cache = {}
if cache_file.exists():
    with open(cache_file, 'r', encoding='utf-8') as f:
        pdf_text_cache = json.load(f)
    print(f"✅ テキストキャッシュを読み込み: {len(pdf_text_cache)}エントリ")
else:
    print(f"⚠️ テキストキャッシュが見つかりません: {cache_file}")

# ステップ3: VisualDocumentオブジェクトを作成してチャンクに分割
print(f"\n[ステップ 3/10] {len(image_files)}個のVisualDocumentを作成してチャンク分割中...")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# テキスト分割器の設定
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

visual_documents = []
total_chunks = 0

for img_path in image_files:
    img_path_obj = Path(img_path)
    cached_text = pdf_text_cache.get(str(img_path_obj), "")
    
    # テキストが十分にある場合はチャンクに分割
    if cached_text and len(cached_text.strip()) > 100:
        # チャンク分割
        chunks = text_splitter.split_text(cached_text)
        
        for chunk_idx, chunk_text in enumerate(chunks):
            # 各チャンクをVisualDocumentとして保存
            doc = VisualDocument(
                image_path=str(img_path_obj),
                metadata={
                    'source': img_path_obj.stem,
                    'page_number': img_path_obj.stem.split('_page')[-1].replace('.png', '') if '_page' in img_path_obj.stem else '1',
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                }
            )
            doc.cached_text = chunk_text
            visual_documents.append(doc)
        
        total_chunks += len(chunks)
    else:
        # テキストが短い場合はそのまま1つのドキュメントとして扱う
        doc = VisualDocument(
            image_path=str(img_path_obj),
            metadata={
                'source': img_path_obj.stem,
                'page_number': img_path_obj.stem.split('_page')[-1].replace('.png', '') if '_page' in img_path_obj.stem else '1',
                'chunk_index': 0,
                'total_chunks': 1
            }
        )
        doc.cached_text = cached_text if cached_text else ""
        visual_documents.append(doc)
        total_chunks += 1

print(f"✅ {len(image_files)}ページから{len(visual_documents)}個のチャンクを作成完了")
print(f"   平均チャンク数/ページ: {len(visual_documents)/len(image_files):.1f}")

# ステップ4: 検索クエリ生成
print("\n[ステップ 4/10] 検索クエリと関連性判定データを生成中...")

queries = [
    {
        'query_id': 'Q1',
        'query': '津波警報の発令状況と避難指示について',
        'relevant_docs': []
    },
    {
        'query_id': 'Q2',
        'query': '避難所の運営と物資配布の方法',
        'relevant_docs': []
    },
    {
        'query_id': 'Q3',
        'query': '災害時の通信手段と情報伝達',
        'relevant_docs': []
    },
    {
        'query_id': 'Q4',
        'query': '復旧・復興計画の策定プロセス',
        'relevant_docs': []
    },
    {
        'query_id': 'Q5',
        'query': 'ボランティア活動の調整と支援',
        'relevant_docs': []
    }
]

print(f"✅ {len(queries)}個のクエリ生成完了")

print("\n" + "=" * 80)
print("ColVBERT (BLIP) でRAPTOR Tree構築 + 評価")
print("=" * 80)

# ステップ5: ColVBERT (BLIP) でRAPTOR Tree構築
print(f"\n[ステップ 5/10] ColVBERT (BLIP) でRAPTOR Tree構築中...")
print(f"  対象チャンク数: {len(visual_documents)}")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# GPU並列処理でテキスト埋め込み（OllamaよりHTTP通信を回避して高速化）
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Using device for text embeddings: {device}")

embeddings_model = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 64}
)

# 日本語対応の汎用LLMを使用（要約タスクに適している）
llm = ChatOllama(
    model="qwen2.5:7b",  # granite-code:8b → qwen2.5:7b (日本語対応、汎用タスク向け)
    base_url="http://localhost:11434",
    temperature=0.3,  # 0.0 → 0.3 (やや創造的な要約を可能にする)
    num_ctx=8192  # コンテキストウィンドウを明示的に設定
)

colbert_config = {
    'encoder_type': 'colbert',
    'text_model': 'intfloat/multilingual-e5-large',  # Vision encoderが使用するHuggingFaceモデル
    'vision_model': 'Salesforce/blip-image-captioning-base',
    'embedding_dim': 768,
    'use_cross_attention': False
}

pdf_source_dir = Path("data/disaster_visual_documents")

# GPU使用量測定（開始前）
gpu_before_colbert = get_gpu_memory_usage()
print(f"GPU状態 (開始前): {gpu_before_colbert['memory_used_mb']:.0f}MB / {gpu_before_colbert['memory_total_mb']:.0f}MB")

colbert_system = VisualRAPTORColBERT(
    embeddings_model=embeddings_model,
    llm=llm,
    use_modern_vbert=False,
    colbert_config=colbert_config,
    pdf_source_dir=str(pdf_source_dir)
)

print("ColVBERT初期化完了 - RAPTOR Tree構築開始...")

# RAPTOR Treeを構築またはキャッシュから読み込み
colbert_tree = None
colbert_tree_build_time = 0
colbert_tree_stats = {'num_leaf_nodes': 0, 'num_internal_nodes': 0, 'total_nodes': 0, 'max_depth': 0}
colbert_tree_pickle = trees_dir / "colbert_blip_tree_46pdfs_chunked.pkl"

if colbert_tree_pickle.exists():
    print(f"📂 既存のColVBERT Treeを読み込み中: {colbert_tree_pickle}")
    try:
        import pickle
        with open(colbert_tree_pickle, 'rb') as f:
            tree_data = pickle.load(f)
            colbert_tree = tree_data['tree']
            colbert_tree_build_time = tree_data['build_time']
            colbert_tree_stats = tree_data['stats']
        print(f"✅ ColVBERT Tree読み込み完了")
        print(f"  構築時間 (前回): {colbert_tree_build_time:.2f}秒 ({colbert_tree_build_time/60:.1f}分)")
        print(f"  総ノード数: {colbert_tree_stats['total_nodes']}")
        print(f"  リーフノード: {colbert_tree_stats['num_leaf_nodes']}")
        print(f"  内部ノード: {colbert_tree_stats['num_internal_nodes']}")
        print(f"  最大深度: {colbert_tree_stats['max_depth']}")
        # Treeをシステムに設定
        colbert_system.tree = colbert_tree
    except Exception as e:
        print(f"⚠️ Tree読み込みエラー: {e}")
        print("新規にTreeを構築します...")
        colbert_tree = None

if colbert_tree is None:
    print("🌳 新規にRAPTOR Treeを構築中...")
    colbert_tree_start_time = time.time()
    colbert_gpu_during_tree = []

    try:
        colbert_tree = colbert_system.build_tree(visual_documents)
        
        # 定期的にGPU使用量を記録
        for i in range(5):
            time.sleep(0.5)
            colbert_gpu_during_tree.append(get_gpu_memory_usage())
        
        colbert_tree_build_time = time.time() - colbert_tree_start_time
        
        # ツリー統計を取得
        colbert_tree_stats = count_tree_nodes(colbert_tree)
        
        print(f"✅ ColVBERT RAPTOR Tree構築完了")
        print(f"  構築時間: {colbert_tree_build_time:.2f}秒 ({colbert_tree_build_time/60:.1f}分)")
        print(f"  総ノード数: {colbert_tree_stats['total_nodes']}")
        print(f"  リーフノード: {colbert_tree_stats['num_leaf_nodes']}")
        print(f"  内部ノード: {colbert_tree_stats['num_internal_nodes']}")
        print(f"  最大深度: {colbert_tree_stats['max_depth']}")
        
        # ツリーをpickleとして保存
        try:
            import pickle
            with open(colbert_tree_pickle, 'wb') as f:
                pickle.dump({
                    'tree': colbert_tree,
                    'build_time': colbert_tree_build_time,
                    'stats': colbert_tree_stats
                }, f)
            print(f"  ツリーキャッシュ保存: {colbert_tree_pickle}")
        except Exception as e:
            print(f"  ⚠️ ツリーキャッシュ保存エラー: {e}")
        
        # ツリー統計をJSONとして保存
        try:
            colbert_tree_file = trees_dir / "colbert_blip_tree_46pdfs_chunked.json"
            with open(colbert_tree_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'build_time': colbert_tree_build_time,
                    'stats': colbert_tree_stats,
                    'num_chunks': len(visual_documents),
                    'num_pages': len(image_files),
                    'chunk_size': CHUNK_SIZE,
                    'chunk_overlap': CHUNK_OVERLAP,
                    'note': 'Tree structure saved (Document objects not serializable)'
                }, f, indent=2, ensure_ascii=False)
            print(f"  ツリー統計保存: {colbert_tree_file}")
        except Exception as e:
            print(f"  ⚠️ ツリー統計保存エラー: {e}")
        
    except Exception as e:
        print(f"❌ ColVBERT RAPTOR Tree構築エラー: {e}")
        import traceback
        traceback.print_exc()
else:
    # キャッシュから読み込んだ場合はGPU測定をスキップ
    colbert_gpu_during_tree = []

# GPU使用量測定（完了後）
gpu_after_colbert_tree = get_gpu_memory_usage()
colbert_tree_gpu_peak = max([g['memory_used_mb'] for g in colbert_gpu_during_tree]) if colbert_gpu_during_tree else gpu_after_colbert_tree['memory_used_mb']

# ステップ6: ColVBERT (BLIP) で階層的検索評価
print(f"\n[ステップ 6/10] ColVBERT (BLIP) で階層的検索評価中...")

colbert_search_times = []
colbert_search_results = []

if colbert_tree:
    for query in queries:
        search_start = time.perf_counter()
        try:
            # 階層的検索実行
            results = colbert_system.retrieve(
                query['query'],
                tree_traversal='collapsed',
                top_k=10
            )
            search_time = time.perf_counter() - search_start
            colbert_search_times.append(search_time)
            colbert_search_results.append(results)
            
            print(f"  クエリ '{query['query_id']}': {search_time*1000:.2f}ms, {len(results)}件取得")
        except Exception as e:
            print(f"  ⚠️ クエリ '{query['query_id']}' 検索エラー: {e}")
            colbert_search_times.append(0)
            colbert_search_results.append([])

avg_search_time = np.mean(colbert_search_times) if colbert_search_times else 0
print(f"✅ ColVBERT 階層的検索完了")
print(f"  平均検索時間: {avg_search_time*1000:.2f}ms")

# メモリクリア
del colbert_system
torch.cuda.empty_cache()

print("\n" + "=" * 80)
print("ColModernVBERT (SigLIP) でRAPTOR Tree構築 + 評価")
print("=" * 80)

# ステップ7: ColModernVBERT (SigLIP) でRAPTOR Tree構築
print(f"\n[ステップ 7/10] ColModernVBERT (SigLIP) でRAPTOR Tree構築中...")
print(f"  対象チャンク数: {len(visual_documents)}")

modern_config = {
    'encoder_type': 'modern',
    'text_model': 'google/siglip-base-patch16-224',  # SigLIPのテキストエンコーダー
    'vision_model': 'google/siglip-base-patch16-224',
    'embedding_dim': 768,
    'use_cross_attention': True
}

# GPU使用量測定（開始前）
gpu_before_modern = get_gpu_memory_usage()
print(f"GPU状態 (開始前): {gpu_before_modern['memory_used_mb']:.0f}MB / {gpu_before_modern['memory_total_mb']:.0f}MB")

modern_system = VisualRAPTORColBERT(
    embeddings_model=embeddings_model,
    llm=llm,
    use_modern_vbert=True,
    colbert_config=modern_config,
    pdf_source_dir=str(pdf_source_dir)
)

print("ColModernVBERT初期化完了 - RAPTOR Tree構築開始...")

# RAPTOR Treeを構築またはキャッシュから読み込み
modern_tree = None
modern_tree_build_time = 0
modern_tree_stats = {'num_leaf_nodes': 0, 'num_internal_nodes': 0, 'total_nodes': 0, 'max_depth': 0}
modern_tree_pickle = trees_dir / "colmodern_siglip_tree_46pdfs_chunked.pkl"

if modern_tree_pickle.exists():
    print(f"📂 既存のColModernVBERT Treeを読み込み中: {modern_tree_pickle}")
    try:
        import pickle
        with open(modern_tree_pickle, 'rb') as f:
            tree_data = pickle.load(f)
            modern_tree = tree_data['tree']
            modern_tree_build_time = tree_data['build_time']
            modern_tree_stats = tree_data['stats']
        print(f"✅ ColModernVBERT Tree読み込み完了")
        print(f"  構築時間 (前回): {modern_tree_build_time:.2f}秒 ({modern_tree_build_time/60:.1f}分)")
        print(f"  総ノード数: {modern_tree_stats['total_nodes']}")
        print(f"  リーフノード: {modern_tree_stats['num_leaf_nodes']}")
        print(f"  内部ノード: {modern_tree_stats['num_internal_nodes']}")
        print(f"  最大深度: {modern_tree_stats['max_depth']}")
        # Treeをシステムに設定
        modern_system.tree = modern_tree
    except Exception as e:
        print(f"⚠️ Tree読み込みエラー: {e}")
        print("新規にTreeを構築します...")
        modern_tree = None

if modern_tree is None:
    print("🌳 新規にRAPTOR Treeを構築中...")
    modern_tree_start_time = time.time()
    modern_gpu_during_tree = []

    try:
        modern_tree = modern_system.build_tree(visual_documents)
        
        # 定期的にGPU使用量を記録
        for i in range(5):
            time.sleep(0.5)
            modern_gpu_during_tree.append(get_gpu_memory_usage())
        
        modern_tree_build_time = time.time() - modern_tree_start_time
        
        # ツリー統計を取得
        modern_tree_stats = count_tree_nodes(modern_tree)
        
        print(f"✅ ColModernVBERT RAPTOR Tree構築完了")
        print(f"  構築時間: {modern_tree_build_time:.2f}秒 ({modern_tree_build_time/60:.1f}分)")
        print(f"  総ノード数: {modern_tree_stats['total_nodes']}")
        print(f"  リーフノード: {modern_tree_stats['num_leaf_nodes']}")
        print(f"  内部ノード: {modern_tree_stats['num_internal_nodes']}")
        print(f"  最大深度: {modern_tree_stats['max_depth']}")
        
        # ツリーをpickleとして保存
        try:
            import pickle
            with open(modern_tree_pickle, 'wb') as f:
                pickle.dump({
                    'tree': modern_tree,
                    'build_time': modern_tree_build_time,
                    'stats': modern_tree_stats
                }, f)
            print(f"  ツリーキャッシュ保存: {modern_tree_pickle}")
        except Exception as e:
            print(f"  ⚠️ ツリーキャッシュ保存エラー: {e}")
        
        # ツリー統計をJSONとして保存
        try:
            modern_tree_file = trees_dir / "colmodern_siglip_tree_46pdfs_chunked.json"
            with open(modern_tree_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'build_time': modern_tree_build_time,
                    'stats': modern_tree_stats,
                    'num_chunks': len(visual_documents),
                    'num_pages': len(image_files),
                    'chunk_size': CHUNK_SIZE,
                    'chunk_overlap': CHUNK_OVERLAP,
                    'note': 'Tree structure saved (Document objects not serializable)'
                }, f, indent=2, ensure_ascii=False)
            print(f"  ツリー統計保存: {modern_tree_file}")
        except Exception as e:
            print(f"  ⚠️ ツリー統計保存エラー: {e}")
        
    except Exception as e:
        print(f"❌ ColModernVBERT RAPTOR Tree構築エラー: {e}")
        import traceback
        traceback.print_exc()
else:
    # キャッシュから読み込んだ場合はGPU測定をスキップ
    modern_gpu_during_tree = []

# GPU使用量測定（完了後）
gpu_after_modern_tree = get_gpu_memory_usage()
modern_tree_gpu_peak = max([g['memory_used_mb'] for g in modern_gpu_during_tree]) if modern_gpu_during_tree else gpu_after_modern_tree['memory_used_mb']

# ステップ8: ColModernVBERT (SigLIP) で階層的検索評価
print(f"\n[ステップ 8/10] ColModernVBERT (SigLIP) で階層的検索評価中...")

modern_search_times = []
modern_search_results = []

if modern_tree:
    for query in queries:
        search_start = time.perf_counter()
        try:
            results = modern_system.retrieve(
                query['query'],
                tree_traversal='collapsed',
                top_k=10
            )
            search_time = time.perf_counter() - search_start
            modern_search_times.append(search_time)
            modern_search_results.append(results)
            
            print(f"  クエリ '{query['query_id']}': {search_time*1000:.2f}ms, {len(results)}件取得")
        except Exception as e:
            print(f"  ⚠️ クエリ '{query['query_id']}' 検索エラー: {e}")
            modern_search_times.append(0)
            modern_search_results.append([])

avg_modern_search_time = np.mean(modern_search_times) if modern_search_times else 0
print(f"✅ ColModernVBERT 階層的検索完了")
print(f"  平均検索時間: {avg_modern_search_time*1000:.2f}ms")

# ステップ9: 比較結果を保存
print("\n[ステップ 9/10] 比較結果を保存中...")

comparison_results = {
    'timestamp': datetime.now().isoformat(),
    'num_chunks': len(visual_documents),
    'num_pages': len(image_files),
    'chunk_size': CHUNK_SIZE,
    'chunk_overlap': CHUNK_OVERLAP,
    'avg_chunks_per_page': len(visual_documents) / len(image_files),
    'num_queries': len(queries),
    'pdf_source': str(pdf_source_dir),
    'colbert_blip': {
        'encoder_type': 'ColVBERT (BLIP)',
        'tree_build_time': colbert_tree_build_time,
        'tree_stats': colbert_tree_stats,
        'gpu_peak_memory_tree_mb': colbert_tree_gpu_peak,
        'avg_search_time_ms': float(np.mean(colbert_search_times) * 1000) if colbert_search_times else 0,
        'median_search_time_ms': float(np.median(colbert_search_times) * 1000) if colbert_search_times else 0,
        'tree_file': str(trees_dir / "colbert_blip_tree_46pdfs_chunked.json")
    },
    'colmodern_vbert_siglip': {
        'encoder_type': 'ColModernVBERT (SigLIP)',
        'tree_build_time': modern_tree_build_time,
        'tree_stats': modern_tree_stats,
        'gpu_peak_memory_tree_mb': modern_tree_gpu_peak,
        'avg_search_time_ms': float(np.mean(modern_search_times) * 1000) if modern_search_times else 0,
        'median_search_time_ms': float(np.median(modern_search_times) * 1000) if modern_search_times else 0,
        'tree_file': str(trees_dir / "colmodern_siglip_tree_46pdfs_chunked.json")
    },
    'comparison': {
        'tree_build_speedup': colbert_tree_build_time / modern_tree_build_time if modern_tree_build_time > 0 else 0,
        'search_speedup': avg_search_time / avg_modern_search_time if avg_modern_search_time > 0 else 0,
        'tree_size_difference': modern_tree_stats['total_nodes'] - colbert_tree_stats['total_nodes'],
        'gpu_memory_difference_mb': modern_tree_gpu_peak - colbert_tree_gpu_peak
    }
}

results_file = results_dir / "raptor_comparison_results_46pdfs_chunked.json"
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(comparison_results, f, indent=2, ensure_ascii=False)

print(f"✅ 結果を {results_file} に保存")

# ステップ10: サマリー出力
print("\n[ステップ 10/10] サマリー出力...")

print("\n" + "=" * 80)
print("📊 RAPTOR Tree構築 + 階層的検索 性能比較サマリー（46 PDF、チャンク分割版）")
print("=" * 80)

print(f"\n【データセット】")
print(f"ページ数: {len(image_files)}")
print(f"チャンク数: {len(visual_documents)}")
print(f"チャンクサイズ: {CHUNK_SIZE}トークン")
print(f"平均チャンク数/ページ: {len(visual_documents)/len(image_files):.1f}")

print(f"\n【RAPTOR Tree構築】")
print(f"ColVBERT (BLIP):")
print(f"  - 構築時間: {colbert_tree_build_time:.2f}秒 ({colbert_tree_build_time/60:.1f}分)")
print(f"  - 総ノード数: {colbert_tree_stats['total_nodes']}")
print(f"  - リーフノード: {colbert_tree_stats['num_leaf_nodes']}")
print(f"  - 内部ノード: {colbert_tree_stats['num_internal_nodes']}")
print(f"  - 最大深度: {colbert_tree_stats['max_depth']}")

print(f"\nColModernVBERT (SigLIP):")
print(f"  - 構築時間: {modern_tree_build_time:.2f}秒 ({modern_tree_build_time/60:.1f}分)")
print(f"  - 総ノード数: {modern_tree_stats['total_nodes']}")
print(f"  - リーフノード: {modern_tree_stats['num_leaf_nodes']}")
print(f"  - 内部ノード: {modern_tree_stats['num_internal_nodes']}")
print(f"  - 最大深度: {modern_tree_stats['max_depth']}")

if modern_tree_build_time > 0:
    speedup = colbert_tree_build_time / modern_tree_build_time
    time_saved = colbert_tree_build_time - modern_tree_build_time
    print(f"\n⚡ Tree構築高速化率: {speedup:.2f}x")
    print(f"⚡ 時間短縮: {time_saved:.1f}秒 ({time_saved/60:.1f}分)")

print(f"\n【階層的検索性能】")
if colbert_search_times:
    print(f"ColVBERT (BLIP):")
    print(f"  - 平均検索時間: {np.mean(colbert_search_times)*1000:.2f}ms")
    print(f"  - 中央値: {np.median(colbert_search_times)*1000:.2f}ms")

if modern_search_times:
    print(f"\nColModernVBERT (SigLIP):")
    print(f"  - 平均検索時間: {np.mean(modern_search_times)*1000:.2f}ms")
    print(f"  - 中央値: {np.median(modern_search_times)*1000:.2f}ms")

print(f"\n【GPU使用量】")
print(f"ColVBERT (BLIP) ピークメモリ: {colbert_tree_gpu_peak:.0f}MB")
print(f"ColModernVBERT (SigLIP) ピークメモリ: {modern_tree_gpu_peak:.0f}MB")
if modern_tree_gpu_peak > 0:
    gpu_diff = modern_tree_gpu_peak - colbert_tree_gpu_peak
    gpu_diff_percent = (gpu_diff / colbert_tree_gpu_peak) * 100 if colbert_tree_gpu_peak > 0 else 0
    print(f"差分: {gpu_diff:.0f}MB ({gpu_diff_percent:+.1f}%)")

print("\n" + "=" * 80)
print("✅ RAPTOR Tree構築 + 階層的検索 性能比較完了（チャンク分割版）!")
print(f"📁 結果ディレクトリ: {results_dir}")
print(f"📄 詳細結果: {results_file}")
print(f"🌳 RAPTORツリー: {trees_dir}")
print("=" * 80)
