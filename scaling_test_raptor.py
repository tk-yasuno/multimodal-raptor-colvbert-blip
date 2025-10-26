"""
RAPTOR Tree構築のスケーリングテスト
段階的にチャンク数を増やして性能と品質を評価

テスト段階: 4250チャンク（全データ）からサンプリング4ケース
1)  250チャンク
2)  500チャンク  
3) 1000チャンク
4) 2000チャンク（Note. time up exit）

評価指標:
- 構築時間（埋め込み生成、クラスタリング、要約）
- GPU メモリ使用量
- ツリー構造（深さ、ノード数）
- クラスタリング品質（Silhouette Score等）
"""

import os
import sys
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
import subprocess

# UTF-8出力設定（Windows対応）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 日本語フォント設定
matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# プロジェクトルート追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '0_base_tsunami-lesson-rag'))

from visual_raptor_colbert import VisualRAPTORColBERT, VisualDocument
from PIL import Image

print("=" * 80)
print("RAPTOR Tree構築 スケーリングテスト")
print("ColVBERT (BLIP) エンコーダー使用")
print("=" * 80)

# GPU情報取得
def get_gpu_memory_usage():
    """GPUメモリ使用量を取得"""
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
        if not node or not isinstance(node, dict):
            return (0, 0, depth)
        
        clusters = node.get('clusters', {})
        if not clusters:
            return (1, 0, depth)
        
        total_leaf = 0
        total_internal = 1
        max_child_depth = depth
        
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

# ディレクトリ設定
data_dir = Path("data/encoder_comparison_46pdfs")
images_dir = data_dir / "images"
cache_file = data_dir / "pdf_text_cache.json"
results_dir = data_dir / "results"
results_dir.mkdir(exist_ok=True, parents=True)

print(f"\n[ステップ 1/6] データ読み込み中...")
print(f"  画像ディレクトリ: {images_dir}")
print(f"  テキストキャッシュ: {cache_file}")

# 画像ファイルとテキストキャッシュ読み込み
image_files = sorted(list(images_dir.glob("*.png")))
print(f"✅ {len(image_files)}枚の画像を発見")

with open(cache_file, 'r', encoding='utf-8') as f:
    text_cache = json.load(f)
print(f"✅ テキストキャッシュを読み込み: {len(text_cache)}エントリ")

# VisualDocumentを作成してチャンク分割
print(f"\n[ステップ 2/6] VisualDocumentを作成してチャンク分割中...")
visual_documents = []
chunk_size = 500
chunk_overlap = 50

total_pages_with_text = 0
total_chunks_created = 0

for img_file in image_files:
    # キャッシュのキーは相対パス形式
    page_key_short = img_file.stem  # ファイル名のみ
    page_key_full = str(img_file)   # フルパス
    
    # 両方の形式で試す
    cached_text = None
    if page_key_full in text_cache:
        cached_text = text_cache[page_key_full]
    elif page_key_short in text_cache:
        cached_text = text_cache[page_key_short]
    
    if cached_text:
        total_pages_with_text += 1
        
        # 空テキストをスキップ
        if not cached_text or len(cached_text.strip()) == 0:
            continue
        
        # チャンク分割
        text_length = len(cached_text)
        if text_length <= chunk_size:
            chunks = [cached_text]
        else:
            chunks = []
            start = 0
            while start < text_length:
                end = min(start + chunk_size, text_length)
                chunk_text = cached_text[start:end]
                chunks.append(chunk_text)
                start += chunk_size - chunk_overlap
        
        # 各チャンクをVisualDocumentとして作成
        for chunk_idx, chunk_text in enumerate(chunks):
            visual_doc = VisualDocument(
                image_path=str(img_file),
                text_content=chunk_text,
                metadata={
                    'source': page_key_short,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                }
            )
            visual_documents.append(visual_doc)
            total_chunks_created += 1

print(f"✅ {len(image_files)}ページ中{total_pages_with_text}ページにテキストあり")
print(f"✅ {total_chunks_created}個のチャンクを作成完了")
print(f"   平均チャンク数/ページ: {total_chunks_created/max(total_pages_with_text, 1):.1f}")

# チャンク数チェック
if len(visual_documents) == 0:
    print("\n❌ エラー: チャンクが作成されませんでした")
    print("  原因: テキストキャッシュにデータがないか、全て空のテキストです")
    sys.exit(1)

# サンプリングサイズ（実際のデータ量を超えないように調整）
max_chunks = len(visual_documents)
# sample_sizes = []
# for size in [250, 500, 1000, 2000]:
#     if size <= max_chunks:
#         sample_sizes.append(size)
# 
# # 重複を削除してソート
# sample_sizes = sorted(list(set(sample_sizes)))

# 2000チャンクのみ実行（combined戦略 vs silhouette戦略の比較）
sample_sizes = [2000] if 2000 <= max_chunks else [max_chunks]

print(f"\n実際のチャンク数: {max_chunks}")
print(f"調整後のテストサイズ: {sample_sizes}")

results = []

print(f"\n[ステップ 3/6] スケーリングテスト開始")
print(f"  テストサイズ: {sample_sizes}")
print("=" * 80)

# 各サンプルサイズでテスト
for test_num, sample_size in enumerate(sample_sizes, 1):
    print(f"\n{'='*80}")
    print(f"テスト {test_num}/4: {sample_size}チャンク")
    print(f"{'='*80}")
    
    # このテスト用のタイムスタンプ
    test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # サンプリング（再現性のためseed固定）
    random.seed(42)
    if sample_size < len(visual_documents):
        sampled_docs = random.sample(visual_documents, sample_size)
        print(f"  ランダムサンプリング: {sample_size}/{len(visual_documents)}チャンク")
    else:
        sampled_docs = visual_documents
        print(f"  全データ使用: {sample_size}チャンク")
    
    # ColVBERTシステム初期化
    print(f"\n[初期化] ColVBERT (BLIP) システム...")
    init_start = time.time()
    
    # Embeddings と LLM の作成
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 64}
    )
    
    llm = ChatOllama(
        model="gpt-oss:20b",
        base_url="http://localhost:11434",
        temperature=0.3,
        num_ctx=100000
    )
    
    colbert_config = {
        'encoder_type': 'colbert',
        'text_model': 'intfloat/multilingual-e5-large',
        'vision_model': 'Salesforce/blip-image-captioning-base',
        'embedding_dim': 768,
        'use_cross_attention': False
    }
    
    # クラスタリング評価指標設定
    # DBI戦略: クラスタ分離度を最大化（k=2バイアスを軽減する試み）
    colbert_system = VisualRAPTORColBERT(
        embeddings_model=embeddings_model,
        llm=llm,
        use_modern_vbert=False,
        colbert_config=colbert_config,
        use_multimodal=True,
        multimodal_weight=0.3,
        selection_strategy='combined',  # 複数指標の組み合わせ
        metric_weights={
            'silhouette': 0.5,  # クラスタ品質（ミクロ視点）- バランス
            'dbi': 0.5,         # クラスタ分離度（マクロ視点）- バランス
            'chi': 0.0          # CHI除外（k=2バイアス回避）
        }
    )
    
    init_time = time.time() - init_start
    print(f"✅ 初期化完了 ({init_time:.1f}秒)")
    
    # GPU状態記録
    gpu_before = get_gpu_memory_usage()
    print(f"GPU状態 (開始前): {gpu_before['memory_used_mb']:.0f}MB / {gpu_before['memory_total_mb']:.0f}MB")
    
    # ログファイル設定
    log_file = results_dir / f"scaling_test_log_{sample_size}chunks_{test_timestamp}.txt"
    
    # ツリー構築（ログをファイルにも出力）
    print(f"\n[構築] RAPTOR Tree構築中...")
    print(f"  ログファイル: {log_file.name}")
    build_start = time.time()
    
    # 標準出力をログファイルにも書き込む
    import io
    log_buffer = io.StringIO()
    
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    try:
        # ログファイルオープン
        with open(log_file, 'w', encoding='utf-8') as log_f:
            # ヘッダー情報
            log_f.write("="*80 + "\n")
            log_f.write(f"RAPTOR Tree構築ログ - {sample_size}チャンク\n")
            log_f.write(f"タイムスタンプ: {test_timestamp}\n")
            log_f.write("="*80 + "\n\n")
            log_f.write(f"テスト {test_num}/4: {sample_size}チャンク\n")
            log_f.write(f"サンプリング方法: {'ランダムサンプリング' if sample_size < len(visual_documents) else '全データ'}\n")
            log_f.write(f"GPU状態 (開始前): {gpu_before['memory_used_mb']:.0f}MB / {gpu_before['memory_total_mb']:.0f}MB\n")
            log_f.write(f"初期化時間: {init_time:.1f}秒\n")
            log_f.write("\n" + "-"*80 + "\n")
            log_f.write("ツリー構築開始\n")
            log_f.write("-"*80 + "\n\n")
            log_f.flush()
            
            # 元の標準出力を保存
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            # 標準出力をTeeOutputに切り替え（コンソールとログファイル両方に出力）
            sys.stdout = TeeOutput(original_stdout, log_f)
            sys.stderr = TeeOutput(original_stderr, log_f)
            
            try:
                tree = colbert_system.build_tree(sampled_docs)
                build_time = time.time() - build_start
            finally:
                # 標準出力を元に戻す
                sys.stdout = original_stdout
                sys.stderr = original_stderr
            
            # 構築完了情報をログに記録
            log_f.write("\n" + "-"*80 + "\n")
            log_f.write("ツリー構築完了\n")
            log_f.write("-"*80 + "\n")
        
        # GPU状態記録
        gpu_after = get_gpu_memory_usage()
        gpu_peak = max(gpu_before['memory_used_mb'], gpu_after['memory_used_mb'])
        
        # ツリー統計
        tree_stats = count_tree_nodes(tree)
        
        # 結果記録
        result = {
            'test_number': test_num,
            'sample_size': sample_size,
            'init_time': init_time,
            'build_time': build_time,
            'total_time': init_time + build_time,
            'gpu_memory_before_mb': gpu_before['memory_used_mb'],
            'gpu_memory_after_mb': gpu_after['memory_used_mb'],
            'gpu_memory_peak_mb': gpu_peak,
            'tree_depth': tree_stats['max_depth'],
            'total_nodes': tree_stats['total_nodes'],
            'leaf_nodes': tree_stats['num_leaf_nodes'],
            'internal_nodes': tree_stats['num_internal_nodes'],
            'success': True,
            'log_file': str(log_file)
        }
        
        # クラスタリング統計を追加
        if hasattr(colbert_system, 'stats'):
            stats = colbert_system.stats
            if 'silhouette_scores' in stats and stats['silhouette_scores']:
                result['avg_silhouette'] = np.mean(stats['silhouette_scores'])
                result['avg_dbi'] = np.mean(stats['dbi_scores'])
                result['avg_chi'] = np.mean(stats['chi_scores'])
        
        results.append(result)
        
        # ツリーをpickleで保存
        tree_file = results_dir / f"scaling_test_tree_{sample_size}chunks_{test_timestamp}.pkl"
        try:
            import pickle
            with open(tree_file, 'wb') as f:
                pickle.dump({
                    'tree': tree,
                    'sample_size': sample_size,
                    'build_time': build_time,
                    'stats': tree_stats,
                    'timestamp': test_timestamp
                }, f)
            print(f"  💾 ツリー保存: {tree_file.name}")
            result['tree_file'] = str(tree_file)
        except Exception as e:
            print(f"  ⚠️ ツリー保存失敗: {e}")
        
        # 結果をログファイルにも記録
        with open(log_file, 'a', encoding='utf-8') as log_f:
            log_f.write(f"\n構築時間: {build_time:.1f}秒 ({build_time/60:.1f}分)\n")
            log_f.write(f"GPU メモリ使用量:\n")
            log_f.write(f"  開始前: {gpu_before['memory_used_mb']:.0f}MB\n")
            log_f.write(f"  完了後: {gpu_after['memory_used_mb']:.0f}MB\n")
            log_f.write(f"  ピーク: {gpu_peak:.0f}MB\n")
            log_f.write(f"ツリー統計:\n")
            log_f.write(f"  深度: {tree_stats['max_depth']}\n")
            log_f.write(f"  総ノード数: {tree_stats['total_nodes']}\n")
            log_f.write(f"  リーフノード: {tree_stats['num_leaf_nodes']}\n")
            log_f.write(f"  内部ノード: {tree_stats['num_internal_nodes']}\n")
            if 'avg_silhouette' in result:
                log_f.write(f"クラスタリング品質:\n")
                log_f.write(f"  平均Silhouette: {result['avg_silhouette']:.3f}\n")
                log_f.write(f"  平均DBI: {result['avg_dbi']:.3f}\n")
                log_f.write(f"  平均CHI: {result['avg_chi']:.1f}\n")
            log_f.write("\n" + "="*80 + "\n")
            log_f.write("テスト完了\n")
            log_f.write("="*80 + "\n")
        
        print(f"\n✅ テスト {test_num} 完了")
        print(f"  構築時間: {build_time:.1f}秒 ({build_time/60:.1f}分)")
        print(f"  GPU メモリピーク: {gpu_peak:.0f}MB")
        print(f"  ツリー深度: {tree_stats['max_depth']}")
        print(f"  総ノード数: {tree_stats['total_nodes']}")
        print(f"  リーフノード: {tree_stats['num_leaf_nodes']}")
        print(f"  内部ノード: {tree_stats['num_internal_nodes']}")
        
        if 'avg_silhouette' in result:
            print(f"  平均Silhouette: {result['avg_silhouette']:.3f}")
        print(f"  📄 ログ保存: {log_file.name}")
        
    except Exception as e:
        print(f"\n❌ テスト {test_num} 失敗: {e}")
        
        # エラーログを記録
        import traceback
        error_detail = traceback.format_exc()
        
        with open(log_file, 'a', encoding='utf-8') as log_f:
            log_f.write("\n" + "="*80 + "\n")
            log_f.write("❌ エラー発生\n")
            log_f.write("="*80 + "\n")
            log_f.write(f"エラーメッセージ: {str(e)}\n\n")
            log_f.write("詳細トレースバック:\n")
            log_f.write(error_detail)
            log_f.write("\n" + "="*80 + "\n")
        
        result = {
            'test_number': test_num,
            'sample_size': sample_size,
            'success': False,
            'error': str(e),
            'log_file': str(log_file)
        }
        results.append(result)
        print(f"  📄 エラーログ保存: {log_file.name}")
    
    # メモリクリーンアップ
    del colbert_system
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"  メモリクリーンアップ完了")

# 結果保存
print(f"\n[ステップ 4/6] 結果をJSON保存中...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = results_dir / f"scaling_test_{timestamp}.json"

with open(results_file, 'w', encoding='utf-8') as f:
    json.dump({
        'timestamp': timestamp,
        'test_description': 'RAPTOR Tree Scaling Test',
        'sample_sizes': sample_sizes,
        'results': results
    }, f, indent=2, ensure_ascii=False)

print(f"✅ 結果保存: {results_file}")

# グラフ生成
print(f"\n[ステップ 5/6] グラフ生成中...")

successful_results = [r for r in results if r.get('success', False)]

if len(successful_results) > 0:
    # データ抽出
    sizes = [r['sample_size'] for r in successful_results]
    build_times = [r['build_time'] for r in successful_results]
    gpu_peaks = [r['gpu_memory_peak_mb'] for r in successful_results]
    tree_depths = [r['tree_depth'] for r in successful_results]
    total_nodes = [r['total_nodes'] for r in successful_results]
    
    # 4つのサブプロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RAPTOR Tree構築 スケーリングテスト結果', fontsize=16, fontweight='bold')
    
    # 1. 構築時間
    ax1 = axes[0, 0]
    ax1.plot(sizes, build_times, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('チャンク数', fontsize=11)
    ax1.set_ylabel('構築時間 (秒)', fontsize=11)
    ax1.set_title('1. 構築時間の推移', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(sizes, build_times)):
        ax1.annotate(f'{y:.1f}秒\n({y/60:.1f}分)', 
                    xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=9)
    
    # 2. GPU メモリ使用量
    ax2 = axes[0, 1]
    ax2.plot(sizes, gpu_peaks, marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('チャンク数', fontsize=11)
    ax2.set_ylabel('GPU メモリピーク (MB)', fontsize=11)
    ax2.set_title('2. GPU メモリ使用量の推移', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(sizes, gpu_peaks)):
        ax2.annotate(f'{y:.0f}MB', 
                    xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=9)
    
    # 3. ツリー深度
    ax3 = axes[1, 0]
    ax3.plot(sizes, tree_depths, marker='^', linewidth=2, markersize=8, color='#F18F01')
    ax3.set_xlabel('チャンク数', fontsize=11)
    ax3.set_ylabel('ツリー深度', fontsize=11)
    ax3.set_title('3. ツリー深度の推移', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(tree_depths) + 1)
    for i, (x, y) in enumerate(zip(sizes, tree_depths)):
        ax3.annotate(f'{y}', 
                    xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=9)
    
    # 4. 総ノード数
    ax4 = axes[1, 1]
    ax4.plot(sizes, total_nodes, marker='D', linewidth=2, markersize=8, color='#6A994E')
    ax4.set_xlabel('チャンク数', fontsize=11)
    ax4.set_ylabel('総ノード数', fontsize=11)
    ax4.set_title('4. 総ノード数の推移', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(sizes, total_nodes)):
        ax4.annotate(f'{y}', 
                    xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # グラフ保存
    graph_file = results_dir / f"scaling_test_graph_{timestamp}.png"
    plt.savefig(graph_file, dpi=300, bbox_inches='tight')
    print(f"✅ グラフ保存: {graph_file}")
    
    # 追加グラフ: 効率性分析
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('効率性分析', fontsize=16, fontweight='bold')
    
    # チャンクあたりの処理時間
    ax5 = axes2[0]
    time_per_chunk = [t/s for t, s in zip(build_times, sizes)]
    ax5.plot(sizes, time_per_chunk, marker='o', linewidth=2, markersize=8, color='#BC4749')
    ax5.set_xlabel('チャンク数', fontsize=11)
    ax5.set_ylabel('チャンクあたり処理時間 (秒)', fontsize=11)
    ax5.set_title('5. チャンクあたり処理時間', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(sizes, time_per_chunk)):
        ax5.annotate(f'{y:.3f}秒', 
                    xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=9)
    
    # ノードあたりの処理時間
    ax6 = axes2[1]
    time_per_node = [t/n if n > 0 else 0 for t, n in zip(build_times, total_nodes)]
    ax6.plot(sizes, time_per_node, marker='s', linewidth=2, markersize=8, color='#386641')
    ax6.set_xlabel('チャンク数', fontsize=11)
    ax6.set_ylabel('ノードあたり処理時間 (秒)', fontsize=11)
    ax6.set_title('6. ノードあたり処理時間', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(sizes, time_per_node)):
        ax6.annotate(f'{y:.2f}秒', 
                    xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    graph_file2 = results_dir / f"scaling_test_efficiency_{timestamp}.png"
    plt.savefig(graph_file2, dpi=300, bbox_inches='tight')
    print(f"✅ 効率性グラフ保存: {graph_file2}")

# 知見まとめ
print(f"\n[ステップ 6/6] 知見のまとめ")
print("=" * 80)

if len(successful_results) > 0:
    print("\n📊 スケーリング特性:")
    
    # 時間のスケーリング
    if len(successful_results) >= 2:
        time_ratio = build_times[-1] / build_times[0]
        size_ratio = sizes[-1] / sizes[0]
        print(f"  ・構築時間の増加率: {time_ratio:.2f}x (データ量 {size_ratio:.2f}x)")
        print(f"    → スケーリング係数: O(n^{np.log(time_ratio)/np.log(size_ratio):.2f})")
    
    # メモリのスケーリング
    if len(successful_results) >= 2:
        mem_increase = gpu_peaks[-1] - gpu_peaks[0]
        print(f"  ・GPU メモリ増加量: +{mem_increase:.0f}MB ({gpu_peaks[0]:.0f}MB → {gpu_peaks[-1]:.0f}MB)")
    
    # ツリー構造の変化
    print(f"\n🌳 ツリー構造の推移:")
    for r in successful_results:
        print(f"  ・{r['sample_size']:4d}チャンク: 深度{r['tree_depth']}, ノード{r['total_nodes']:3d}個 "
              f"(リーフ{r['leaf_nodes']:3d}, 内部{r['internal_nodes']:2d})")
    
    # クラスタリング品質
    if any('avg_silhouette' in r for r in successful_results):
        print(f"\n🎯 クラスタリング品質:")
        for r in successful_results:
            if 'avg_silhouette' in r:
                print(f"  ・{r['sample_size']:4d}チャンク: Silhouette={r['avg_silhouette']:.3f}, "
                      f"DBI={r['avg_dbi']:.3f}, CHI={r['avg_chi']:.1f}")
    
    # 実用性評価
    print(f"\n💡 実用性評価:")
    for r in successful_results:
        time_min = r['build_time'] / 60
        if time_min < 5:
            rating = "⭐⭐⭐ 高速"
        elif time_min < 15:
            rating = "⭐⭐ 実用的"
        else:
            rating = "⭐ 要検討"
        print(f"  ・{r['sample_size']:4d}チャンク: {time_min:.1f}分 {rating}")

print("\n" + "=" * 80)
print("✅ スケーリングテスト完了")
print("=" * 80)
