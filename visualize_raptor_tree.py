"""
RAPTOR Tree可視化スクリプト
NetworkXとMatplotlibを使用してツリー構造を可視化

保存されたscaling_test_tree_*.pklファイルからRAPTOR Treeを読み込み、
階層構造を視覚化してPNG/PDFとして保存します。
"""

import os
import sys
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from datetime import datetime
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

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


class RAPTORTreeVisualizer:
    """RAPTOR Treeの可視化クラス"""
    
    def __init__(self, tree_path: str):
        """
        Args:
            tree_path: RAPTORツリーのpickleファイルパス
        """
        self.tree_path = Path(tree_path)
        self.tree = None
        self.graph = None
        self.node_depths = {}
        self.node_types = {}  # 'leaf' or 'internal'
        self.node_keywords = {}  # ノードごとのキーワード
        self.all_summaries = []  # TF-IDF計算用の全サマリーテキスト
        self.node_to_summary_idx = {}  # ノードIDからサマリーインデックスへのマッピング
    
    def extract_keywords_tfidf(self, texts: List[str], top_n: int = 3) -> List[List[str]]:
        """
        TF-IDFを使用して各テキストから重要なキーワードを抽出
        
        Args:
            texts: サマリーテキストのリスト
            top_n: 各テキストから抽出するキーワード数
        
        Returns:
            各テキストのキーワードリスト
        """
        if not texts or len(texts) == 0:
            return []
        
        # TF-IDFベクトライザー設定
        # token_pattern: 日本語（2文字以上）と英語（3文字以上）をマッチ
        vectorizer = TfidfVectorizer(
            token_pattern=r'[一-龥ぁ-んァ-ヶー]{2,}|[a-zA-Z]{3,}',
            max_features=1000,
            stop_words=None
        )
        
        # ストップワードフィルタリング用
        stop_words = {'こと', 'もの', 'ため', 'よう', 'ところ', 'これ', 'それ', 
                     'あれ', 'この', 'その', 'あの', 'など', 'ほか', '以上',
                     'について', 'における', 'に関する', 'による', 'によって',
                     'として', 'とする', 'である', 'です', 'ます', 'した'}
        
        try:
            # TF-IDF行列を計算
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # 各テキストのキーワード抽出
            all_keywords = []
            for i in range(len(texts)):
                # i番目のテキストのTF-IDFスコアを取得
                tfidf_scores = tfidf_matrix[i].toarray()[0]
                
                # スコアでソート（降順）
                top_indices = tfidf_scores.argsort()[::-1]
                
                # ストップワードを除外しながら上位N個を取得
                keywords = []
                for idx in top_indices:
                    if len(keywords) >= top_n:
                        break
                    word = feature_names[idx]
                    if word not in stop_words and tfidf_scores[idx] > 0:
                        keywords.append(word)
                
                all_keywords.append(keywords)
            
            return all_keywords
            
        except Exception as e:
            print(f"⚠️ TF-IDF抽出エラー: {e}")
            # エラー時は空リストを返す
            return [[] for _ in texts]
        
    def load_tree(self):
        """Pickleファイルからツリーを読み込み"""
        print(f"📂 ツリー読み込み中: {self.tree_path.name}")
        
        with open(self.tree_path, 'rb') as f:
            data = pickle.load(f)
        
        # 辞書形式の場合、'tree'キーからツリーシステムを取得
        if isinstance(data, dict) and 'tree' in data:
            self.tree = data['tree']
            print(f"✅ ツリー読み込み完了（辞書形式）")
        else:
            self.tree = data
            print(f"✅ ツリー読み込み完了")
        
        return self.tree
    
    def build_graph(self):
        """NetworkXグラフを構築"""
        print("🔨 NetworkXグラフ構築中...")
        
        self.graph = nx.DiGraph()
        
        # ノードカウンター
        self.node_counter = 0
        self.leaf_nodes = []
        self.internal_nodes = []
        
        # 第1パス: 全サマリーテキストを収集
        temp_nodes = []  # (node_id, cluster_id, summary_text, depth, parent_id, has_children) のリスト
        
        def get_node_id():
            """ユニークなノードIDを生成"""
            self.node_counter += 1
            return f"node_{self.node_counter}"
        
        def collect_summaries(tree_dict, depth=0, parent_id=None):
            """
            第1パス: サマリーテキストを収集
            """
            if not isinstance(tree_dict, dict):
                return
            
            if 'clusters' in tree_dict and tree_dict['clusters']:
                clusters = tree_dict['clusters']
                
                for cluster_id, cluster_data in clusters.items():
                    # サマリーテキスト抽出
                    summary_text = ""
                    if 'summary' in cluster_data:
                        summary = cluster_data['summary']
                        if hasattr(summary, 'page_content'):
                            summary_text = summary.page_content
                        elif isinstance(summary, dict):
                            summary_text = summary.get('page_content', summary.get('text', ''))
                    
                    # 子の有無判定
                    has_children = ('children' in cluster_data and 
                                  cluster_data['children'] and 
                                  'clusters' in cluster_data['children'] and
                                  cluster_data['children']['clusters'])
                    
                    # ノードID生成
                    node_id = get_node_id()
                    
                    # 情報を保存
                    temp_nodes.append({
                        'node_id': node_id,
                        'cluster_id': cluster_id,
                        'summary_text': summary_text,
                        'depth': depth,
                        'parent_id': parent_id,
                        'has_children': has_children
                    })
                    
                    # サマリーテキストをリストに追加
                    self.all_summaries.append(summary_text if summary_text else "")
                    self.node_to_summary_idx[node_id] = len(self.all_summaries) - 1
                    
                    # 子ツリーを再帰的に処理
                    if has_children:
                        collect_summaries(
                            cluster_data['children'],
                            depth + 1,
                            node_id
                        )
        
        # 第1パス実行: サマリー収集
        if isinstance(self.tree, dict):
            collect_summaries(self.tree)
        else:
            print("❌ エラー: ツリーが辞書形式ではありません")
            return None
        
        print(f"   第1パス完了: {len(self.all_summaries)}個のサマリーを収集")
        
        # 第2パス: TF-IDFでキーワード抽出
        if len(self.all_summaries) > 0:
            print("   TF-IDFキーワード抽出中...")
            all_keywords = self.extract_keywords_tfidf(self.all_summaries, top_n=3)
        else:
            all_keywords = []
        
        # 第3パス: グラフ構築
        print("   グラフノード作成中...")
        for node_info in temp_nodes:
            node_id = node_info['node_id']
            cluster_id = node_info['cluster_id']
            summary_text = node_info['summary_text']
            depth = node_info['depth']
            parent_id = node_info['parent_id']
            has_children = node_info['has_children']
            
            # TF-IDFキーワード取得
            summary_idx = self.node_to_summary_idx[node_id]
            keywords = all_keywords[summary_idx] if summary_idx < len(all_keywords) else []
            
            self.node_keywords[node_id] = keywords
            
            # ノードタイプ判定
            node_type = 'internal' if has_children else 'leaf'
            
            if node_type == 'leaf':
                self.leaf_nodes.append(node_id)
            else:
                self.internal_nodes.append(node_id)
            
            self.node_types[node_id] = node_type
            self.node_depths[node_id] = depth
            
            # キーワードラベル作成（よりコンパクトに）
            if keywords:
                # 最も重要なキーワード1つのみ、または2つを改行で区切る
                if len(keywords) >= 2:
                    keyword_label = f"{keywords[0]}\n{keywords[1]}"
                else:
                    keyword_label = keywords[0]
            else:
                keyword_label = f"C{cluster_id}"
            
            self.graph.add_node(
                node_id,
                depth=depth,
                text=str(summary_text[:100]) if summary_text else "",
                keywords=keywords,
                node_type=node_type,
                label=keyword_label
            )
            
            # 親ノードとの接続
            if parent_id:
                self.graph.add_edge(parent_id, node_id)
        
        if self.graph.number_of_nodes() == 0:
            print("❌ エラー: ノードが抽出できませんでした")
            return None
        
        print(f"✅ グラフ構築完了")
        print(f"   総ノード数: {self.graph.number_of_nodes()}")
        print(f"   リーフノード: {len(self.leaf_nodes)}")
        print(f"   内部ノード: {len(self.internal_nodes)}")
        print(f"   エッジ数: {self.graph.number_of_edges()}")
        print(f"   最大深度: {max(self.node_depths.values()) if self.node_depths else 0}")
        
        return self.graph
    
    def visualize_tree(self, output_path: str, figsize: Tuple[int, int] = None, dpi: int = 150):
        """
        ツリーを可視化して保存
        
        Args:
            output_path: 出力ファイルパス（PNG/PDF）
            figsize: 図のサイズ（幅、高さ）。Noneの場合はノード数に応じて自動調整
            dpi: 解像度
        """
        if self.graph is None:
            print("❌ エラー: グラフが構築されていません。先にbuild_graph()を実行してください")
            return
        
        print(f"🎨 ツリー可視化中...")
        
        # ノード数に応じて図のサイズを自動調整
        if figsize is None:
            total_nodes = self.graph.number_of_nodes()
            if total_nodes > 30:
                figsize = (28, 14)  # 大きなツリー
            elif total_nodes > 20:
                figsize = (24, 12)  # 中規模ツリー
            else:
                figsize = (20, 10)  # 小規模ツリー
        
        # 図とサブプロット作成
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 階層レイアウト計算
        pos = self._compute_hierarchical_layout()
        
        # ノードの色設定
        node_colors = []
        for node in self.graph.nodes():
            if self.node_types[node] == 'leaf':
                node_colors.append('#90EE90')  # ライトグリーン（リーフ）
            else:
                depth = self.node_depths[node]
                # 深さに応じて色を変化（青系のグラデーション）
                intensity = 1.0 - (depth * 0.3)
                node_colors.append((0.3, 0.5, intensity))
        
        # ノードサイズ設定（よりコンパクトに）
        node_sizes = []
        for node in self.graph.nodes():
            if self.node_types[node] == 'leaf':
                node_sizes.append(500)  # リーフノードを少し大きく
            else:
                depth = self.node_depths[node]
                size = 1000 + (depth * 300)  # 内部ノードも調整
                node_sizes.append(size)
        
        # エッジ描画
        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            edge_color='gray',
            alpha=0.5,
            arrows=True,
            arrowsize=15,
            width=1.5,
            connectionstyle='arc3,rad=0.1'
        )
        
        # ノード描画
        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=2
        )
        
        # ラベル描画（キーワード表示）
        labels = nx.get_node_attributes(self.graph, 'label')
        
        # ラベル位置をノードの少し下に調整
        label_pos = {}
        y_offset = 0.08  # Y軸方向のオフセット（ノードの下に配置）
        for node, (x, y) in pos.items():
            label_pos[node] = (x, y - y_offset)
        
        # フォントサイズを調整（ノード数に応じて）
        total_nodes = self.graph.number_of_nodes()
        if total_nodes > 50:
            font_size = 5
        elif total_nodes > 30:
            font_size = 6
        elif total_nodes > 20:
            font_size = 7
        else:
            font_size = 8
        
        nx.draw_networkx_labels(
            self.graph, label_pos, labels, ax=ax,
            font_size=font_size,
            font_weight='bold',
            font_family='MS Gothic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.85)
        )
        
        # タイトル設定
        tree_name = self.tree_path.stem
        total_nodes = self.graph.number_of_nodes()
        max_depth = max(self.node_depths.values()) + 1
        leaf_count = sum(1 for t in self.node_types.values() if t == 'leaf')
        internal_count = total_nodes - leaf_count
        
        ax.set_title(
            f"RAPTOR Tree: {tree_name}\n"
            f"Total Nodes: {total_nodes} (Leaf: {leaf_count}, Internal: {internal_count}) | "
            f"Max Depth: {max_depth}",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # 凡例追加
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#90EE90', edgecolor='black', label='リーフノード (Depth 0)'),
            Patch(facecolor=(0.3, 0.5, 0.7), edgecolor='black', label='内部ノード (Depth 1+)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.axis('off')
        plt.tight_layout()
        
        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 可視化完了: {output_path}")
        print(f"   ファイルサイズ: {output_path.stat().st_size / 1024:.1f} KB")
    
    def _compute_hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """階層レイアウトを計算（ラベル重複を防ぐ）"""
        pos = {}
        
        # 深度ごとにノードをグループ化
        depth_groups = {}
        for node, depth in self.node_depths.items():
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(node)
        
        max_depth = max(self.node_depths.values())
        
        # 各深度で必要な最小幅を計算
        # リーフノードが多い場合は幅を広げる
        max_nodes_in_depth = max(len(nodes) for nodes in depth_groups.values())
        
        # ラベルの重複を避けるための最小間隔を設定
        # ノード数が多いほど広い範囲に配置
        if max_nodes_in_depth > 15:
            x_margin = 0.02
            x_range = (0.01, 0.99)
        elif max_nodes_in_depth > 10:
            x_margin = 0.05
            x_range = (0.03, 0.97)
        else:
            x_margin = 0.1
            x_range = (0.05, 0.95)
        
        # 各深度のノードを配置
        for depth, nodes in depth_groups.items():
            y = max_depth - depth  # 上から下へ（ルートが上）
            n = len(nodes)
            
            # X座標を均等配置（広い範囲で）
            if n == 1:
                x_coords = [0.5]
            else:
                x_coords = np.linspace(x_range[0], x_range[1], n)
            
            for i, node in enumerate(nodes):
                pos[node] = (x_coords[i], y)
        
        return pos
    
    def create_statistics_plot(self, output_path: str):
        """ツリー統計のプロット作成"""
        print("📊 統計プロット作成中...")
        
        # 深度ごとのノード数集計
        depth_counts = {}
        for node, depth in self.node_depths.items():
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        depths = sorted(depth_counts.keys())
        counts = [depth_counts[d] for d in depths]
        
        # プロット
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
        
        # 1. 深度別ノード数
        colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
        ax1.bar(depths, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('深度 (Depth)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ノード数', fontsize=12, fontweight='bold')
        ax1.set_title('深度別ノード数', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 値を棒グラフ上に表示
        for i, (d, c) in enumerate(zip(depths, counts)):
            ax1.text(d, c + max(counts)*0.02, str(c), ha='center', va='bottom', fontweight='bold')
        
        # 2. ツリー統計サマリー
        total_nodes = sum(counts)
        leaf_nodes = counts[0] if 0 in depth_counts else 0
        internal_nodes = total_nodes - leaf_nodes
        max_depth = max(depths)
        avg_branching = internal_nodes / max(1, max_depth)
        
        stats_text = (
            f"総ノード数: {total_nodes}\n"
            f"リーフノード: {leaf_nodes}\n"
            f"内部ノード: {internal_nodes}\n"
            f"最大深度: {max_depth}\n"
            f"平均分岐数: {avg_branching:.2f}\n"
            f"リーフ比率: {leaf_nodes/total_nodes*100:.1f}%"
        )
        
        ax2.text(0.5, 0.5, stats_text, 
                 ha='center', va='center',
                 fontsize=14,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 family='MS Gothic')
        ax2.axis('off')
        ax2.set_title('ツリー統計サマリー', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 統計プロット保存: {output_path}")


def visualize_all_trees(
    pkl_dir: str,
    output_dir: str,
    pattern: str = "scaling_test_tree_*.pkl"
):
    """
    指定ディレクトリ内の全RAPTORツリーを可視化
    
    Args:
        pkl_dir: pklファイルがあるディレクトリ
        output_dir: 出力先ディレクトリ
        pattern: ファイル名パターン
    """
    pkl_dir = Path(pkl_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # pklファイルを検索
    pkl_files = list(pkl_dir.glob(pattern))
    
    if not pkl_files:
        print(f"❌ エラー: {pkl_dir} に {pattern} が見つかりません")
        return
    
    print("=" * 80)
    print("RAPTOR Tree 可視化")
    print(f"検出ファイル数: {len(pkl_files)}")
    print("=" * 80)
    
    for i, pkl_file in enumerate(sorted(pkl_files), 1):
        print(f"\n[{i}/{len(pkl_files)}] 処理中: {pkl_file.name}")
        print("-" * 80)
        
        try:
            # 可視化インスタンス作成
            visualizer = RAPTORTreeVisualizer(pkl_file)
            
            # ツリー読み込み
            visualizer.load_tree()
            
            # グラフ構築
            visualizer.build_graph()
            
            # 出力ファイル名生成
            base_name = pkl_file.stem
            tree_png = output_dir / f"{base_name}_tree.png"
            stats_png = output_dir / f"{base_name}_stats.png"
            
            # ツリー図作成
            visualizer.visualize_tree(tree_png, figsize=(24, 14), dpi=150)
            
            # 統計プロット作成
            visualizer.create_statistics_plot(stats_png)
            
            print(f"✅ [{i}/{len(pkl_files)}] 完了\n")
            
        except Exception as e:
            print(f"❌ エラー: {pkl_file.name} の処理中に問題が発生しました")
            print(f"   詳細: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("✅ 全ての可視化が完了しました")
    print(f"📂 出力先: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    # デフォルトパス設定
    pkl_dir = r"C:\Users\yasun\LangChain\learning-langchain\multimodal-raptor-colvbert-blip\data\encoder_comparison_46pdfs\results"
    output_dir = r"C:\Users\yasun\LangChain\learning-langchain\multimodal-raptor-colvbert-blip\data\encoder_comparison_46pdfs\raptor_trees"
    
    # 全ツリーを可視化
    visualize_all_trees(
        pkl_dir=pkl_dir,
        output_dir=output_dir,
        pattern="scaling_test_tree_*.pkl"
    )
