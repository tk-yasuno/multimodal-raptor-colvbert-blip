"""
東日本大震災 教訓継承 RAPTOR システム
Tsunami Lesson RAPTOR: Hierarchical Retrieval for Disaster Education

東日本大震災の教訓を階層的に構造化し、効率的な検索と要約を実現するRAGシステム
- 復興庁資料、今村文彦教授の研究、防災教育事例を統合
- RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) による階層検索
- FAISS + K-means クラスタリングによる最適化

Version: 1.0 - Tsunami Lesson Edition
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import json
import pickle
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# GPU最適化版のRAPTORRetrieverEvalをインポート（同じディレクトリ内）
from raptor_eval import RAPTORRetrieverEval


class TsunamiLessonRAPTOR(RAPTORRetrieverEval):
    """
    東日本大震災の教訓を継承するためのRAPTORシステム
    
    特徴:
    - 災害教訓に特化したチャンク分割
    - 階層的な知識構造の構築
    - 文脈を保持した要約生成
    """
    
    def __init__(
        self,
        embeddings_model,
        llm,
        min_clusters: int = 2,
        max_clusters: int = 5,
        max_depth: int = 3,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        selection_strategy: str = 'silhouette',
        metric_weights: Optional[Dict[str, float]] = None,
        disaster_domain: str = 'tohoku_earthquake',
        **kwargs  # 親クラスの追加パラメータを受け取る
    ):
        """
        Args:
            disaster_domain: 災害ドメイン ('tohoku_earthquake', 'general_disaster' など)
            metric_weights: メトリック重み（combined戦略用）
            max_depth: ツリーの最大深さ（デフォルト3）
            selection_strategy: クラスタ数選択戦略（デフォルト'silhouette'）
            **kwargs: 親クラス(RAPTORRetrieverEval)の追加パラメータ
        """
        # デフォルトのmetric_weightsを設定（silhouette重視）
        if metric_weights is None:
            metric_weights = {'silhouette': 1.0, 'dbi': 0.0, 'chi': 0.0}
        
        super().__init__(
            embeddings_model=embeddings_model,
            llm=llm,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            max_depth=max_depth,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            selection_strategy=selection_strategy,
            metric_weights=metric_weights,
            **kwargs  # n_iterなどの追加パラメータを渡す
        )
        
        self.disaster_domain = disaster_domain
        print(f"🌊 Tsunami Lesson RAPTOR initialized")
        print(f"   Domain: {disaster_domain}")
        print(f"   Optimized for disaster education and lesson inheritance")
    
    def create_disaster_specific_prompt(self) -> ChatPromptTemplate:
        """
        災害教訓に特化した要約プロンプトを作成
        """
        template = """あなたは東日本大震災の教訓を次世代に継承する専門家です。
以下の文書群は、災害対応や防災教育に関する重要な情報を含んでいます。

これらの文書を、以下の観点から要約してください：
1. 災害時の具体的な対応や事例
2. 得られた教訓や学び
3. 今後の防災に活かせる知見
4. 数値データや固有名詞は正確に保持

文書群：
{documents}

要約（日本語で300-500文字）："""
        
        return ChatPromptTemplate.from_template(template)
    
    def load_tohoku_earthquake_data(self, file_path: str = None) -> List[Document]:
        """
        東日本大震災データを読み込み
        
        Args:
            file_path: データファイルのパス（デフォルト: tohoku_earthquake_data.txt）
        """
        if file_path is None:
            file_path = Path(__file__).parent / "tohoku_earthquake_data.txt"
        
        print(f"📖 Loading disaster knowledge base: {file_path}")
        
        # ファイルを直接読み込み（TextLoaderの文字数制限を回避）
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        print(f"Loaded document length: {len(full_text)} characters")
        
        # Document オブジェクトを作成
        doc = Document(page_content=full_text, metadata={"source": str(file_path)})
        
        # チャンクに分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents([doc])
        
        print(f"Split into {len(chunks)} chunks")
        print(f"✅ Loaded {len(chunks)} chunks from disaster knowledge base")
        print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")
        
        return chunks
    
    def build_disaster_tree(
        self,
        documents: List[Document],
        save_dir: str = "saved_models/tsunami_lesson"
    ) -> Dict:
        """
        災害教訓の階層ツリーを構築
        
        Args:
            documents: 入力ドキュメント
            save_dir: 保存先ディレクトリ
            
        Returns:
            ツリー構造の辞書
        """
        print("\n" + "="*80)
        print("🌲 Building Tsunami Lesson Tree (災害教訓階層ツリー構築)")
        print("="*80)
        
        start_time = time.time()
        
        # 災害特化プロンプトを使用して要約
        disaster_prompt = self.create_disaster_specific_prompt()
        
        # 元のLLMチェーンを一時的に置き換え
        original_llm = self.llm
        self.llm = disaster_prompt | self.llm | StrOutputParser()
        
        # 階層構築
        tree = self.build_tree(documents, depth=0)
        
        # LLMチェーンを戻す
        self.llm = original_llm
        
        # 統計情報の表示
        total_time = time.time() - start_time
        print(f"\n✅ Tree construction completed in {total_time:.2f} seconds")
        print(f"   Total nodes: {self._count_nodes(tree)}")
        print(f"   Tree depth: {self._get_tree_depth(tree)}")
        
        # 保存（ツリーをセット - 両方の属性に設定）
        self.tree = tree
        self.tree_structure = tree  # retrieveメソッドが参照する
        if save_dir:
            self.save(save_dir)
        
        return tree
    
    def _count_nodes(self, tree: Dict, count: int = 0) -> int:
        """ツリーのノード数をカウント"""
        count += 1
        if 'children' in tree:
            for child in tree['children']:
                count = self._count_nodes(child, count)
        return count
    
    def _get_tree_depth(self, tree: Dict, current_depth: int = 0) -> int:
        """ツリーの深さを取得"""
        if 'children' not in tree or len(tree['children']) == 0:
            return current_depth
        
        max_child_depth = 0
        for child in tree['children']:
            child_depth = self._get_tree_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def search_lessons(
        self,
        query: str,
        tree: Dict = None,
        top_k: int = 5,
        use_hierarchical: bool = True
    ) -> List[Tuple[str, float, int]]:
        """
        災害教訓を検索
        
        Args:
            query: 検索クエリ
            tree: ツリー構造（Noneの場合は保存済みを読み込み）
            top_k: 取得する文書数
            use_hierarchical: 階層検索を使用するか
            
        Returns:
            (文書内容, スコア, 階層レベル) のリスト
        """
        print(f"\n🔍 Searching for lessons: '{query}'")
        
        # ツリーが指定されていない場合は、self.treeまたはself.tree_structureを使用
        if tree is None:
            if hasattr(self, 'tree') and self.tree is not None:
                # build_disaster_treeで構築した場合
                self.tree_structure = self.tree
            elif hasattr(self, 'tree_structure') and self.tree_structure is not None:
                # loadで読み込んだ場合
                pass  # self.tree_structureをそのまま使用
            else:
                raise ValueError("ツリーが構築されていません。先にbuild_disaster_tree()を実行してください。")
        else:
            # 明示的にツリーが指定された場合
            self.tree_structure = tree
        
        # 基底クラスのretrieveメソッドを呼び出し（内部でself.tree_structureを使用）
        documents = self.retrieve(query, top_k)
        
        # Document オブジェクトを (content, score, level) のタプルに変換
        results = []
        for doc in documents:
            content = doc.page_content
            # メタデータからスコアとレベルを取得（存在しない場合はデフォルト値）
            # 注: search_treeメソッドは'similarity'キーで保存する
            score = doc.metadata.get('similarity', doc.metadata.get('score', 0.0))
            level = doc.metadata.get('level', 0)
            results.append((content, score, level))
        
        print(f"✅ Found {len(results)} relevant documents")
        
        return results
    
    def generate_lesson_summary(
        self,
        query: str,
        search_results: List[Tuple[str, float, int]],
        context_window: int = 3
    ) -> str:
        """
        検索結果から教訓要約を生成
        
        Args:
            query: 元のクエリ
            search_results: 検索結果
            context_window: 使用する文書数
            
        Returns:
            生成された要約
        """
        # 上位の文書を選択
        top_docs = search_results[:context_window]
        
        # 文書を結合
        context = "\n\n---\n\n".join([doc for doc, _, _ in top_docs])
        
        # 要約生成プロンプト
        prompt = ChatPromptTemplate.from_template(
            """あなたは防災教育の専門家です。以下の質問に対して、提供された文書から得られる教訓を要約してください。

質問: {query}

参考文書:
{context}

回答（重要な教訓とその根拠を含めて、300-500文字で）:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        print(f"\n💡 Generating lesson summary...")
        
        summary = chain.invoke({
            "query": query,
            "context": context
        })
        
        return summary
    
    def export_tree_visualization(
        self,
        tree: Dict,
        output_path: str = "tsunami_lesson_tree.json"
    ):
        """
        ツリー構造を可視化用にエクスポート
        
        Args:
            tree: ツリー構造
            output_path: 出力先パス
        """
        print(f"\n📊 Exporting tree structure for visualization...")
        
        # 簡略化されたツリー構造を作成
        def simplify_tree(node, level=0):
            simplified = {
                "level": level,
                "has_summary": "summary" in node and node["summary"] is not None,
                "summary_length": len(node.get("summary", "")) if node.get("summary") else 0,
                "num_chunks": len(node.get("chunk_ids", [])),
                "children": []
            }
            
            if "children" in node:
                for child in node["children"]:
                    simplified["children"].append(simplify_tree(child, level + 1))
            
            return simplified
        
        viz_tree = simplify_tree(tree)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(viz_tree, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Tree structure exported to {output_path}")
    
    def evaluate_search_quality(
        self,
        test_queries: List[str],
        tree: Dict,
        ground_truth: Dict[str, List[str]] = None
    ) -> Dict[str, float]:
        """
        検索品質を評価
        
        Args:
            test_queries: テストクエリのリスト
            tree: ツリー構造
            ground_truth: 正解データ（クエリ -> 期待される回答のリスト）
            
        Returns:
            評価メトリクス
        """
        print(f"\n📈 Evaluating search quality with {len(test_queries)} queries...")
        
        results = {
            'avg_search_time': 0.0,
            'avg_num_results': 0.0,
            'queries_tested': len(test_queries)
        }
        
        total_time = 0.0
        total_results = 0
        
        for query in test_queries:
            start = time.time()
            search_results = self.search_lessons(query, tree, top_k=5)
            search_time = time.time() - start
            
            total_time += search_time
            total_results += len(search_results)
        
        results['avg_search_time'] = total_time / len(test_queries)
        results['avg_num_results'] = total_results / len(test_queries)
        
        print(f"✅ Evaluation completed")
        print(f"   Average search time: {results['avg_search_time']:.3f} seconds")
        print(f"   Average results per query: {results['avg_num_results']:.1f}")
        
        return results


def main():
    """
    メイン実行関数
    """
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    
    print("="*80)
    print("Tsunami Lesson RAPTOR System")
    print("東日本大震災 教訓継承システム")
    print("="*80)
    
    # モデル初期化
    print("\nInitializing models...")
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )
    llm = ChatOllama(
        model="granite-code:8b",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=300  # 5分のタイムアウト
    )
    
    # TsunamiLessonRAPTOR初期化
    raptor = TsunamiLessonRAPTOR(
        embeddings_model=embeddings,
        llm=llm,
        min_clusters=2,
        max_clusters=5,
        max_depth=3,
        chunk_size=500,
        chunk_overlap=100,
        selection_strategy='silhouette',
        metric_weights={
            'silhouette': 1.0,
            'dbi': 0.0,
            'chi': 0.0
        },
        disaster_domain='tohoku_earthquake'
    )
    
    # データ読み込み
    print("\n" + "="*80)
    print("📚 Step 1: Loading Disaster Knowledge Base")
    print("="*80)
    documents = raptor.load_tohoku_earthquake_data()
    
    # ツリー構築
    print("\n" + "="*80)
    print("🌲 Step 2: Building Hierarchical Tree")
    print("="*80)
    tree = raptor.build_disaster_tree(documents, save_dir="saved_models/tsunami_lesson")
    
    # ツリー可視化エクスポート
    raptor.export_tree_visualization(tree, "tsunami_lesson_tree_viz.json")
    
    # テストクエリ
    print("\n" + "="*80)
    print("🔍 Step 3: Testing Search Functionality")
    print("="*80)
    
    test_queries = [
        "津波避難で有効だった行動は？",
        "カウンターパート方式とは何ですか？",
        "釜石の奇跡について教えてください",
        "災害時の情報伝達で重要なことは？",
        "復興まちづくりの課題は何ですか？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print('='*80)
        
        # 検索実行
        results = raptor.search_lessons(query, tree, top_k=3)
        
        # 結果表示
        print(f"\n📄 Top {len(results)} Results:")
        for j, (content, score, level) in enumerate(results, 1):
            print(f"\n[{j}] Score: {score:.4f} | Level: {level}")
            print(f"Content preview: {content[:200]}...")
        
        # 要約生成
        summary = raptor.generate_lesson_summary(query, results, context_window=2)
        print(f"\n💡 Generated Summary:")
        print(summary)
        print()
    
    # 評価
    print("\n" + "="*80)
    print("📊 Step 4: Evaluation")
    print("="*80)
    eval_results = raptor.evaluate_search_quality(test_queries, tree)
    
    print("\n✅ All tasks completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
