"""
RAPTOR with FAISS + Multiple Clustering Evaluation Metrics

複数のクラスタリング評価指標を統合した RAPTOR 実装
- Silhouette Score (クラスタの凝集度と分離度)
- Davies-Bouldin Index (クラスタ間類似度とクラスタ内分散)
- Calinski-Harabasz Index (分散比率)
- BIC (Bayesian Information Criterion)
- AIC (Akaike Information Criterion)

Version: 3.0 - Evaluation Metrics Edition
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


class RAPTORRetrieverEval:
    """
    RAPTOR with Multiple Clustering Evaluation Metrics
    
    複数の評価指標でクラスタ数を最適化
    """
    
    def __init__(
        self,
        embeddings_model,
        llm,
        min_clusters: int = 2,
        max_clusters: int = 10,
        max_depth: int = 3,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        n_iter: int = 20,
        selection_strategy: str = 'silhouette',
        metric_weights: Dict[str, float] = None,
        visual_encoder=None,
        use_multimodal: bool = False,
        multimodal_weight: float = 0.5
    ):
        """
        Args:
            embeddings_model: Embeddings model
            llm: LLM for summarization
            min_clusters: 最小クラスタ数
            max_clusters: 最大クラスタ数
            max_depth: 最大階層深さ
            chunk_size: チャンクサイズ
            chunk_overlap: チャンク重複
            n_iter: FAISS k-meansの反復回数
            selection_strategy: クラスタ数選択戦略
                - 'silhouette': Silhouette Score (推奨)
                - 'dbi': Davies-Bouldin Index
                - 'chi': Calinski-Harabasz Index
                - 'bic': BIC (従来手法)
                - 'aic': AIC
                - 'combined': 複数指標の組み合わせ
            metric_weights: combined戦略での重み
                例: {'silhouette': 0.4, 'dbi': 0.3, 'chi': 0.3}
            visual_encoder: 画像エンコーダー（ColVBERT/ColModernVBERT）
            use_multimodal: マルチモーダル埋め込みを使用するか
            multimodal_weight: 画像埋め込みの重み（0.0-1.0）
        """
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.max_depth = max_depth
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.n_iter = n_iter
        self.selection_strategy = selection_strategy
        self.visual_encoder = visual_encoder
        self.use_multimodal = use_multimodal
        self.multimodal_weight = multimodal_weight
        
        # デフォルトの重み設定
        # 実験結果: CHIはk=2を強く好むバイアスがあるため除外
        # Silhouette + DBI のバランスで最適化
        if metric_weights is None:
            self.metric_weights = {
                'silhouette': 0.5,  # クラスタ品質（ミクロ視点）
                'dbi': 0.5,         # クラスタ分離度（マクロ視点）
                'chi': 0.0          # CHI除外（k=2バイアス回避）
            }
        else:
            self.metric_weights = metric_weights
        
        self.tree_structure = {}
        self.stats = {
            'selections': [],
            'silhouette_scores': [],
            'dbi_scores': [],
            'chi_scores': [],
            'bic_scores': [],
            'aic_scores': []
        }
        
        print(f"RAPTOR with {selection_strategy.upper()} evaluation initialized")
        print(f"   Parameters:")
        print(f"   - Cluster range: {min_clusters}-{max_clusters}")
        print(f"   - Strategy: {selection_strategy}")
        if selection_strategy == 'combined':
            print(f"   - Weights: {metric_weights}")
        print(f"   - max_depth: {max_depth}")
        print(f"   - chunk_size: {chunk_size}")
    
    def load_and_split_documents(self, file_path: str, encoding: str = "utf-8") -> List[Document]:
        """ドキュメントを読み込み、チャンクに分割"""
        loader = TextLoader(file_path, encoding=encoding)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        print(f"Loaded document length: {len(documents[0].page_content)} characters")
        print(f"Split into {len(chunks)} chunks")
        
        return chunks
    
    def embed_documents(self, documents: List[Document], batch_size: int = 40) -> np.ndarray:
        """ドキュメントをバッチで埋め込み（マルチモーダル対応・GPU最適化版）"""
        import torch
        from PIL import Image
        
        all_embeddings = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        if self.use_multimodal and self.visual_encoder:
            print(f"  🎨 マルチモーダル埋め込み生成: {len(documents)}ドキュメント, {total_batches}バッチ (サイズ: {batch_size})")
        else:
            print(f"  📝 テキスト埋め込み生成: {len(documents)}ドキュメント, {total_batches}バッチ (サイズ: {batch_size})")
        
        for batch_idx, i in enumerate(range(0, len(documents), batch_size), 1):
            batch = documents[i:i + batch_size]
            
            # テキスト準備
            texts = []
            for doc in batch:
                if hasattr(doc, 'page_content'):
                    texts.append(doc.page_content)
                elif hasattr(doc, 'cached_text'):
                    texts.append(doc.cached_text)
                elif isinstance(doc, str):
                    texts.append(doc)
                else:
                    texts.append(str(doc))
            
            # マルチモーダル処理（GPU最適化）
            if self.use_multimodal and self.visual_encoder:
                try:
                    # 画像準備
                    images = []
                    valid_indices = []
                    for idx, doc in enumerate(batch):
                        if hasattr(doc, 'image_path') and doc.image_path:
                            try:
                                img = Image.open(doc.image_path).convert('RGB')
                                images.append(img)
                                valid_indices.append(idx)
                            except Exception as e:
                                pass
                    
                    if images:
                        # ✨ GPU最適化: テキストと画像の埋め込みをGPU上で融合
                        with torch.no_grad():
                            # 1. テキスト埋め込み（CPUで生成）
                            text_embeddings_np = np.array(
                                self.embeddings_model.embed_documents(texts), 
                                dtype=np.float32
                            )
                            text_embeddings_tensor = torch.from_numpy(text_embeddings_np).to(
                                self.visual_encoder.device
                            )
                            
                            # 2. 画像埋め込み（GPUで生成）
                            image_embeddings_tensor = self.visual_encoder.encode_image(images)
                            
                            # 3. GPU上で融合処理（メモリ転送を最小化）
                            for local_idx, doc_idx in enumerate(valid_indices):
                                text_emb = text_embeddings_tensor[doc_idx]
                                img_emb = image_embeddings_tensor[local_idx]
                                
                                # 次元調整
                                if text_emb.shape[0] != img_emb.shape[0]:
                                    if img_emb.shape[0] < text_emb.shape[0]:
                                        # パディング（GPU上で）
                                        padding = torch.zeros(
                                            text_emb.shape[0] - img_emb.shape[0],
                                            device=img_emb.device,
                                            dtype=img_emb.dtype
                                        )
                                        img_emb = torch.cat([img_emb, padding])
                                    else:
                                        # トランケート
                                        img_emb = img_emb[:text_emb.shape[0]]
                                
                                # GPU上で重み付き結合
                                combined = (1 - self.multimodal_weight) * text_emb + \
                                          self.multimodal_weight * img_emb
                                text_embeddings_tensor[doc_idx] = combined
                            
                            # 4. 一括でCPUに転送（転送回数を削減）
                            text_embeddings = text_embeddings_tensor.cpu().numpy()
                    else:
                        # 画像なしの場合
                        text_embeddings = np.array(
                            self.embeddings_model.embed_documents(texts), 
                            dtype=np.float32
                        )
                            
                except Exception as e:
                    print(f"    ⚠️ 画像埋め込み生成エラー (バッチ {batch_idx}): {e}")
                    # エラー時はテキストのみ使用
                    text_embeddings = np.array(
                        self.embeddings_model.embed_documents(texts), 
                        dtype=np.float32
                    )
            else:
                # テキストのみ
                text_embeddings = np.array(
                    self.embeddings_model.embed_documents(texts), 
                    dtype=np.float32
                )
            
            all_embeddings.extend(text_embeddings)
            
            # 進行状況を表示
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"    バッチ {batch_idx}/{total_batches} 完了 ({len(all_embeddings)}/{len(documents)})")
        
        if self.use_multimodal and self.visual_encoder:
            print(f"  ✅ マルチモーダル埋め込み生成完了: {len(all_embeddings)}個")
        else:
            print(f"  ✅ テキスト埋め込み生成完了: {len(all_embeddings)}個")
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def faiss_kmeans(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        FAISSによる高速k-means
        
        Args:
            X: データ行列 (n_samples, n_features)
            k: クラスタ数
            
        Returns:
            labels: クラスタラベル
            centroids: クラスタ中心
        """
        n_samples, d = X.shape
        
        if n_samples < k:
            k = n_samples
        
        kmeans = faiss.Kmeans(
            d=d,
            k=k,
            niter=self.n_iter,
            verbose=False,
            gpu=False
        )
        
        kmeans.train(X)
        _, labels = kmeans.index.search(X, 1)
        
        return labels.ravel(), kmeans.centroids
    
    def compute_silhouette(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Silhouette Score を計算
        
        範囲: -1 〜 +1 (大きいほど良い)
        """
        n_clusters = len(np.unique(labels))
        
        if n_clusters <= 1 or n_clusters >= len(X):
            return -1.0
        
        try:
            return silhouette_score(X, labels, metric='euclidean')
        except:
            return -1.0
    
    def compute_dbi(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Davies-Bouldin Index を計算
        
        小さいほど良い
        """
        n_clusters = len(np.unique(labels))
        
        if n_clusters <= 1:
            return float('inf')
        
        try:
            return davies_bouldin_score(X, labels)
        except:
            return float('inf')
    
    def compute_chi(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calinski-Harabasz Index を計算
        
        大きいほど良い
        """
        n_clusters = len(np.unique(labels))
        
        if n_clusters <= 1 or n_clusters >= len(X):
            return 0.0
        
        try:
            return calinski_harabasz_score(X, labels)
        except:
            return 0.0
    
    def compute_bic(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """BIC計算（従来手法との比較用）"""
        n_samples, dim = X.shape
        k = len(np.unique(labels))
        
        distances = []
        for i in range(n_samples):
            cluster_id = labels[i]
            centroid = centroids[cluster_id]
            dist = np.linalg.norm(X[i] - centroid)
            distances.append(dist)
        
        distances = np.array(distances)
        variance = np.sum(distances ** 2) / (n_samples - k)
        
        penalty = k * np.log(n_samples) * dim
        likelihood = n_samples * np.log(variance + 1e-10)
        
        return penalty + likelihood
    
    def compute_aic(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """AIC計算（従来手法との比較用）"""
        n_samples, dim = X.shape
        k = len(np.unique(labels))
        
        distances = []
        for i in range(n_samples):
            cluster_id = labels[i]
            centroid = centroids[cluster_id]
            dist = np.linalg.norm(X[i] - centroid)
            distances.append(dist)
        
        distances = np.array(distances)
        variance = np.sum(distances ** 2) / (n_samples - k)
        
        penalty = 2 * k * dim
        likelihood = n_samples * np.log(variance + 1e-10)
        
        return penalty + likelihood
    
    def normalize_scores(self, scores: List[float], reverse: bool = False) -> np.ndarray:
        """
        スコアを0-1に正規化
        
        Args:
            scores: スコアリスト
            reverse: Trueの場合、大きい値が良いスコアを反転（小さい値が良い形式に）
        """
        scores = np.array(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score < 1e-10:
            return np.zeros_like(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        
        if reverse:
            normalized = 1.0 - normalized
        
        return normalized
    
    def select_optimal_k(self, X: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, Dict]:
        """
        複数の評価指標による最適クラスタ数選択
        
        Returns:
            best_k: 最適クラスタ数
            best_labels: クラスタラベル
            best_centroids: クラスタ中心
            scores_dict: 各指標のスコア
        """
        n_samples = X.shape[0]
        
        max_k = min(self.max_clusters, n_samples)
        min_k = min(self.min_clusters, max_k)
        
        print(f"\n🔍 Evaluating cluster count using {self.selection_strategy.upper()}...")
        print(f"   Range: {min_k} to {max_k} clusters")
        
        # 各kに対してクラスタリングと評価
        silhouette_scores = []
        dbi_scores = []
        chi_scores = []
        bic_scores = []
        aic_scores = []
        labels_list = []
        centroids_list = []
        
        for k in range(min_k, max_k + 1):
            labels, centroids = self.faiss_kmeans(X, k)
            labels_list.append(labels)
            centroids_list.append(centroids)
            
            # 各評価指標を計算
            sil = self.compute_silhouette(X, labels)
            dbi = self.compute_dbi(X, labels)
            chi = self.compute_chi(X, labels)
            bic = self.compute_bic(X, labels, centroids)
            aic = self.compute_aic(X, labels, centroids)
            
            silhouette_scores.append(sil)
            dbi_scores.append(dbi)
            chi_scores.append(chi)
            bic_scores.append(bic)
            aic_scores.append(aic)
            
            print(f"   k={k}: Sil={sil:.4f}, DBI={dbi:.4f}, CHI={chi:.2f}, BIC={bic:.2f}, AIC={aic:.2f}")
        
        # 戦略に応じて最適kを選択
        if self.selection_strategy == 'silhouette':
            best_idx = np.argmax(silhouette_scores)
            print(f"\n✅ Strategy: Silhouette Score (maximize)")
            print(f"   Selected k={min_k + best_idx} (Score={silhouette_scores[best_idx]:.4f})")
            
        elif self.selection_strategy == 'dbi':
            best_idx = np.argmin(dbi_scores)
            print(f"\n✅ Strategy: Davies-Bouldin Index (minimize)")
            print(f"   Selected k={min_k + best_idx} (Score={dbi_scores[best_idx]:.4f})")
            
        elif self.selection_strategy == 'chi':
            best_idx = np.argmax(chi_scores)
            print(f"\n✅ Strategy: Calinski-Harabasz Index (maximize)")
            print(f"   Selected k={min_k + best_idx} (Score={chi_scores[best_idx]:.2f})")
            
        elif self.selection_strategy == 'bic':
            best_idx = np.argmin(bic_scores)
            print(f"\n✅ Strategy: BIC (minimize)")
            print(f"   Selected k={min_k + best_idx} (Score={bic_scores[best_idx]:.2f})")
            
        elif self.selection_strategy == 'aic':
            best_idx = np.argmin(aic_scores)
            print(f"\n✅ Strategy: AIC (minimize)")
            print(f"   Selected k={min_k + best_idx} (Score={aic_scores[best_idx]:.2f})")
            
        elif self.selection_strategy == 'combined':
            # 正規化して組み合わせ
            sil_norm = self.normalize_scores(silhouette_scores, reverse=True)  # 大→小に反転
            dbi_norm = self.normalize_scores(dbi_scores, reverse=False)  # 小さい方が良い
            chi_norm = self.normalize_scores(chi_scores, reverse=True)  # 大→小に反転
            
            # 加重スコア（小さいほど良い形式に統一）
            combined = (
                self.metric_weights.get('silhouette', 0.4) * sil_norm +
                self.metric_weights.get('dbi', 0.3) * dbi_norm +
                self.metric_weights.get('chi', 0.3) * chi_norm
            )
            
            best_idx = np.argmin(combined)
            print(f"\n✅ Strategy: Combined Metrics")
            print(f"   Weights: {self.metric_weights}")
            print(f"   Selected k={min_k + best_idx}")
            print(f"   - Silhouette: {silhouette_scores[best_idx]:.4f}")
            print(f"   - DBI: {dbi_scores[best_idx]:.4f}")
            print(f"   - CHI: {chi_scores[best_idx]:.2f}")
            print(f"   - Combined score: {combined[best_idx]:.4f}")
        
        else:
            # デフォルトはSilhouette
            best_idx = np.argmax(silhouette_scores)
            print(f"\n⚠️  Unknown strategy, using Silhouette")
        
        best_k = min_k + best_idx
        best_labels = labels_list[best_idx]
        best_centroids = centroids_list[best_idx]
        
        scores_dict = {
            'silhouette': silhouette_scores,
            'dbi': dbi_scores,
            'chi': chi_scores,
            'bic': bic_scores,
            'aic': aic_scores,
            'k_range': list(range(min_k, max_k + 1))
        }
        
        # 統計記録
        self.stats['selections'].append(best_k)
        self.stats['silhouette_scores'].append(silhouette_scores[best_idx])
        self.stats['dbi_scores'].append(dbi_scores[best_idx])
        self.stats['chi_scores'].append(chi_scores[best_idx])
        
        return best_k, best_labels, best_centroids, scores_dict
    
    def cluster_documents(self, X: np.ndarray) -> Tuple[np.ndarray, int]:
        """ドキュメントをクラスタリング"""
        best_k, labels, centroids, scores = self.select_optimal_k(X)
        return labels, best_k
    
    def summarize_cluster(self, documents: List[Document]) -> str:
        """クラスタを要約（8000文字対応版）"""
        print(f"   🔄 Summarizing {len(documents)} documents...", end=" ", flush=True)
        
        import time
        start_time = time.time()
        
        # ドキュメントからテキストを抽出
        texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            elif hasattr(doc, 'cached_text'):
                texts.append(doc.cached_text)
            elif hasattr(doc, 'text_content'):
                texts.append(doc.text_content)
            else:
                texts.append(str(doc))
        
        # テキストを結合し、8000文字まで（高速化）
        combined_text = "\n\n".join(texts)
        max_input_length = 8000  # num_ctx=16384で余裕あり
        
        if len(combined_text) > max_input_length:
            # 均等にサンプリング
            sample_ratio = max_input_length / len(combined_text)
            sampled_texts = []
            for text in texts:
                sample_len = int(len(text) * sample_ratio)
                if sample_len > 0:
                    sampled_texts.append(text[:sample_len])
            combined_text = "\n\n".join(sampled_texts)[:max_input_length]
        
        # シンプルなプロンプトで要約
        prompt = ChatPromptTemplate.from_template(
            "以下は災害教訓に関する複数のドキュメントです。\n\n"
            "【要約タスク】\n"
            "- 主要な災害事例、教訓、対策を抽出してください\n"
            "- 300-500文字で簡潔にまとめてください\n"
            "- 箇条書きではなく、段落形式で記述してください\n\n"
            "【テキスト】\n{text}\n\n"
            "【要約】"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            summary = chain.invoke({"text": combined_text})
            elapsed = time.time() - start_time
            print(f"✅ ({len(summary)} chars, {elapsed:.1f}s)")
            return summary
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"⚠️ Error ({elapsed:.1f}s): {str(e)[:100]}")
            # エラー時は先頭1000文字を返す
            return combined_text[:1000]
    
    def _hierarchical_summarize(self, documents: List[Document]) -> str:
        """大量文書の階層的要約処理"""
        print(f"\n      📊 Hierarchical summarization for {len(documents)} docs...", end=" ", flush=True)
        
        # ドキュメントからテキストを抽出
        texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            elif hasattr(doc, 'cached_text'):
                texts.append(doc.cached_text)
            elif hasattr(doc, 'text_content'):
                texts.append(doc.text_content)
            else:
                texts.append(str(doc))
        
        # バッチサイズを動的に決定（文書数に応じて）
        if len(documents) > 200:
            batch_size = 20
        elif len(documents) > 100:
            batch_size = 15
        else:
            batch_size = 10
        
        print(f"\n      🔄 Processing in batches of {batch_size}...", end=" ", flush=True)
        
        # バッチ要約を実行
        batch_summaries = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_summary = self._summarize_batch(batch_texts, f"batch {i//batch_size + 1}")
            if batch_summary:
                batch_summaries.append(batch_summary)
            
            # プログレス表示
            if i % (batch_size * 5) == 0:
                progress = (i + batch_size) / len(texts) * 100
                print(f"\n      📈 Progress: {progress:.1f}% ({i + batch_size}/{len(texts)})", end=" ", flush=True)
        
        print(f"\n      🔗 Combining {len(batch_summaries)} batch summaries...", end=" ", flush=True)
        
        # バッチ要約を統合
        if len(batch_summaries) == 1:
            final_summary = batch_summaries[0]
        else:
            final_summary = self._combine_summaries(batch_summaries)
        
        print(f"✅ Done ({len(final_summary)} chars)")
        return final_summary
    
    def _standard_summarize(self, documents: List[Document]) -> str:
        """標準的な要約処理（50文書以下）"""
        # ドキュメントからテキストを抽出
        texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            elif hasattr(doc, 'cached_text'):
                texts.append(doc.cached_text)
            elif hasattr(doc, 'text_content'):
                texts.append(doc.text_content)
            else:
                texts.append(str(doc))
        
        # テキストを結合し、適切な長さに制限 (8000文字まで)
        combined_text = "\n\n".join(texts)
        max_input_length = 8000
        if len(combined_text) > max_input_length:
            # 均等にサンプリング
            sample_ratio = max_input_length / len(combined_text)
            sampled_texts = []
            for text in texts:
                sample_len = int(len(text) * sample_ratio)
                if sample_len > 0:
                    sampled_texts.append(text[:sample_len])
            combined_text = "\n\n".join(sampled_texts)[:max_input_length]
        
        # より具体的なプロンプト
        prompt = ChatPromptTemplate.from_template(
            "以下は災害教訓に関する複数のドキュメントです。\n\n"
            "【要約タスク】\n"
            "- 主要な災害事例、教訓、対策を抽出してください\n"
            "- 300-500文字で簡潔にまとめてください\n"
            "- 箇条書きではなく、段落形式で記述してください\n\n"
            "【テキスト】\n{text}\n\n"
            "【要約】"
        )
        
        return self._execute_summarization(prompt, combined_text)
    
    def _summarize_batch(self, texts: List[str], batch_label: str = "") -> str:
        """バッチテキストの要約"""
        # テキストを結合（最大8000文字まで）
        combined_text = "\n\n".join(texts)
        max_length = 8000
        if len(combined_text) > max_length:
            # 文書を均等にサンプリング
            sample_ratio = max_length / len(combined_text)
            sampled_texts = []
            for text in texts:
                sample_len = int(len(text) * sample_ratio)
                if sample_len > 0:
                    sampled_texts.append(text[:sample_len])
            combined_text = "\n\n".join(sampled_texts)[:max_length]
        
        # バッチ用プロンプト
        prompt = ChatPromptTemplate.from_template(
            "以下の災害関連文書を要約してください。\n\n"
            "【要約方針】\n"
            "- 重要な災害事例と教訓を抽出\n"
            "- 300-400文字で簡潔に\n"
            "- 段落形式で記述\n\n"
            "【文書】\n{text}\n\n"
            "【要約】"
        )
        
        return self._execute_summarization(prompt, combined_text, batch_label)
    
    def _combine_summaries(self, summaries: List[str]) -> str:
        """複数の要約を統合"""
        if not summaries:
            return "要約がありません。"
        
        if len(summaries) == 1:
            return summaries[0]
        
        # 要約を統合
        combined_summaries = "\n\n".join(summaries)
        
        # 統合用プロンプト
        prompt = ChatPromptTemplate.from_template(
            "以下は災害教訓に関する複数の要約です。これらを統合して、\n"
            "一つの包括的な要約（400-600文字）を作成してください。\n\n"
            "【統合方針】\n"
            "- 重複を避け、最も重要な内容を抽出\n"
            "- 災害の種類、時期、場所、教訓を明確に\n"
            "- 段落形式で論理的に構成\n\n"
            "【要約群】\n{summaries}\n\n"
            "【統合要約】"
        )
        
        return self._execute_summarization(prompt, combined_summaries, "final_integration")
    
    def _execute_summarization(self, prompt, text: str, context: str = "") -> str:
        """要約実行の共通処理"""
        chain = prompt | self.llm | StrOutputParser()
        
        # リトライ機能（最大3回試行、タイムアウト設定）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import time
                start_time = time.time()
                
                # プロンプトの入力変数を確認
                input_key = "summaries" if hasattr(prompt, 'input_variables') and "summaries" in prompt.input_variables else "text"
                
                # タイムアウト付きで実行 (30秒)
                summary = chain.invoke({input_key: text})
                
                elapsed = time.time() - start_time
                
                # 要約が短すぎる場合は警告
                if len(summary) < 50:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                
                if context:
                    print(f"      ✅ {context} done ({len(summary)} chars, {elapsed:.1f}s)", end=" ", flush=True)
                else:
                    print(f"✅ Done ({len(summary)} chars, {elapsed:.1f}s)")
                return summary
                
            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else 0
                if attempt < max_retries - 1:
                    if context:
                        print(f"      ⚠️ {context} retry {attempt + 1}...", end=" ", flush=True)
                    time.sleep(2)
                else:
                    if context:
                        print(f"      ❌ {context} failed ({elapsed:.1f}s)", end=" ", flush=True)
                    else:
                        print(f"❌ Failed after {max_retries} attempts ({elapsed:.1f}s)")
                    # フォールバック: 最初の1000文字を返す
                    fallback = text[:1000]
                    if len(fallback) < 1000:
                        fallback = text
                    return fallback + "..."
        
        # 万が一ここに到達した場合
        return text[:1000] + "..."
    
    def build_tree(self, documents: List[Document], depth: int = 0) -> Dict:
        """再帰的に階層ツリーを構築（最適化版）"""
        print(f"\n{'='*80}")
        print(f"Building tree at depth {depth} with {len(documents)} documents")
        
        # GPUメモリ使用量を表示（利用可能な場合）
        try:
            if hasattr(self, 'embeddings_model') and hasattr(self.embeddings_model, 'device'):
                import torch
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"🔧 GPU Memory: {memory_used:.2f}GB / {memory_total:.2f}GB ({memory_used/memory_total*100:.1f}%)")
        except:
            pass
        
        print(f"{'='*80}")
        
        if depth >= self.max_depth or len(documents) < self.min_clusters:
            print(f"✋ Reached max depth or insufficient documents. Creating leaf node.")
            return {
                'depth': depth,
                'documents': documents,
                'summaries': [],
                'clusters': {}
            }
        
        # 大量文書の場合は処理時間の警告
        if len(documents) > 200:
            print(f"⚠️  Large document set detected. This may take several minutes...")
            print(f"   💡 Tip: Consider increasing min_clusters to reduce tree depth")
        
        # Embedding
        embed_start = time.time()
        print(f"🔄 Generating embeddings for {len(documents)} documents...")
        embeddings = self.embed_documents(documents)
        embed_time = time.time() - embed_start
        print(f"⏱️  Embedding time: {embed_time:.2f}秒 ({embed_time/60:.1f}分)")
        
        # クラスタリング
        cluster_start = time.time()
        print(f"🔄 Clustering {len(documents)} documents...")
        labels, n_clusters = self.cluster_documents(embeddings)
        cluster_time = time.time() - cluster_start
        print(f"⏱️  Clustering time: {cluster_time:.2f}秒")
        print(f"📊 Generated {n_clusters} clusters")
        
        # クラスタサイズ統計
        cluster_sizes = {}
        for label in labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        
        avg_cluster_size = sum(cluster_sizes.values()) / len(cluster_sizes)
        max_cluster_size = max(cluster_sizes.values())
        min_cluster_size = min(cluster_sizes.values())
        print(f"📈 Cluster sizes - Avg: {avg_cluster_size:.1f}, Max: {max_cluster_size}, Min: {min_cluster_size}")
        
        # 大きなクラスタの警告
        large_clusters = [cid for cid, size in cluster_sizes.items() if size > 100]
        if large_clusters:
            print(f"⚠️  Large clusters detected: {large_clusters} (may take longer to summarize)")
        
        # クラスタごとに処理
        clusters = {}
        summaries = []
        
        summarize_start = time.time()
        for cluster_id in range(n_clusters):
            cluster_docs = [doc for i, doc in enumerate(documents) if labels[i] == cluster_id]
            print(f"\n📦 Cluster {cluster_id}: {len(cluster_docs)} documents")
            
            if len(cluster_docs) == 0:
                continue
            
            # 要約生成
            cluster_start_time = time.time()
            summary_text = self.summarize_cluster(cluster_docs)
            cluster_end_time = time.time()
            
            summary_doc = Document(
                page_content=summary_text,
                metadata={'cluster_id': cluster_id, 'depth': depth}
            )
            summaries.append(summary_doc)
            
            print(f"   ⏱️  Cluster {cluster_id} summary: {cluster_end_time - cluster_start_time:.1f}秒")
            
            # 再帰的に子ノードを構築（小さなクラスタのみ）
            if len(cluster_docs) >= self.min_clusters and depth < self.max_depth - 1:
                children = self.build_tree(cluster_docs, depth + 1)
            else:
                children = {
                    'depth': depth + 1,
                    'documents': cluster_docs,
                    'summaries': [],
                    'clusters': {}
                }
            
            clusters[cluster_id] = {
                'summary': summary_doc,
                'documents': cluster_docs,
                'children': children
            }
        
        total_summarize_time = time.time() - summarize_start
        print(f"\n⏱️  Total summarization time: {total_summarize_time:.1f}秒 ({total_summarize_time/60:.1f}分)")
        
        # メモリクリーンアップ
        try:
            import gc
            gc.collect()
            if hasattr(self, 'embeddings_model'):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("🧹 GPU cache cleared")
        except:
            pass
        
        return {
            'depth': depth,
            'documents': documents,
            'summaries': summaries,
            'clusters': clusters
        }
    
    def search_tree(self, tree: Dict, query: str, top_k: int = 5) -> List[Document]:
        """ツリーを検索"""
        if not tree or 'clusters' not in tree:
            return []
        
        # リーフノード
        if not tree['clusters']:
            docs = tree.get('documents', [])
            if not docs:
                return []
            
            query_embedding = np.array(self.embeddings_model.embed_query(query), dtype=np.float32)
            doc_embeddings = self.embed_documents(docs)
            
            similarities = np.dot(doc_embeddings, query_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                doc = docs[idx]
                doc.metadata['similarity'] = float(similarities[idx])
                results.append(doc)
            
            return results
        
        # 内部ノード
        summaries = tree.get('summaries', [])
        if not summaries:
            return []
        
        query_embedding = np.array(self.embeddings_model.embed_query(query), dtype=np.float32)
        summary_embeddings = self.embed_documents(summaries)
        
        similarities = np.dot(summary_embeddings, query_embedding) / (
            np.linalg.norm(summary_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        best_cluster_idx = np.argmax(similarities)
        cluster_id = summaries[best_cluster_idx].metadata['cluster_id']
        
        print(f"Selected cluster {cluster_id} at depth {tree['depth']} (similarity: {similarities[best_cluster_idx]:.4f})")
        
        best_cluster = tree['clusters'][cluster_id]
        return self.search_tree(best_cluster['children'], query, top_k)
    
    def index(self, file_path: str, encoding: str = "utf-8"):
        """ドキュメントをインデックス化"""
        print(f"\n{'='*80}")
        print(f"🚀 RAPTOR Indexing with {self.selection_strategy.upper()}")
        print(f"{'='*80}")
        print(f"📄 File: {file_path}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        documents = self.load_and_split_documents(file_path, encoding)
        self.tree_structure = self.build_tree(documents)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"✅ Indexing complete!")
        print(f"   Total time: {total_time:.2f}秒 ({int(total_time//60)}:{int(total_time%60):02d})")
        print(f"{'='*80}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """クエリを実行"""
        print(f"\n{'='*80}")
        print(f"🔍 Searching for: '{query}'")
        print(f"{'='*80}")
        
        results = self.search_tree(self.tree_structure, query, top_k)
        
        print(f"✅ Found {len(results)} results")
        
        return results
    
    def save(self, save_dir: str):
        """
        RAGモデルを保存（必要最小限）
        
        保存内容:
        1. tree_structure.json - ツリー構造（要約含む）
        2. stats.json - 統計情報
        3. config.json - 設定パラメータ
        
        Args:
            save_dir: 保存先ディレクトリ
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"💾 Saving RAPTOR model to: {save_dir}")
        print(f"{'='*80}")
        
        # 1. ツリー構造をJSON形式で保存
        tree_dict = self._tree_to_dict(self.tree_structure)
        with open(save_path / "tree_structure.json", "w", encoding="utf-8") as f:
            json.dump(tree_dict, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved tree_structure.json")
        
        # 2. 統計情報を保存（numpy型をPython標準型に変換）
        stats_serializable = self._make_serializable(self.stats)
        with open(save_path / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_serializable, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved stats.json")
        
        # 3. 設定パラメータを保存
        config = {
            'min_clusters': self.min_clusters,
            'max_clusters': self.max_clusters,
            'max_depth': self.max_depth,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'n_iter': self.n_iter,
            'selection_strategy': self.selection_strategy,
            'metric_weights': self.metric_weights
        }
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved config.json")
        
        # ファイルサイズを表示
        total_size = sum(f.stat().st_size for f in save_path.glob("*.json"))
        print(f"\n📊 Total size: {total_size / 1024:.2f} KB")
        print(f"{'='*80}")
    
    def _make_serializable(self, obj):
        """numpy型などをJSON serializable な形式に変換"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _tree_to_dict(self, node: Dict) -> Dict:
        """ツリー構造をJSON serializable な辞書に変換"""
        result = {
            'depth': node.get('depth', 0),
            'summaries': [],
            'documents': [],  # ドキュメント内容も保存
            'clusters': {}
        }
        
        # 要約を文字列化
        summaries = node.get('summaries', [])
        for summary in summaries:
            if isinstance(summary, Document):
                result['summaries'].append({
                    'content': summary.page_content,
                    'metadata': summary.metadata
                })
            else:
                result['summaries'].append({'content': str(summary), 'metadata': {}})
        
        # ドキュメント内容も保存（完全な検索のため）
        documents = node.get('documents', [])
        for doc in documents:
            if isinstance(doc, Document):
                result['documents'].append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
            else:
                result['documents'].append({'content': str(doc), 'metadata': {}})
        
        # 子ノードを再帰的に変換
        clusters = node.get('clusters', {})
        for cluster_id, cluster_data in clusters.items():
            result['clusters'][str(cluster_id)] = {
                'summary': self._doc_to_dict(cluster_data.get('summary')),
                'documents': [],  # 子ノードのdocumentsはchildrenに含まれる
                'children': self._tree_to_dict(cluster_data['children']) if 'children' in cluster_data else {}
            }
        
        return result
    
    def _doc_to_dict(self, doc) -> Optional[Dict]:
        """Documentオブジェクトを辞書に変換"""
        if doc is None:
            return None
        if isinstance(doc, Document):
            return {
                'content': doc.page_content,
                'metadata': doc.metadata
            }
        return {'content': str(doc), 'metadata': {}}
    
    @classmethod
    def load(cls, save_dir: str, embeddings_model, llm):
        """
        保存されたRAGモデルを読み込み
        
        Args:
            save_dir: 保存先ディレクトリ
            embeddings_model: Embeddings model
            llm: LLM for summarization
            
        Returns:
            RAPTORRetrieverEval インスタンス
        """
        save_path = Path(save_dir)
        
        print(f"\n{'='*80}")
        print(f"📂 Loading RAPTOR model from: {save_dir}")
        print(f"{'='*80}")
        
        # 設定を読み込み
        with open(save_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"✅ Loaded config.json")
        
        # インスタンスを作成
        instance = cls(
            embeddings_model=embeddings_model,
            llm=llm,
            **config
        )
        
        # ツリー構造を読み込み
        with open(save_path / "tree_structure.json", "r", encoding="utf-8") as f:
            tree_dict = json.load(f)
        instance.tree_structure = instance._dict_to_tree(tree_dict)
        print(f"✅ Loaded tree_structure.json")
        
        # 統計情報を読み込み
        with open(save_path / "stats.json", "r", encoding="utf-8") as f:
            instance.stats = json.load(f)
        print(f"✅ Loaded stats.json")
        
        print(f"{'='*80}")
        print(f"✅ Model loaded successfully!")
        print(f"{'='*80}")
        
        return instance
    
    def _dict_to_tree(self, tree_dict: Dict) -> Dict:
        """辞書からツリー構造を復元"""
        result = {
            'depth': tree_dict.get('depth', 0),
            'summaries': [],
            'documents': [],  # ドキュメントも復元
            'clusters': {}
        }
        
        # 要約をDocumentオブジェクトに復元
        for summary_dict in tree_dict.get('summaries', []):
            doc = Document(
                page_content=summary_dict['content'],
                metadata=summary_dict.get('metadata', {})
            )
            result['summaries'].append(doc)
        
        # ドキュメントをDocumentオブジェクトに復元
        for doc_dict in tree_dict.get('documents', []):
            doc = Document(
                page_content=doc_dict['content'],
                metadata=doc_dict.get('metadata', {})
            )
            result['documents'].append(doc)
        
        # 子ノードを再帰的に復元
        clusters = tree_dict.get('clusters', {})
        for cluster_id, cluster_data in clusters.items():
            summary_dict = cluster_data.get('summary')
            summary_doc = None
            if summary_dict:
                summary_doc = Document(
                    page_content=summary_dict['content'],
                    metadata=summary_dict.get('metadata', {})
                )
            
            # childrenが空でない場合のみ再帰的に復元
            children_data = cluster_data.get('children')
            if children_data and children_data.get('clusters'):
                # 内部ノード
                children = self._dict_to_tree(children_data)
            elif children_data:
                # リーフノード：childrenのdocumentsを持つ
                children = self._dict_to_tree(children_data)
            else:
                # childrenなし
                children = {}
            
            result['clusters'][int(cluster_id)] = {
                'summary': summary_doc,
                'documents': [],  # 子ノードのdocumentsはchildrenに含まれる
                'children': children
            }
        
        return result
