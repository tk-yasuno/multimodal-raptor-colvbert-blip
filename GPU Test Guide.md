# GPU Test Guide
Visual RAPTOR ColBERT システムのGPU最適化ガイド

## 📋 概要

このガイドでは、Visual RAPTOR ColBERTシステムでのGPU使用方法、性能テスト、トラブルシューティングについて説明します。

## 🔧 GPU環境セットアップ

### 1. システム要件

- **GPU**: CUDA対応GPU（推奨：RTX 3060以上、8GB VRAM以上）
- **CUDA**: バージョン 12.1以上
- **Python**: 3.8-3.12
- **PyTorch**: CUDA版（2.5.1+cu121以上）

### 2. CUDA PyTorchインストール

```bash
# 既存のCPU版PyTorchをアンインストール
pip uninstall torch torchvision torchaudio -y

# CUDA版PyTorchをインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. GPU認識確認

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## 🚀 GPU性能テスト

### テスト実行方法

```bash
# GPU性能テストを実行
python test_gpu.py
```

### テスト内容

1. **GPU可用性チェック**
   - CUDA利用可能性
   - GPU情報表示
   - メモリ容量確認

2. **ColBERT GPU性能テスト**
   - テキストエンコード性能
   - 画像エンコード性能  
   - マルチモーダル統合性能
   - GPUメモリ使用量監視

3. **CPU vs GPU性能比較**
   - 処理速度比較
   - 効率性測定

## 📊 ベンチマーク結果（RTX 4060 Ti 16GB）

### エンコーダー選択オプション

システムでは2つのエンコーダーを選択可能：

1. **ColModernVBERTEncoder** (SigLIP使用) - 最新アーキテクチャ
2. **ColVBERTEncoder** (BLIP使用) - 従来アーキテクチャ

### ColVBERTEncoder（BLIP）性能指標

| 項目 | CPU | GPU | 速度向上 |
|------|-----|-----|----------|
| テキスト処理 | ~20 texts/sec | 130.1 texts/sec | **6.5x** |
| 画像処理 | ~2 images/sec | 31.4 images/sec | **15.7x** |
| マルチモーダル | ~2 pairs/sec | 39.3 pairs/sec | **19.7x** |
| **総合速度向上** | - | - | **17.7x** |

### ColModernVBERTEncoder（SigLIP）性能指標

| 項目 | CPU | GPU | 速度向上 |
|------|-----|-----|----------|
| テキスト処理 | ~7 texts/sec | 164.8 texts/sec | **23.5x** |
| 画像処理 | ~2 images/sec | 52.6 images/sec | **26.3x** |
| マルチモーダル | ~5 pairs/sec | 116.8 pairs/sec | **23.4x** |
| **総合速度向上** | - | - | **23.2x** |

### エンコーダー比較

| 特性 | ColVBERT (BLIP) | ColModernVBERT (SigLIP) |
|------|-----------------|------------------------|
| **速度** | 17.7x向上 | 23.2x向上 ⭐ |
| **メモリ使用量** | 9.3% (1.48GB) | 2.6% (0.41GB) ⭐ |
| **安定性** | ✅ 高い | ✅ 高い |
| **精度** | ✅ 確立済み | ⭐ 最新技術 |
| **互換性** | ✅ 幅広い | ⚠️ 新しい |

### 推奨使用ケース

#### ColVBERTEncoder（BLIP）を選択する場合：
- 🎯 **安定性重視**: 確立されたBLIPアーキテクチャ
- 🔧 **互換性重視**: 既存システムとの統合
- 📚 **実績重視**: 豊富な利用実績
- 💾 **VRAM余裕**: 16GB以上のGPU

#### ColModernVBERTEncoder（SigLIP）を選択する場合：
- ⚡ **最高性能**: 最速の処理速度
- 💾 **メモリ効率**: 限られたVRAM環境
- 🔬 **最新技術**: 最先端のマルチモーダル技術
- 🚀 **大規模処理**: 大量データの高速処理

### メモリ使用量比較

#### ColVBERTEncoder（BLIP）
- **GPUメモリ使用率**: 9.3% (1.48GB / 16GB)
- **初期化メモリ**: ~1.5GB VRAM
- **処理中ピーク**: ~2-3GB VRAM
- **推奨VRAM**: 8GB以上

#### ColModernVBERTEncoder（SigLIP）
- **GPUメモリ使用率**: 2.6% (0.41GB / 16GB)
- **初期化メモリ**: ~0.4GB VRAM
- **処理中ピーク**: ~1-2GB VRAM
- **推奨VRAM**: 4GB以上

### 実行速度詳細比較

#### テキスト処理性能
```
ColVBERT (BLIP):     130.1 texts/sec
ColModernVBERT:      164.8 texts/sec  (+26.7% faster)
```

#### 画像処理性能
```
ColVBERT (BLIP):     31.4 images/sec
ColModernVBERT:      52.6 images/sec  (+67.5% faster)
```

#### マルチモーダル処理性能
```
ColVBERT (BLIP):     39.3 pairs/sec
ColModernVBERT:      116.8 pairs/sec  (+197% faster)
```

## ⚙️ GPU最適化機能

### 1. 自動最適化

```python
# エンコーダー選択
# Option 1: ColVBERT (BLIP) - 安定性重視
encoder = ColVBERTEncoder(
    device="auto"  # 自動でGPU/CPU選択
)

# Option 2: ColModernVBERT (SigLIP) - 性能重視
encoder = ColModernVBERTEncoder(
    device="auto"  # 自動でGPU/CPU選択
)
```

### エンコーダー初期化例

```python
# ColVBERT（BLIP）- バランス型
colbert_encoder = ColVBERTEncoder(
    text_model_name="intfloat/multilingual-e5-large",
    vision_model_name="Salesforce/blip-image-captioning-base",
    embedding_dim=768,
    device="cuda"
)

# ColModernVBERT（SigLIP）- 高性能型
modern_encoder = ColModernVBERTEncoder(
    text_model_name="google/siglip-base-patch16-224",
    vision_model_name="google/siglip-base-patch16-224",
    embedding_dim=768,
    use_cross_attention=True,
    device="cuda"
)
```

### 2. バッチ処理最適化

```python
# ColVBERT推奨バッチサイズ
text_embeddings = colbert_encoder.encode_text(
    texts, 
    batch_size=32  # テキスト用（BLIPは少し控えめ）
)

image_embeddings = colbert_encoder.encode_image(
    images, 
    batch_size=16  # 画像用（メモリ使用量大）
)

# ColModernVBERT推奨バッチサイズ
text_embeddings = modern_encoder.encode_text(
    texts, 
    batch_size=64  # テキスト用（SigLIPは効率的）
)

image_embeddings = modern_encoder.encode_image(
    images, 
    batch_size=32  # 画像用（メモリ効率良い）
)
```

### 3. メモリ監視

```python
# GPU状態確認
gpu_info = encoder.get_gpu_memory_info()
print(f"GPU使用率: {gpu_info['utilization_percent']:.1f}%")
print(f"使用メモリ: {gpu_info['allocated_gb']:.2f} GB")

# キャッシュクリア
encoder.clear_gpu_cache()
```

## 🔧 最適化設定

### FP16混合精度

```python
# 自動でFP16が有効（GPU使用時）
encoder = ColModernVBERTEncoder()
# ✅ FP16混合精度が有効
# ✅ cuDNN最適化が有効
```

### バッチサイズ調整

| エンコーダー | データ型 | 推奨バッチサイズ | VRAM使用量目安 |
|-------------|---------|-----------------|---------------|
| **ColVBERT** | テキスト | 16-32 | ~2-3GB |
| **ColVBERT** | 画像 | 8-16 | ~4-6GB |
| **ColVBERT** | マルチモーダル | 8-16 | ~6-8GB |
| **ColModernVBERT** | テキスト | 32-64 | ~1-2GB |
| **ColModernVBERT** | 画像 | 16-32 | ~2-4GB |
| **ColModernVBERT** | マルチモーダル | 16-32 | ~4-6GB |

## 🚨 トラブルシューティング

### よくある問題と解決方法

#### 1. CUDA not available

**症状**: `torch.cuda.is_available()` が `False`

**解決方法**:
```bash
# CUDA版PyTorchを再インストール
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. GPU Out of Memory

**症状**: `CUDA out of memory` エラー

**解決方法**:
```python
# バッチサイズを削減
encoder.encode_text(texts, batch_size=8)  # デフォルト32→8

# メモリクリア
encoder.clear_gpu_cache()
```

#### 3. 低いGPU使用率

**症状**: GPU使用率が10%以下

**解決方法**:
```python
# バッチサイズを増加
encoder.encode_text(texts, batch_size=64)  # 32→64

# 並列処理を確認
torch.backends.cudnn.benchmark = True
```

#### 4. 処理が遅い

**解決方法**:
```python
# FP16が有効か確認
print(f"Device: {encoder.device}")
print(f"FP16 enabled: {encoder.device == 'cuda'}")

# cuDNN最適化確認
print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
```

## 📈 パフォーマンス監視

### GPU使用状況の確認

```bash
# システム全体のGPU監視
nvidia-smi

# 連続監視
nvidia-smi -l 1
```

### Python内での監視

```python
import torch

# メモリ使用量確認
allocated = torch.cuda.memory_allocated() / 1024**3
cached = torch.cuda.memory_reserved() / 1024**3
print(f"使用中: {allocated:.2f} GB")
print(f"キャッシュ: {cached:.2f} GB")

# GPU温度確認（nvidia-ml-pyが必要）
try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    print(f"GPU温度: {temp}°C")
except:
    pass
```

## 🎯 最適化のベストプラクティス

### GPU最適化のベストプラクティス

### 1. エンコーダー選択指針

```python
# メモリ制約がある場合（4-8GB VRAM）
encoder = ColModernVBERTEncoder()  # SigLIP - メモリ効率優先

# 安定性重視の場合（8GB+ VRAM）
encoder = ColVBERTEncoder()  # BLIP - 安定性優先

# 最高性能重視の場合（16GB+ VRAM）
encoder = ColModernVBERTEncoder(  # SigLIP - 性能優先
    use_cross_attention=True
)
```

### 2. バッチサイズ自動調整

```python
# GPU VRAMに応じた自動バッチサイズ調整
def get_optimal_batch_size(encoder_type: str, data_type: str, vram_gb: float):
    if encoder_type == "ColVBERT":
        if vram_gb >= 16:
            return {"text": 32, "image": 16, "multimodal": 16}
        elif vram_gb >= 8:
            return {"text": 16, "image": 8, "multimodal": 8}
        else:
            return {"text": 8, "image": 4, "multimodal": 4}
    
    elif encoder_type == "ColModernVBERT":
        if vram_gb >= 16:
            return {"text": 64, "image": 32, "multimodal": 32}
        elif vram_gb >= 8:
            return {"text": 32, "image": 16, "multimodal": 16}
        else:
            return {"text": 16, "image": 8, "multimodal": 8}

# 使用例
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
batch_sizes = get_optimal_batch_size("ColVBERT", "text", vram_gb)
```

### 2. エンコーダー別メモリ管理

```python
# ColVBERT用メモリ管理（メモリ使用量大）
def process_large_dataset_colbert(data, encoder):
    results = []
    
    for i, batch in enumerate(batches(data, batch_size=16)):  # 控えめバッチサイズ
        result = encoder.encode_text(batch)
        results.append(result.cpu())  # CPUに移動
        
        # 頻繁にキャッシュクリア（BLIPはメモリ使用量大）
        if i % 5 == 0:
            encoder.clear_gpu_cache()
    
    return torch.cat(results)

# ColModernVBERT用メモリ管理（メモリ効率良い）
def process_large_dataset_modern(data, encoder):
    results = []
    
    for i, batch in enumerate(batches(data, batch_size=32)):  # 大きめバッチサイズ
        result = encoder.encode_text(batch)
        results.append(result.cpu())  # CPUに移動
        
        # 少ない頻度でキャッシュクリア
        if i % 20 == 0:
            encoder.clear_gpu_cache()
    
    return torch.cat(results)
```

### 3. エンコーダー別エラーハンドリング

```python
def safe_gpu_encode(encoder, data, encoder_type="ColVBERT"):
    """エンコーダー別の安全なGPUエンコード"""
    try:
        if encoder_type == "ColVBERT":
            return encoder.encode_text(data, batch_size=16)
        else:  # ColModernVBERT
            return encoder.encode_text(data, batch_size=32)
            
    except torch.cuda.OutOfMemoryError:
        print("🚨 GPU Out of Memory - バッチサイズを削減中...")
        encoder.clear_gpu_cache()
        
        # エンコーダー別の fallback バッチサイズ
        fallback_batch = 8 if encoder_type == "ColVBERT" else 16
        batch_size = max(1, len(data) // 4, fallback_batch)
        
        return encoder.encode_text(data, batch_size=batch_size)
    
    except Exception as e:
        print(f"⚠️ エラー発生: {e}")
        # CPU fallback
        encoder_cpu = type(encoder)(device="cpu")
        return encoder_cpu.encode_text(data)
```

## 📋 GPU性能チェックリスト

### 基本要件
- [ ] CUDA対応GPU搭載
- [ ] CUDA版PyTorchインストール済み
- [ ] `torch.cuda.is_available()` が `True`
- [ ] GPU温度が85°C以下
- [ ] 十分なVRAM容量（推奨8GB以上）

### エンコーダー別チェック
#### ColVBERTEncoder（BLIP）
- [ ] 8GB以上のVRAM
- [ ] バッチサイズ: テキスト16-32, 画像8-16
- [ ] メモリ使用量9-10%以下（16GB GPU）
- [ ] 処理速度: 100+ texts/sec, 25+ images/sec

#### ColModernVBERTEncoder（SigLIP）
- [ ] 4GB以上のVRAM
- [ ] バッチサイズ: テキスト32-64, 画像16-32
- [ ] メモリ使用量3-5%以下（16GB GPU）
- [ ] 処理速度: 150+ texts/sec, 45+ images/sec

### 最適化機能
- [ ] FP16混合精度が有効
- [ ] cuDNN最適化が有効
- [ ] 適切なバッチサイズ設定
- [ ] メモリ監視機能動作
- [ ] 自動キャッシュクリア機能

## 🔗 関連リンク

- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/)

## 📞 サポート

問題が解決しない場合は、以下の情報を含めてお問い合わせください：

### 基本情報
1. `nvidia-smi` の出力
2. `torch.cuda.is_available()` の結果
3. PyTorchバージョン: `torch.__version__`
4. 使用しているGPUモデルとVRAM容量

### エンコーダー情報
5. 使用エンコーダー: `ColVBERTEncoder` または `ColModernVBERTEncoder`
6. 初期化パラメータ（テキスト/ビジョンモデル名）
7. 設定したバッチサイズ

### エラー情報
8. エラーメッセージの全文
9. GPU使用率: `encoder.get_gpu_memory_info()`
10. 処理データのサイズ（テキスト数、画像数）

### パフォーマンス情報
11. 実測処理速度（texts/sec, images/sec）
12. 期待される性能との差
13. メモリ使用量の変化

---

**最終更新**: 2025年10月25日  
**テスト環境**: 
- GPU: NVIDIA GeForce RTX 4060 Ti (16GB VRAM)
- ColVBERTEncoder: 17.7x速度向上, 9.3% VRAM使用
- ColModernVBERTEncoder: 23.2x速度向上, 2.6% VRAM使用