#!/usr/bin/env python3
"""
GPU動作テストスクリプト
Visual RAPTOR ColBERT システムのGPU性能を検証

使用方法:
python test_gpu.py
"""

import torch
import time
from PIL import Image
import numpy as np
from pathlib import Path

# ローカルモジュールをインポート
from visual_raptor_colbert import ColVBERTEncoder

def test_gpu_availability():
    """GPU利用可能性をテスト"""
    print("=" * 60)
    print("🔍 GPU可用性チェック")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    
    print()

def test_colbert_gpu():
    """ColBERT エンコーダーのGPU性能をテスト"""
    print("=" * 60)
    print("🚀 ColBERT GPU性能テスト")
    print("=" * 60)
    
    # エンコーダー初期化
    encoder = ColVBERTEncoder(
        text_model_name="intfloat/multilingual-e5-large",
        vision_model_name="Salesforce/blip-image-captioning-base", 
        embedding_dim=768,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\n📊 初期GPU状態:")
    gpu_info = encoder.get_gpu_memory_info()
    for key, value in gpu_info.items():
        if key.endswith('_gb'):
            print(f"  {key}: {value:.2f} GB")
        elif key.endswith('_percent'):
            print(f"  {key}: {value:.1f}%")
    
    # テストデータ準備
    test_texts = [
        "津波避難経路の確認",
        "災害時の情報収集方法", 
        "緊急避難所の場所",
        "防災グッズの準備",
        "家族との連絡手段"
    ] * 10  # 50テキスト
    
    # テスト画像作成
    test_images = []
    for i in range(20):
        # ランダムな色の画像を生成
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_images.append(Image.fromarray(img_array))
    
    print(f"\n🔥 性能テスト実行:")
    print(f"  テキスト数: {len(test_texts)}")
    print(f"  画像数: {len(test_images)}")
    
    # テキストエンコード性能テスト
    start_time = time.time()
    text_embeddings = encoder.encode_text(test_texts, batch_size=16)
    text_time = time.time() - start_time
    
    print(f"\n📝 テキストエンコード結果:")
    print(f"  処理時間: {text_time:.2f}秒")
    print(f"  スループット: {len(test_texts)/text_time:.1f} texts/sec")
    print(f"  出力形状: {text_embeddings.shape}")
    print(f"  デバイス: {text_embeddings.device}")
    
    # GPU状態確認
    gpu_info = encoder.get_gpu_memory_info()
    print(f"  GPU使用率: {gpu_info.get('utilization_percent', 0):.1f}%")
    
    # 画像エンコード性能テスト
    start_time = time.time()
    image_embeddings = encoder.encode_image(test_images, batch_size=8)
    image_time = time.time() - start_time
    
    print(f"\n🖼️ 画像エンコード結果:")
    print(f"  処理時間: {image_time:.2f}秒")
    print(f"  スループット: {len(test_images)/image_time:.1f} images/sec")
    print(f"  出力形状: {image_embeddings.shape}")
    print(f"  デバイス: {image_embeddings.device}")
    
    # マルチモーダルテスト
    start_time = time.time()
    multimodal_embeddings = encoder.encode_multimodal(
        test_texts[:len(test_images)], 
        test_images
    )
    multimodal_time = time.time() - start_time
    
    print(f"\n🔗 マルチモーダルエンコード結果:")
    print(f"  処理時間: {multimodal_time:.2f}秒")
    print(f"  スループット: {len(test_images)/multimodal_time:.1f} pairs/sec")
    print(f"  出力形状: {multimodal_embeddings.shape}")
    
    # 最終GPU状態
    print(f"\n📊 最終GPU状態:")
    gpu_info = encoder.get_gpu_memory_info()
    for key, value in gpu_info.items():
        if key.endswith('_gb'):
            print(f"  {key}: {value:.2f} GB")
        elif key.endswith('_percent'):
            print(f"  {key}: {value:.1f}%")
    
    # GPU キャッシュクリア
    encoder.clear_gpu_cache()
    
    print(f"\n✅ GPU性能テスト完了!")

def benchmark_cpu_vs_gpu():
    """CPU vs GPU性能比較"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping CPU vs GPU benchmark")
        return
    
    print("=" * 60)
    print("⚡ CPU vs GPU 性能比較")
    print("=" * 60)
    
    test_texts = ["テスト用テキスト"] * 100
    test_images = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(50)]
    
    # CPU テスト
    print("🖥️ CPU性能測定...")
    cpu_encoder = ColVBERTEncoder(device="cpu")
    
    start_time = time.time()
    cpu_embeddings = cpu_encoder.encode_text(test_texts[:20])  # 少数で測定
    cpu_time = time.time() - start_time
    
    # GPU テスト
    print("🚀 GPU性能測定...")
    gpu_encoder = ColVBERTEncoder(device="cuda")
    
    start_time = time.time()
    gpu_embeddings = gpu_encoder.encode_text(test_texts[:20])
    gpu_time = time.time() - start_time
    
    print(f"\n📊 性能比較結果:")
    print(f"  CPU処理時間: {cpu_time:.2f}秒")
    print(f"  GPU処理時間: {gpu_time:.2f}秒")
    print(f"  速度向上: {cpu_time/gpu_time:.1f}x")
    print(f"  GPU効率: {((cpu_time - gpu_time)/cpu_time)*100:.1f}% faster")

def main():
    """メイン実行関数"""
    print("🔥 Visual RAPTOR ColBERT GPU Test Suite")
    print("=" * 60)
    
    test_gpu_availability()
    
    if torch.cuda.is_available():
        test_colbert_gpu()
        benchmark_cpu_vs_gpu()
    else:
        print("❌ CUDA not available. Please install CUDA-compatible PyTorch.")
        print("   Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    main()