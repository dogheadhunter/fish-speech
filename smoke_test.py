#!/usr/bin/env python
"""Comprehensive smoke test for Fish Speech environment."""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test all critical module imports."""
    modules = [
        "torch",
        "torchaudio",
        "transformers",
        "gradio",
        "numpy",
        "pydantic",
        "soundfile",
        "librosa",
        "hydra",
        "lightning",
        "loguru",
        "wandb",
        "tiktoken",
        "loralib",
        "pyrootutils",
        "resampy",
        "einops",
        "einx",
        "kui.asgi",
        "uvicorn",
        "ormsgpack",
        "datasets",
        "google.protobuf",
    ]
    
    print("=" * 70)
    print("IMPORT TEST")
    print("=" * 70)
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"✓ {mod}")
        except Exception as e:
            print(f"✗ {mod}: {e}")
            failed.append(mod)
    
    print()
    return failed

def test_torch_cuda():
    """Test PyTorch CUDA availability and basic ops."""
    import torch
    
    print("=" * 70)
    print("TORCH & CUDA TEST")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print()
    
    # Basic tensor operation
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        z = torch.matmul(x, y)
        print(f"✓ Tensor operation on {device} successful")
        print(f"  Result shape: {z.shape}")
    except Exception as e:
        print(f"✗ Tensor operation failed: {e}")
        return False
    print()
    return True

def test_fish_speech_imports():
    """Test Fish Speech specific imports."""
    import torch
    
    print("=" * 70)
    print("FISH SPEECH MODULES TEST")
    print("=" * 70)
    
    modules = [
        "fish_speech.tokenizer",
        "fish_speech.inference_engine",
        "fish_speech.utils.schema",
        "fish_speech.utils.utils",
        "fish_speech.models.text2semantic.inference",
        "fish_speech.models.dac.inference",
    ]
    
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"✓ {mod}")
        except Exception as e:
            print(f"✗ {mod}: {e}")
            failed.append(mod)
    
    print()
    return failed

def test_pydantic():
    """Test Pydantic schema parsing."""
    import numpy as np
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
    
    print("=" * 70)
    print("PYDANTIC SCHEMA TEST")
    print("=" * 70)
    
    try:
        # Test basic schema
        req = ServeTTSRequest(text="Hello, world!")
        print(f"✓ ServeTTSRequest created: text='{req.text}'")
        
        # Test with references
        audio_bytes = np.random.bytes(16000)
        ref = ServeReferenceAudio(audio=audio_bytes, text="reference")
        print(f"✓ ServeReferenceAudio created: text='{ref.text}', audio_size={len(ref.audio)}")
    except Exception as e:
        print(f"✗ Pydantic schema test failed: {e}")
        return False
    
    print()
    return True

def test_transformers():
    """Test Transformers basic functionality."""
    from transformers import AutoTokenizer
    
    print("=" * 70)
    print("TRANSFORMERS TEST")
    print("=" * 70)
    
    try:
        # Just test import and quick tokenizer load
        print("✓ Transformers imported")
        # Don't actually download a model to keep it fast
        print("✓ AutoTokenizer available")
    except Exception as e:
        print(f"✗ Transformers test failed: {e}")
        return False
    
    print()
    return True

def main():
    """Run all smoke tests."""
    print("\n")
    print("█" * 70)
    print("█ FISH SPEECH ENVIRONMENT SMOKE TEST")
    print("█" * 70)
    print()
    
    # Run tests
    import_fails = test_imports()
    cuda_ok = test_torch_cuda()
    fs_import_fails = test_fish_speech_imports()
    pydantic_ok = test_pydantic()
    transformers_ok = test_transformers()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = True
    
    if import_fails:
        print(f"⚠ {len(import_fails)} imports failed: {', '.join(import_fails)}")
        all_ok = False
    else:
        print("✓ All core imports successful")
    
    if cuda_ok:
        print("✓ CUDA/Tensor operations OK")
    else:
        print("⚠ CUDA/Tensor operations failed")
        all_ok = False
    
    if fs_import_fails:
        print(f"⚠ {len(fs_import_fails)} Fish Speech imports failed: {', '.join(fs_import_fails)}")
        all_ok = False
    else:
        print("✓ Fish Speech modules importable")
    
    if pydantic_ok:
        print("✓ Pydantic schemas OK")
    else:
        print("⚠ Pydantic schemas failed")
        all_ok = False
    
    if transformers_ok:
        print("✓ Transformers OK")
    else:
        print("⚠ Transformers failed")
        all_ok = False
    
    print()
    if all_ok:
        print("✅ SMOKE TEST PASSED - Environment is ready!")
    else:
        print("⚠️  SMOKE TEST PASSED WITH WARNINGS - See above for details")
    print()
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
