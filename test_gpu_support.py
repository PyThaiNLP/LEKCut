#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for GPU support in LEKCut
"""
import onnxruntime as ort
from lekcut import word_tokenize

def test_default_provider():
    """Test with default provider (should use CPU)"""
    print("Test 1: Default provider")
    result = word_tokenize("ทดสอบการตัดคำ")
    assert result == ['ทดสอบ', 'การ', 'ตัด', 'คำ'], f"Expected ['ทดสอบ', 'การ', 'ตัด', 'คำ'], got {result}"
    print("✓ Default provider test passed")
    print()

def test_explicit_cpu_provider():
    """Test with explicit CPU provider"""
    print("Test 2: Explicit CPU provider")
    result = word_tokenize("สวัสดีครับ", providers=['CPUExecutionProvider'])
    assert result == ['สวัสดี', 'ครับ'], f"Expected ['สวัสดี', 'ครับ'], got {result}"
    print("✓ Explicit CPU provider test passed")
    print()

def test_multiple_providers():
    """Test with multiple providers (CUDA + CPU fallback)"""
    print("Test 3: Multiple providers (CUDA with CPU fallback)")
    # This should work even without CUDA by falling back to CPU
    result = word_tokenize("ขอบคุณครับ", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    assert result == ['ขอบคุณ', 'ครับ'], f"Expected ['ขอบคุณ', 'ครับ'], got {result}"
    print("✓ Multiple providers test passed")
    print()

def test_provider_switching():
    """Test switching between different providers"""
    print("Test 4: Provider switching")
    result1 = word_tokenize("ทดสอบ")
    result2 = word_tokenize("การตัด", providers=['CPUExecutionProvider'])
    result3 = word_tokenize("คำภาษาไทย")
    assert result1 == ['ทดสอบ']
    assert result2 == ['การ', 'ตัด']
    assert result3 == ['คำ', 'ภาษา', 'ไทย']
    print("✓ Provider switching test passed")
    print()

def test_available_providers():
    """Display available ONNX Runtime providers"""
    print("Available ONNX Runtime providers:")
    providers = ort.get_available_providers()
    for provider in providers:
        print(f"  - {provider}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("LEKCut GPU Support Test Suite")
    print("=" * 60)
    print()
    
    test_available_providers()
    test_default_provider()
    test_explicit_cpu_provider()
    test_multiple_providers()
    test_provider_switching()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
