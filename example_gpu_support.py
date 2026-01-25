#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example demonstrating GPU support in LEKCut

This example shows how to use different execution providers
including GPU acceleration through CUDA and TensorRT.
"""

import onnxruntime as ort
from lekcut import word_tokenize

def main():
    # Display available execution providers
    print("Available ONNX Runtime Execution Providers:")
    providers = ort.get_available_providers()
    for i, provider in enumerate(providers, 1):
        print(f"  {i}. {provider}")
    print()
    
    # Example 1: Default execution (CPU)
    print("Example 1: Default execution (uses CPU)")
    text = "ทดสอบการตัดคำภาษาไทยด้วย LEKCut"
    result = word_tokenize(text)
    print(f"Input:  {text}")
    print(f"Output: {result}")
    print()
    
    # Example 2: Explicit CPU execution
    print("Example 2: Explicit CPU execution")
    result = word_tokenize(text, providers=['CPUExecutionProvider'])
    print(f"Input:  {text}")
    print(f"Output: {result}")
    print()
    
    # Example 3: GPU execution with CPU fallback
    # This will use CUDA if available, otherwise fall back to CPU
    print("Example 3: GPU execution (CUDA) with CPU fallback")
    result = word_tokenize(text, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"Input:  {text}")
    print(f"Output: {result}")
    if 'CUDAExecutionProvider' in providers:
        print("Note: Using CUDA GPU acceleration")
    else:
        print("Note: CUDA not available, using CPU fallback")
    print()
    
    # Example 4: TensorRT optimization (for NVIDIA GPUs)
    print("Example 4: TensorRT optimization with CUDA and CPU fallback")
    result = word_tokenize(
        text,
        providers=[
            'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
    )
    print(f"Input:  {text}")
    print(f"Output: {result}")
    if 'TensorrtExecutionProvider' in providers:
        print("Note: Using TensorRT optimization")
    elif 'CUDAExecutionProvider' in providers:
        print("Note: TensorRT not available, using CUDA")
    else:
        print("Note: GPU providers not available, using CPU")
    print()
    
    # Example 5: Multiple sentences
    print("Example 5: Processing multiple sentences")
    sentences = [
        "สวัสดีครับ",
        "ยินดีต้อนรับสู่ LEKCut",
        "ห้องสมุดสำหรับตัดคำภาษาไทย"
    ]
    print("Using GPU acceleration with CPU fallback:")
    for sentence in sentences:
        result = word_tokenize(sentence, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"  {sentence:40s} -> {result}")
    print()

if __name__ == "__main__":
    print("=" * 70)
    print("LEKCut GPU Support Examples")
    print("=" * 70)
    print()
    main()
    print("=" * 70)
    print("For more information, see:")
    print("  - ONNX Runtime docs: https://onnxruntime.ai/docs/execution-providers/")
    print("  - To enable GPU: pip install onnxruntime-gpu")
    print("=" * 70)
