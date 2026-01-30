# LEKCut
<a href="https://pypi.python.org/pypi/lekcut"><img alt="pypi" src="https://img.shields.io/pypi/v/lekcut.svg"/></a>

LEKCut (เล็ก คัด) is a Thai tokenization library that ports the deep learning model to the onnx model.

## Install

> pip install lekcut

## How to use

```python
from lekcut import word_tokenize
word_tokenize("ทดสอบการตัดคำ")
# output: ['ทดสอบ', 'การ', 'ตัด', 'คำ']
```

**API**

```python
word_tokenize(text: str, model: str="deepcut", path: str="default", providers: List[str]=None) -> List[str]
```

**Parameters:**
- `text`: Text to tokenize
- `model`: Model to use (default: "deepcut")
- `path`: Path to custom model file (default: "default")
- `providers`: List of ONNX Runtime execution providers (default: None, which uses default CPU provider)

### GPU Support

LEKCut supports GPU acceleration through ONNX Runtime execution providers. To use GPU acceleration:

1. Install ONNX Runtime with GPU support:
   ```bash
   pip install onnxruntime-gpu
   ```

2. Use the `providers` parameter to specify GPU execution:
   ```python
   from lekcut import word_tokenize
   
   # Use CUDA GPU
   result = word_tokenize("ทดสอบการตัดคำ", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
   
   # Use TensorRT (if available)
   result = word_tokenize("ทดสอบการตัดคำ", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
   ```

**Available Execution Providers:**
- `CPUExecutionProvider` - Default CPU execution
- `CUDAExecutionProvider` - NVIDIA CUDA GPU acceleration
- `TensorrtExecutionProvider` - NVIDIA TensorRT optimization
- `DmlExecutionProvider` - DirectML for Windows GPU
- And more (see [ONNX Runtime documentation](https://onnxruntime.ai/docs/execution-providers/))

**Note:** The providers are tried in order, and the first available one will be used. Always include `CPUExecutionProvider` as a fallback.

## Model
- ```deepcut``` - We ported deepcut model from tensorflow.keras to ONNX model. The model and code come from [Deepcut's Github](https://github.com/rkcosmos/deepcut). The model is [here](https://github.com/PyThaiNLP/LEKCut/blob/main/lekcut/model/deepcut.onnx).

### Load custom model

If you have trained your custom model from deepcut or other that LEKCut support, You can load the custom model by ```path``` in ```word_tokenize``` after porting your model.

- How to train custom model with your dataset by deepcut - [Notebook](https://github.com/rkcosmos/deepcut/blob/master/notebooks/training.ipynb) (Needs to update ```deepcut/train.py``` before train model)

## How to porting model?

See ```notebooks/```
