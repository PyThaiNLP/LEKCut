# LEKCut
<a href="https://pypi.python.org/pypi/lekcut"><img alt="pypi" src="https://img.shields.io/pypi/v/lekcut.svg"/></a>

LEKCut (เล็ก คัด) is a Thai tokenization library that ports the deep learning model to the onnx model.

## Install

> pip install lekcut

## How to use

```python
from lekcut import word_tokenize

# DeepCut model (default)
word_tokenize("ทดสอบการตัดคำ")
# output: ['ทดสอบ', 'การ', 'ตัด', 'คำ']

# AttaCut syllable + character model
word_tokenize("ทดสอบการตัดคำ", model="attacut-sc")
# output: ['ทดสอบ', 'การ', 'ตัด', 'คำ']

# AttaCut character-only model
word_tokenize("ทดสอบการตัดคำ", model="attacut-c")
# output: ['ทดสอบ', 'การ', 'ตัด', 'คำ']

# OSKut model
word_tokenize("เบียร์ยูไม่อร่อย", model="oskut")
# output: ['เบียร์', 'ยู', 'ไม่', 'อ', 'ร่อย']

# OSKut with a specific engine
word_tokenize("เบียร์ยูไม่อร่อย", model="oskut", engine="tnhc")
# output: ['เบียร์', 'ยู', 'ไม่', 'อร่อย']

# SEFR_CUT model
word_tokenize("เบียร์ยูไม่อร่อย", model="sefr-tnhc")
# output: ['เบียร์', 'ยู', 'ไม่', 'อร่อย']
```

**API**

```python
word_tokenize(
    text: str,
    model: str = "deepcut",
    path: str = "default",
    providers: List[str] = None,
    engine: str = "ws",
    k: int = 1,
) -> List[str]
```

**Parameters:**
- `text`: Text to tokenize
- `model`: Model to use. Options: `"deepcut"` (default), `"attacut-sc"`, `"attacut-c"`, `"oskut"`, `"sefr-best"`, `"sefr-tnhc"`, `"sefr-ws1000"`
- `path`: Path to custom model file (default: "default", applies to `deepcut` and `attacut-*` models)
- `providers`: List of ONNX Runtime execution providers (default: None, which uses default CPU provider)
- `engine`: OSKut engine variant (applies to `"oskut"` model only). Options: `"ws"` (default), `"ws-augment-60p"`, `"tnhc"`, `"scads"`, `"tl-deepcut-ws"`, `"tl-deepcut-tnhc"`, `"deepcut"`
- `k`: Percentage of characters to refine for OSKut (applies to `"oskut"` model only). The special default value of `1` is a sentinel that lets OSKut automatically select an appropriate percentage based on the engine. Pass any integer from 2 to 100 to override.

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
- ```attacut-sc``` - We ported the AttaCut syllable + character model from PyTorch to ONNX. The model and code come from [AttaCut's Github](https://github.com/PyThaiNLP/attacut). Requires the `ssg` package for syllable tokenization.
- ```attacut-c``` - We ported the AttaCut character-only model from PyTorch to ONNX. The model and code come from [AttaCut's Github](https://github.com/PyThaiNLP/attacut).
- ```oskut``` - We ported the OSKut (Out-of-domain Stacked Cut) stacked ensemble models from TensorFlow/Keras to ONNX. The model and code come from [OSKut's Github](https://github.com/mrpeerat/OSKut). Requires the `pyahocorasick` package. Supports multiple engines: `ws` (default), `ws-augment-60p`, `tnhc`, `scads`, `tl-deepcut-ws`, `tl-deepcut-tnhc`, `deepcut`.
- ```SEFR_CUT```- We ported the SEFR CUT (Stacked Ensemble Filter and Refine for Word Segmentation) model from PyTorch to ONNX. The model and code come from [SEFR_CUT's Github](https://github.com/mrpeerat/SEFR_CUT). List models: `"sefr-best"`, `"sefr-tnhc"`, `"sefr-ws1000"`

### Load custom model

If you have trained your custom model from deepcut or other that LEKCut support, You can load the custom model by ```path``` in ```word_tokenize``` after porting your model.

- How to train custom model with your dataset by deepcut - [Notebook](https://github.com/rkcosmos/deepcut/blob/master/notebooks/training.ipynb) (Needs to update ```deepcut/train.py``` before train model)

## How to porting model?

See ```notebooks/```
