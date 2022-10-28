# LEKCut
<a href="https://pypi.python.org/pypi/lekcut"><img alt="pypi" src="https://img.shields.io/pypi/v/lekcut.svg"/></a>

LEKCut (เล็ก คัด) is a Thai tokenization library that porting deep learning model to onnx model.

## Install

> pip install lekcut

## How to use

```python
from lekcut import word_tokenize
word_tokenize("ทดสอบการตัดคำ")
# output: ['ทดสอบ', 'การ', 'ตัด', 'คำ']
```

**Model**
- ```deepcut``` - We ported deepcut model from tensorflow.keras to ONNX model. The model and code come from [Deepcut's Github](https://github.com/rkcosmos/deepcut).

## How to porting model?

See ```noebooks/```