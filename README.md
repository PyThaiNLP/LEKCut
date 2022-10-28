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

**API**

```python
word_tokenize(text: str, model: str="deepcut", path: str="default") -> List[str]
```

## Model
- ```deepcut``` - We ported deepcut model from tensorflow.keras to ONNX model. The model and code come from [Deepcut's Github](https://github.com/rkcosmos/deepcut). The model is [here](https://github.com/PyThaiNLP/LEKCut/blob/main/lekcut/model/deepcut.onnx).

### Load custom model

If you has trained custom your model from deepcut or other that LEKCut support, You can load the custom model by ```path``` in ```word_tokenize``` after porting your model.

- How to train custom model ith your dataset by deepcut - [Notebook](https://github.com/rkcosmos/deepcut/blob/master/notebooks/training.ipynb) (Needs to update ```deepcut/train.py``` before train model)

## How to porting model?

See ```notebooks/```