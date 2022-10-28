# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

with open("README.md","r",encoding="utf-8-sig") as f:
    readme = f.read()

with open("requirements.txt","r",encoding="utf-8-sig") as f:
    requirements = [i.strip() for i in f.readlines()]

setup(
    name="LEKCut",
    version="0.1",
    description="LEKCut (เล็ก คัด) is a Thai tokenization library that ports the deep learning model to the onnx model.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Wannaphong Phatthiyaphaibun",
    author_email="wannaphong@yahoo.com",
    url="https://github.com/PyThaiNLP/LEKCut",
    packages=find_packages(),
    test_suite="tests",
    python_requires=">=3.6",
    package_data={
        "lekcut": [
            "model/*",
        ]
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords=[
        "thai",
        "NLP",
        "natural language processing",
        "text analytics",
        "text processing",
        "localization",
        "computational linguistics",
        "Thai language",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: General",
        "Topic :: Text Processing :: Linguistic",
    ],
    project_urls={
        "Source": "https://github.com/PyThaiNLP/LEKCut",
        "Bug Reports": "https://github.com/PyThaiNLP/LEKCut/issues",
    },
)