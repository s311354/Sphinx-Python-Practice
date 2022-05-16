# Sphinx Python Practice

## Introduction

This repository would be useful as a reference for documentation with Python 3 and Sphinx. There is also a short [blog post]() on this on my website.

It is strongly recommended to watch Brandon Rhodes's [Sphinx tutorial session](https://www.youtube.com/watch?v=QNHM7q2hLh8) at PyCon 2013 and [leimao's blog post](https://leimao.github.io/blog/Python-Documentation-Using-Sphinx/).

## Files

```
.
├── Makefile
├── README.md
├── build
├── doc
│   ├── Makefile
│   ├── README.md
│   ├── __init__.py
│   ├── build
│   ├── make.bat
│   └── source
│       ├── _static
│       ├── _templates
│       ├── api.rst
│       ├── conf.py
│       ├── guide.rst
│       └── index.rst
├── leetcode
│   ├── __init__.py
│   ├── impl
│   │   └── solution.py
│   └── mocktest.py
├── make.bat
├── requirements.txt
└── source
    ├── _static
    ├── _templates
    ├── conf.py
    └── index.rst
```

## Installation

The `leetcode` folder used for the tutorial has no dependency. But for completeness, we added `setuptools` and `sphinx` to our `requirements.txt`. 

To install the dependencies, please run the following command in the terminal.

```
$ pip install -r requirements.txt
```

To install the `leetcode`, which we would not probably do, please run the following command in the terminal.
```
$ pip install .
```

## Build Documentations

Please check the [`README`](doc/README.md) in [`doc`](doc/) for details.


## References
* [Lei Mao's Sphinx Tutorial](https://github.com/leimao/Sphinx-Python-TriangleLib)
* [Brandon Rhodes's Sphinx Tutorial Session at PyCon 2013](https://www.youtube.com/watch?v=QNHM7q2hLh8)
* [Python Documentation Using Sphinx](https://leimao.github.io/blog/Python-Documentation-Using-Sphinx/)
* [reStructuredText Markup Specification](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html)
