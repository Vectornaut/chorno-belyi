# chorno-belyi
Code for drawing the ribbon graphs and Strebel differentials associated with algebraic curves

## Working languages

Python 3, Sage, GLSL.

## How to run `explorer.py`

### Install dependencies
1. Install [Vispy](http://vispy.org/installation.html) in your Python 3 distribution by calling `pip3 install vispy`.
2. Install Qt5 and Qt OpenGL.
  - Their Ubuntu 20.04 packages are `python3-pyqt5` and `python3-pyqt5.qtopengl`.
  - Install their Python bindings by calling `pip3 install PyQt5`.
    - If you're using Sage's internal Python distribution, as described below, call `sage -pip install PyQt5` instead.

### Run program
1. Call `python3 explorer.py`. A window with a tiling should appear.
  - If the interpreter complains that there's "`no module named 'sage.all'`", try using Sage's internal Python distribution by calling `sage -python explorer.py`.
    - If this also gives you trouble, call `sage -python --version` to make sure your Sage has Python 3 or above.
