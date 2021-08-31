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

## How to run `puzzlemaker.py`

### Install dependencies
1. Install Vispy, as described above.
2. Install Django by calling `pip3 install django`.

### Run program
1. [*Optional.*] Do a dry run by calling `python3 puzzlemaker.py --dry-run`. The program should print 20 passports, with each one's dessins listed underneath. You can change the maximum number of passports listed with the option `-n N_MAX`, where *N_MAX* is a number.
2. Render puzzle pages and dessins by calling `python3 puzzlemaker.py`. For each passport listed in the dry run, the program should create a directory in `docs/` and stock it with a puzzle page `index.html` and an image of each dessin. It should also make the puzzle list `docs/puzzles.html`.
3. [*As needed.*] Use the `--no-pics` option to quickly remake the puzzle pages and the puzzle list without redrawing the pictures. (Drawing the pictures can take a while, because the `Dessin` constructor is weirdly slow.)
