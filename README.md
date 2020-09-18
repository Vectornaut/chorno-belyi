# chorno-belyi
Code for drawing the ribbon graphs and Strebel differentials associated with algebraic curves

## Working languages

Python 3 and Sage, maybe calling GLSL (through Vispy) for real-time visualization.

## How to run `tile_vispy.py`

### Install dependencies
1. Install [Vispy](http://vispy.org/installation.html) in your Python 3 distribution by calling `pip3 install vispy`.
2. Give your machine a backend window manager for Vispy to use. I'm using GLFW; its Ubuntu 18.04 packages are `libglfw3` and `libglfw3-dev` (I'm not sure the dev package is necessary). You can also use other window managers, listed at the link above.
### Run program
1. Call `python3 tile_vispy.py`. A window with a tiling should appear. I don't recommend leaving it open in the background, because it's working your GPU pretty hard, rendering the tiling in realtime.

## How to run `tile.py`

### Install dependencies
1. Install [Pillow](https://pillow.readthedocs.io/en/stable/installation.html) in your Python 3 distribution by calling `pip3 install Pillow` (unless you have PIL installed; you can check by calling `pip3 show PIL`).
### Run program
1. Go to the folder where you want output to appear.
1. Launch the REPL by calling `python3`.
2. Call `import tile`
3. Call `tile.paint(2, 3, 7)`. The arguments are the vertex orders; change as desired. After a few seconds, you'll get an output file called `tiling.png`.
