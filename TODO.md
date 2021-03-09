1. Take point in fundamental domain for Delta and return point in CP^1
   1. Figure out disc of convergence of series for phi
   2. ~~Python code that gives list of series coefficients (for possibly multiple series) to be passed to shader~~
   3. *(Low priority.)* Get the series around infinity working?
2. ~~Drawing code for leaves of quadratic differential~~
3. Modify coset representative code
   1. Package original code to make it more usable
   2. Figure out how to reassemble the fundamental domain to put all the vertices inside
4. Dessin view application
   1. **Aaron**—Allow each half-triangle in a triangle to have its own boundary.
   2. **Aaron**—Add [image export](https://stackoverflow.com/a/45101563/1644283)
   3. Clean up colors
   4. **Aaron**—Antialias dessins
   5. **Aaron**—Add viewpoint transformations
   6. Polishing
      1. For orders 3, 6, 4, we exceed maximum reflection iterations around edges.
5. *(Long-term test application.)* Take a bunch of Belyi maps with the same passport, draw dessins for all of them, and look for similarities between the dessins of the Galois-conjugate maps
   1. Draw fundamental domains for:
      1. ~~[Passport 7T6-7_5.1.1_5.1.1](https://beta.lmfdb.org/Belyi/7T6/7/5.1.1/5.1.1/)~~
      2. [Passport 7T6-5.1.1_4.2.1_3.2.2](https://beta.lmfdb.org/Belyi/7T6/5.1.1/4.2.1/3.2.2/)
      3. **Mike**—[Passport 7T6-7_7_3.2.2/](https://beta.lmfdb.org/Belyi/7T6/7/7/3.2.2/)
   2. Lay out web page for potential presentation at Bridges
      ~1. Learn about Javascript drag-and-drop for sort and check game mechanic~
      ~2. Mock up landing page~
      3. Realize landing page in HTML / CSS / JavaScript.
      4. Keep writing [copy](https://www.overleaf.com/project/602d325ad20f545c8b297465).
   3. Make images to populate web page
      1. Color 0 white and 1 black.
      1. Describe domain-finding algorithm, at least informally (to standardize our work).
