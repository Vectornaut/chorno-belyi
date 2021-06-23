

# This file was *autogenerated* from the file test-sage.sage
from sage.all_cmdline import *   # import sage library

_sage_const_0p3 = RealNumber('0.3'); _sage_const_0p2 = RealNumber('0.2'); _sage_const_2 = Integer(2); _sage_const_3 = Integer(3); _sage_const_9 = Integer(9); _sage_const_1 = Integer(1)
test_z = _sage_const_0p3  - _sage_const_0p2 *I
A = _sage_const_2 
B = _sage_const_3 
C = _sage_const_9 

print("values of series centered at 0")
print("hypergeometric parameters = %s, %s, %s" % (A,B,C))
h1_0 = hypergeometric([A, B], [C], t).series(t, prec)
h2_0 = hypergeometric([A-C+_sage_const_1 , B-C+_sage_const_1 ], [_sage_const_2 -C], t).series(t, prec)

