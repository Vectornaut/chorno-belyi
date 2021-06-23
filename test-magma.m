CC<I> := ComplexField();
test_z := 0.3 - 0.2*I;
//A := 2; B := 3; C := 9;
p := 2;
q := 3;
r := 9;

A := (1/2)*(1/p - 1/q - 1/r + 1);
B := (1/2)*(1/p - 1/q + 1/r + 1);
C := 1 + 1/p;

printf "test input = %o\n", test_z;
print "values of series centered at 0";
/*
  h1_0 = hypergeometric([A, B], [C], t).series(t, prec)
  h2_0 = hypergeometric([A-C+1, B-C+1], [2-C], t).series(t, prec)
*/
printf "hypergeometric parameters = %o, %o, %o\n", A,B,C;
HypergeometricSeries2F1(A,B,C,test_z);
printf "hypergeometric parameters = %o, %o, %o\n", A-C+1, B-C+1, 2-C;
HypergeometricSeries2F1(A-C+1, B-C+1, 2-C, test_z);

print "values of series centered at 1";
test_z_1 := 1 - test_z;
/*
  h1_1 = hypergeometric([C-A, C-B], [1-A-B+C], t).series(t, prec) # y_6
  h2_1 = hypergeometric([  A,   B], [1+A+B-C], t).series(t, prec) # y_5
*/
printf "hypergeometric parameters = %o, %o, %o\n", C-A, C-B, 1-A-B+C;
HypergeometricSeries2F1(C-A, C-B, 1-A-B+C, test_z_1);
printf "hypergeometric parameters = %o, %o, %o\n", A, B, 1+A+B-C;
HypergeometricSeries2F1(A, B, 1+A+B-C, test_z_1);


test_z := 1.5+0.1*I;
test_z_1 := 1 - test_z;
printf "test input = %o\n", test_z;
print "values of series centered at 1";
/*
  h1_1 = hypergeometric([C-A, C-B], [1-A-B+C], t).series(t, prec) # y_6
  h2_1 = hypergeometric([  A,   B], [1+A+B-C], t).series(t, prec) # y_5
*/
printf "hypergeometric parameters = %o, %o, %o\n", C-A, C-B, 1-A-B+C;
HypergeometricSeries2F1(C-A, C-B, 1-A-B+C, test_z_1);
printf "hypergeometric parameters = %o, %o, %o\n", A, B, 1+A+B-C;
HypergeometricSeries2F1(A, B, 1+A+B-C, test_z_1);

print "values of series centered at oo";
test_z_oo := 1/test_z;
/*
  h1_inf = hypergeometric([B, 1+B-C], [1+B-A], t).series(t, prec) # y_10
  h2_inf = hypergeometric([A, 1+A-C], [1+A-B], t).series(t, prec) # y_9
*/
printf "hypergeometric parameters = %o, %o, %o\n", B, 1+B-C, 1+B-A;
HypergeometricSeries2F1(B, 1+B-C, 1+B-A, test_z_oo);
printf "hypergeometric parameters = %o, %o, %o\n", A, 1+A-C, 1+A-B;
HypergeometricSeries2F1(A, 1+A-C, 1+A-B, test_z_oo);
