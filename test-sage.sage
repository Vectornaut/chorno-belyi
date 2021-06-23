test_z = 0.3 - 0.2*I
A = 2
B = 3
C = 9
t = polygen(QQ, 't')
prec = 30

print("test input value = %s" % test_z)

print("values of series centered at 0")
print("hypergeometric parameters = %s, %s, %s" % (A,B,C))
h1_0 = hypergeometric([A, B], [C], t).series(t, prec)
print(h1_0(t=test_z))
print("hypergeometric parameters = %s, %s, %s" % (A-C+1, B-C+1, 2-C))
h2_0 = hypergeometric([A-C+1, B-C+1], [2-C], t).series(t, prec)
print(h2_0(t=test_z))

print("values of series centered at 1")
print("hypergeometric parameters = %s, %s, %s" % (C-A, C-B, 1-A-B+C))
h1_1 = hypergeometric([C-A, C-B], [1-A-B+C], t).series(t, prec) # y_6
print(h1_1(t=test_z))
print("hypergeometric parameters = %s, %s, %s" % (A,B,1+A+B-C))
h2_1 = hypergeometric([A,B], [1+A+B-C], t).series(t, prec) # y_5
print(h2_1(t=test_z))


test_z = 1.5+0.1*I
print("test input value = %s" % test_z)

print("values of series centered at 1")
print("hypergeometric parameters = %s, %s, %s" % (C-A, C-B, 1-A-B+C))
h1_1 = hypergeometric([C-A, C-B], [1-A-B+C], t).series(t, prec) # y_6
print(h1_1(t=test_z))
print("hypergeometric parameters = %s, %s, %s" % (A,B,1+A+B-C))
h2_1 = hypergeometric([A,B], [1+A+B-C], t).series(t, prec) # y_5
print(h2_1(t=test_z))

print("values of series centered at oo")
print("hypergeometric parameters = %s, %s, %s" % (B, 1+B-C, 1+B-A))
h1_inf = hypergeometric([B, 1+B-C], [1+B-A], t).series(t, prec) # y_10
print(h1_inf(t=test_z))
print("hypergeometric parameters = %s, %s, %s" % (A, 1+A-C, 1+A-B))
h2_inf = hypergeometric([A, 1+A-C], [1+A-B], t).series(t, prec) # y_9
print(h2_inf(t=test_z))
