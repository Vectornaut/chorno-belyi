p = 5
q = 4
r = 3

CC = ComplexField()

# from Voight's "Quadratic forms and quaternion algebras", pp. 95--97
# our kappa_0 is Voight's C_p
# """ kappa_1 """"""""""" C_q
shift_ba = side_shift_ba(p, q, r)
shift_ca = side_shift_ca(p, q, r)
kappa_0, kappa_1, kappa_inf = scale_consts(p, q, r)

# test lift and covering map computations
lift_0, lift_1, lift_inf, = lifting_series(p, q, r, 32)

# test series centered at 0 and 1
print("testing series centered at 0 and 1")
test_z = 0.3 - 0.2*I
print("test input value = %s" % test_z)

test_w = [
    kappa_0 * lift_0.polynomial()(test_z^(1/p)),
    moebius_transform(shift_ba^(-1), kappa_1 * lift_1.polynomial()((1-test_z)^(1/q))),
]
print("output value for series at 0 = %s" % test_w[0])
print("output value for series at 1 = %s" % test_w[1])
print("difference = %s\n" % (test_w[0]-test_w[1]).abs())

# test series centered at 1 and oo
print("testing series centered at 1 and oo")
test_z = 1.5+0.1*I
print("test input value = %s" % test_z)

test_w = [
    moebius_transform(shift_ba^(-1), kappa_1 * lift_1.polynomial()((1-test_z)^(1/q))),
    moebius_transform(shift_ca^(-1), kappa_inf * lift_inf.polynomial()((1/test_z)^(1/r)))
]

print("output value for series at 1 = %s" % test_w[0])
print("output value for series at oo = %s" % test_w[1])
print("difference = %s\n" % (test_w[0]-test_w[1]).abs())
