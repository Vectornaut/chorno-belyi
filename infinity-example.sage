from sage.geometry.hyperbolic_space.hyperbolic_model import moebius_transform

# in <arXiv:1311.2081>, lambda = cosh(d(a, b)) and mu = exp(d(a, b))

#--------------- upper half plane and disc functions ------------

def cosh_ab(p, q, r):
    return CDF((cos(pi/p)*cos(pi/q) + cos(pi/r)) / (sin(pi/p)*sin(pi/q)))

def plane_to_disc(z):
    CC = ComplexField()
    return CDF((z-I)/(z+I))

def disc_to_plane(w):
    CC = ComplexField()
    return CDF(-I*(w+1)/(w-1))

# for upper half-plane model
# is there some way to double check this formula?
def cross_vertex_plane(p, q, r):
    CC = ComplexField()
    lam = cosh_ab(p, q, r)
    mu = lam + sqrt(lam**2 - 1)
    x_c = (mu**2-1)/(2*(cot(pi/p) + mu*cot(pi/q)))
    y_c = sqrt(csc(pi/p)^2 - ((mu^2 - 1)/(2*(cot(pi/p) + mu*cot(pi/q))) - cot(pi/p))^2)
    return CDF(x_c + I*y_c)

# for disc model
def cross_vertex(p, q, r):
    return plane_to_disc(cross_vertex_plane(p,q,r))

# for disc model
def side_shift_ba(p, q, r):
    lam = cosh_ab(p, q, r)
    mu = lam + sqrt(lam**2 - 1)
    return Matrix([[1+mu, 1-mu], [1-mu, 1+mu]])

# for disc model
def side_shift_ca(p, q, r):
    wc = cross_vertex(p,q,r)
    return translate(wc,0)

# for disc model
# translate origin to a
def translate_origin(a):
    return Matrix([[1, a], [a.conjugate(), 1]])

# for disc model
# translate a to b
def translate(a,b):
    return translate_origin(b)*translate_origin(-a)


# ----------------- series stuff ----------------

# hypergeometric paramaters A, B, C for triangle group with vertex orders p, q, r
def hyper_params(p, q, r):
    return (
        1/2*(1/p-1/q-1/r+1),
        1/2*(1/p-1/q+1/r+1),
        1+1/p
    )

# scaling constants to normalize the solutions to the diff eq
def scale_consts(p, q, r):
    CC = ComplexField()
    A, B, C = hyper_params(p, q, r)
    shift_ba = side_shift_ba(p, q, r)
    shift_ca = side_shift_ca(p, q, r)
    w_b = moebius_transform(shift_ba^(-1), 0)
    lam = cosh_ab(p, q, r)
    mu = lam + sqrt(lam**2 - 1)
    z_b = mu*I
    z_c = cross_vertex_plane(p,q,r)
    C_c = (z_b - z_c)/(z_b - z_c.conjugate())
    w_c = moebius_transform(shift_ca^(-1), 0)
    return (
        w_b * CDF(gamma(2-C)*gamma(C-A)*gamma(C-B) / (gamma(1-A)*gamma(1-B)*gamma(C))),
        -w_b * CDF(gamma(1+A+B-C)*gamma(C-A)*gamma(C-B) / (gamma(A)*gamma(B)*gamma(1-A-B+C))),
        C_c*CDF((gamma(1-A)*gamma(C-A)*gamma(1+A-B))/(gamma(1+B-A)*gamma(1-B)*gamma(C-B)))
    )

# series for lift from PP^1 to DD
def lifting_series(p, q, r, prec):
    # get hypergeometric parameters
    A, B, C = hyper_params(p, q, r)

    # z is the standard coordinate on CP^1
    # w is the standard coordinate on the Poincaré disk
    # around 0,        t =   z,  s^p = t
    # around 1,        t = 1-z,  s^q = t
    # around infinity, t = 1/z,  s^r = t
    t = polygen(QQ, 't')
    s = polygen(QQ, 's')
    w = polygen(QQ, 'w')
    P = PowerSeriesRing(QQ, 's')

    # labels are from Slater, Sections 1.3.1--1.3.2

    # series around 0
    h1_0 = hypergeometric([A, B], [C], t).series(t, prec)
    h2_0 = hypergeometric([A-C+1, B-C+1], [2-C], t).series(t, prec)
    lift_0 = s * P(h1_0(t = s^p), p*prec) / P(h2_0(t = s^p), p*prec)

    # series around 1, with
    #
    #   shift.inverse() . lift_1((1-z)^(1/q)) = lift_0(z^(1/p))
    #
    # on the region where they both converge
    h1_1 = hypergeometric([C-A, C-B], [1-A-B+C], t).series(t, prec) # y_6
    h2_1 = hypergeometric([  A,   B], [1+A+B-C], t).series(t, prec) # y_5
    lift_1 = s * P(h1_1(t = s^q), q*prec) / P(h2_1(t = s^q), q*prec)

    # series around infinity. only valid in the region |z| > 1
    h1_inf = hypergeometric([B, 1+B-C], [1+B-A], t).series(t, prec) # y_10
    h2_inf = hypergeometric([A, 1+A-C], [1+A-B], t).series(t, prec) # y_9
    lift_inf = s * P(h1_inf(t = s^r), r*prec) / P(h2_inf(t = s^r), r*prec) # there's a spare root of unity that goes into the scale constant
    return [f.change_ring(CDF) for f in [lift_0, lift_1, lift_inf]]
