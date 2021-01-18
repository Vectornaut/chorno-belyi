
// Belyi maps downloaded from the LMFDB on 05 January 2021.
// Magma code for Belyi map with label 7T6-7_5.1.1_5.1.1-a


// Group theoretic data

d := 7;
i := 6;
G := TransitiveGroup(d,i);
sigmas := [[Sym(7) | [4, 1, 2, 5, 6, 7, 3], [4, 2, 3, 7, 1, 5, 6], [2, 3, 4, 5, 1, 6, 7]], [Sym(7) | [2, 5, 6, 3, 4, 7, 1], [2, 7, 1, 4, 5, 3, 6], [2, 3, 4, 5, 1, 6, 7]]];
embeddings := [ComplexField(15)![0.5, 0.8660254037844387], ComplexField(15)![0.5, -0.8660254037844387]];

// Geometric data

// Define the base field
R<T> := PolynomialRing(Rationals());
K<nu> := NumberField(R![1, -1, 1]);

// Define the curve
S<x> := PolynomialRing(K);
X := EllipticCurve(S!x^3 - 81/87808*x - 513/4917248,S!0);
// Define the map
KX<x,y> := FunctionField(X);
phi := (1/843308032*(-506250*nu + 253125)*x^2 + 1/23612624896*(-7593750*nu + 3796875)*x + 1/10578455953408*(-141243750*nu + 70621875))/(x^7 + 1/14*x^6 + 2817/87808*x^5 + 17415/4917248*x^4 + 1/53971714048*(32400000*nu + 22926105)*x^3 + 1/10578455953408*(947700000*nu - 798982299)*x^2 + 1/4739148267126784*(-4374000000*nu - 13134339783)*x + 1/91029559914971267072*(-5161757400000*nu + 3221620068951))*y + (1/843308032*(506250*nu - 253125)*x^3 + 1/661153497088*(59231250*nu - 29615625)*x^2 + 1/74049191673856*(-68343750*nu + 34171875)*x + 1/2844673747342852096*(-161304918750*nu + 465086053125))/(x^7 + 1/14*x^6 + 2817/87808*x^5 + 17415/4917248*x^4 + 1/53971714048*(32400000*nu + 22926105)*x^3 + 1/10578455953408*(947700000*nu - 798982299)*x^2 + 1/4739148267126784*(-4374000000*nu - 13134339783)*x + 1/91029559914971267072*(-5161757400000*nu + 3221620068951));