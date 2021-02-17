
// Belyi maps downloaded from the LMFDB on 05 February 2021.
// Magma code for Belyi map with label 7T6-7_7_3.2.2-a


// Group theoretic data

d := 7;
i := 6;
G := TransitiveGroup(d,i);
sigmas := [[Sym(7) | [2, 3, 4, 6, 7, 5, 1], [2, 7, 1, 6, 3, 5, 4], [2, 3, 1, 5, 4, 7, 6]]];
embeddings := [ComplexField(15)![1.0, 0.0]];

// Geometric data

// Define the base field
K<nu> := RationalsAsNumberField();
// Define the curve
S<x> := PolynomialRing(K);
X := HyperellipticCurve(S!x^6 + 4/5*x^5 - 42/25*x^4 - 28/25*x^3 + 721/625*x^2 + 336/625*x - 6048/15625,S!0);
// Define the map
KX<x,y> := FunctionField(X);
phi := 1/2*x/(x^4 + 2/5*x^3 - 23/25*x^2 - 24/125*x + 144/625)*y + 1/2;