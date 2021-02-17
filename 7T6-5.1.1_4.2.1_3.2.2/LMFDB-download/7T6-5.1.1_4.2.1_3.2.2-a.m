
// Belyi maps downloaded from the LMFDB on 05 February 2021.
// Magma code for Belyi map with label 7T6-5.1.1_4.2.1_3.2.2-a


// Group theoretic data

d := 7;
i := 6;
G := TransitiveGroup(d,i);
sigmas := [[Sym(7) | [5, 2, 1, 4, 7, 3, 6], [2, 3, 4, 1, 6, 5, 7], [2, 1, 5, 3, 4, 7, 6]]];
embeddings := [ComplexField(15)![1.0, 0.0]];

// Geometric data

// Define the base field
K<nu> := RationalsAsNumberField();
// Define the curve
X := Curve(ProjectiveSpace(PolynomialRing(K, 2)));
// Define the map
KX<x> := FunctionField(X);
phi := (1119744/634207*x^7 + 373248/634207*x^6 + 746496/15855175*x^5)/(x^7 + 1234/1505*x^6 + 398964/2265025*x^5 - 52936/3171035*x^4 - 5680/634207*x^3 - 1248/3171035*x^2 + 64/634207*x + 128/15855175);