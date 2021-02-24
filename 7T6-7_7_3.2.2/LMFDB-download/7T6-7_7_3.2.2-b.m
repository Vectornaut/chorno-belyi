
// Belyi maps downloaded from the LMFDB on 05 February 2021.
// Magma code for Belyi map with label 7T6-7_7_3.2.2-b


// Group theoretic data

d := 7;
i := 6;
G := TransitiveGroup(d,i);
sigmas := [[Sym(7) | [7, 4, 5, 3, 6, 1, 2], [4, 6, 7, 3, 2, 1, 5], [2, 3, 1, 5, 4, 7, 6]], [Sym(7) | [4, 5, 7, 3, 6, 1, 2], [4, 6, 7, 2, 1, 3, 5], [2, 3, 1, 5, 4, 7, 6]]];
embeddings := [ComplexField(15)![2.79128784747792, 0.0], ComplexField(15)![-1.79128784747792, 0.0]];

// Geometric data

// Define the base field
R<T> := PolynomialRing(Rationals());
K<nu> := NumberField(R![-5, -1, 1]);

// Define the curve
S<x> := PolynomialRing(K);
X := HyperellipticCurve(S!x^6 + 4/5*x^5 + (9792/125*nu + 17518/125)*x^4 + (-209376/625*nu - 375004/625)*x^3 + (-4428576/15625*nu - 7932919/15625)*x^2 + (-33123744/625*nu - 11866832/125)*x - 887354784/15625*nu - 317901568/3125,S!0);
// Define the map
KX<x,y> := FunctionField(X);
phi := (1/2890*(1344*nu + 2661)*x^4 + 1/2890*(1344*nu + 2661)*x^3 + 1/1445*(120132*nu + 215208)*x^2 + 1/36125*(-4524744*nu - 8105136)*x + 1/7225*(-856896*nu - 1534944))/(x^7 + 1/425*(-5376*nu - 9709)*x^6 + 1/180625*(41105736*nu + 73661259)*x^5 + 1/180625*(-239423184*nu - 428878471)*x^4 + 1/180625*(2068063368*nu + 3704497832)*x^3 + 1/4515625*(-759086643168*nu - 1359742679232)*x^2 + 1/265625*(-42421103424*nu - 75988407040)*x + 1/1953125*(-9898422485952*nu - 17730923908288))*y + (1/2890*(1344*nu + 2661)*x^7 + 1/14450*(9408*nu + 18627)*x^6 + 1/361250*(61467336*nu + 110117259)*x^5 + 1/361250*(-148669584*nu - 266310471)*x^4 + 1/180625*(1155404964*nu + 2069663316)*x^3 + 1/4515625*(-374673633264*nu - 671148326016)*x^2 + 1/903125*(-71247521856*nu - 127624820064)*x + 1/564453125*(-2865132383483328*nu - 5132276819951232))/(x^7 + 1/425*(-5376*nu - 9709)*x^6 + 1/180625*(41105736*nu + 73661259)*x^5 + 1/180625*(-239423184*nu - 428878471)*x^4 + 1/180625*(2068063368*nu + 3704497832)*x^3 + 1/4515625*(-759086643168*nu - 1359742679232)*x^2 + 1/265625*(-42421103424*nu - 75988407040)*x + 1/1953125*(-9898422485952*nu - 17730923908288));