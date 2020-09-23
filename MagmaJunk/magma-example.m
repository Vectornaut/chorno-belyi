sigma := [Sym(4) | (1, 2, 3, 4), (1, 3, 4, 2), (1, 3, 4) ];
Gamma := TriangleSubgroup(sigma);
Delta := ContainingTriangleGroup(Gamma);
mp := InternalTriangleMatrixEmbeddingMap(Delta);
Da, Db, Dc := Explode([mp(Delta.i) : i in [1..3]]);
reps, G, sidepairing := TriangleCosetRepresentatives(Gamma);
phi_ser, kappa, psi := TrianglePhi(Gamma);

a,b,c := Explode(DefiningABC(Delta));
// Solve the system to find A, B, C
C := 1+1/a;
B := 1/2*(1/a-1/b+1/c+1);
A := 1/2*(1/a-1/b-1/c+1);

A2 := A-C+1;
B2 := B-C+1;
C2 := 2-C;
