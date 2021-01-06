load "triples.m";
for i := 1 to 3 do
  orb := sigmas[i];
  for sigma in orb do
    name := orb_labels[i];
    name *:= Sprintf("-%o,%o,%o", sigma[1], sigma[2], sigma[3]);
    Gamma := TriangleSubgroup(sigma);
    path := "Pictures/"*name*".tex";
    TriangleDrawDessinToFile(Gamma : prec := 5, filename := path);
  end for;
end for;
