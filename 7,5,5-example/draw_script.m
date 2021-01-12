load "triples.m";
for i := 1 to 3 do
  orb := sigmas[i];
  for sigma in orb do
    name := orb_labels[i]*"-";
    for j := 1 to 3 do
      name *:= Join(Split(Sprint(sigma[j]), " "), ""); // remove spaces
      if j ne 3 then
        name *:= ",";
      end if;
    end for;
    Gamma := TriangleSubgroup(sigma);
    path := "Pictures/"*name*".tex";
    TriangleDrawDessinToFile(Gamma : prec := 5, filename := path);
  end for;
end for;
