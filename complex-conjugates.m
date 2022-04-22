function LineToOrbit(str)
  spl := Split(str, "|");
  label := spl[1];
  d := StringToInteger(Split(label,"T")[1]);
  seq := eval spl[2];
  orbit := [];
  for triple in seq do
    triple_new := [];
    for el in triple do
      Append(~triple_new, Sym(d)!el);
    end for;
    Append(~orbit, triple_new);
  end for;
  return orbit;
end function;

function AreComplexConjugateBruteForce(sigma, sigmap)
  // perms that conjugate sigma_0 to sigmap_0
  S := Parent(sigma[1]);
  C0 := Centralizer(S,sigma[1]);
  T := Transversal(S,C0);
  conjs_0 := [];
  // could be improved by using cosets of centralizers
  for rho in T do
    if sigma[1]^rho eq sigmap[1]^-1 then
      Append(~conjs_0, rho);
    end if;
  end for;
  // perms that conjugate sigma_1 to sigmap_1
  conjs_1 := [];
  for rho in conjs_0 do
    if sigma[2]^rho eq sigmap[2]^-1 then
      Append(~conjs_1, rho);
    end if;
  end for;
  conjugators := Set(conjs_0) meet Set(conjs_1);
  conjugators := SetToSequence(conjugators);
  if #conjugators eq 0 then
    return false;
  else
    return true, conjugators;
  end if;
end function;

function AreComplexConjugate(sigma, sigmap)
  S := Parent(sigma[1]);
  pair := [sigma[1], sigma[2]];
  pair_p := [sigmap[1]^-1, sigmap[2]^-1];
  bool, conj := IsConjugate(S, pair, pair_p);
  if not bool then
    return false;
  else 
    return bool, conj;
  end if;
end function;

function IdentifyComplexConjugatePairs(orbit)
  pairs := [];
  //for i := 1 to #orbit do
  i := 1;
  while i le #orbit do
    sigma := orbit[i];
    //for j := #orbit to 1 by -1 do
    j := #orbit;
    while j gt i do
      sigmap := orbit[j];
      if AreComplexConjugate(sigma, sigmap) then
        Append(~pairs, [sigma, sigmap]);
        Remove(~orbit, j);
      end if;
      j +:= -1;
    end while;
    i +:= 1;
  end while;
  return pairs;
end function;

function IdentifyComplexConjugates(orbit)
  nonconjs := [];
  conjs := [];
  //for i := 1 to #orbit do
  i := 1;
  while i le #orbit do
    sigma := orbit[i];
    conj_bool := false;
    //for j := #orbit to 1 by -1 do
    j := #orbit;
    while j gt i do
      sigmap := orbit[j];
      if AreComplexConjugate(sigma, sigmap) then
        conj_bool := true;
        Remove(~orbit, j);
      end if;
      j +:= -1;
    end while;
    if conj_bool then
      Append(~conjs, sigma);
    else
      Append(~nonconjs, sigma);
    end if;
    i +:= 1;
  end while;
  return nonconjs, conjs;
end function;

function ListOfTriplesToString(list)
  list_one := [];
  for sigma in list do
    Append(~list_one, [Eltseq(el) : el in sigma]);
  end for;
  list_str := Sprint(list_one);
  list_str := ReplaceAll("\n", "", list_str);
  return list_str;
end function;

function ProcessOrbits(input,output)
  T0 := Cputime();
  file := Open(input, "r");
  eof_bool := false;
  while not eof_bool do
    str := Gets(file);
    if IsEof(str) then
      eof_bool := true;
      break;
    end if;
    label := Split(str, "|")[1];
    print label;
    orbit := LineToOrbit(str);
    print "identifying complex conjugate pairs...";
    t0 := Cputime();
    nonconjs, conjs := IdentifyComplexConjugates(orbit);
    t1 := Cputime();
    printf "done. Took %o seconds\n", t1-t0;
    nonconjs_str := ListOfTriplesToString(nonconjs);
    conjs_str := ListOfTriplesToString(conjs);
    output_str := Sprintf("%o|%o|%o", label, nonconjs_str, conjs_str);
    Write(output, output_str);
  end while;
  T1 := Cputime();
  printf "Total time: %o seconds\n", T1-T0;
  return Sprintf("Data written to %o", output);
end function;
