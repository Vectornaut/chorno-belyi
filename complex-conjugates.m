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

function AreComplexConjugate(sigma, sigmap)
  // perms that conjugate sigma_0 to sigmap_0
  S := Parent(sigma[1]);
  conjs_0 := [];
  // could be improved by using cosets of centralizers
  for rho in S do
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

function PermutationTripleToOneLineNotation(sigma);
  return [Eltseq(el) : el in sigma];
end function;

function PairsToString(pairs)
  pairs_one := [];
  for pair in pairs do
    pair_one := [];
    for sigma in pair do
      Append(~pair_one, PermutationTripleToOneLineNotation(sigma));
    end for;
    Append(~pairs_one, pair_one);
  end for;
  pairs_str := Sprint(pairs_one);
  pairs_str := ReplaceAll("\n", "", pairs_str);
  return pairs_str;
end function;

function ProcessOrbits(input,output)
  T0 := Cputime();
  file := Open(input, "r");
  eof_bool := false;
  while not eof_bool do
    str := Gets(file);
    print Split(str, "|")[1];
    if IsEof(str) then
      eof_bool := true;
      break;
    end if;
    orbit := LineToOrbit(str);
    print "identifying complex conjugate pairs...";
    t0 := Cputime();
    pairs := IdentifyComplexConjugatePairs(orbit);
    t1 := Cputime();
    printf "done. Took %o seconds\n", t1-t0;
    pairs_str := PairsToString(pairs);
    output_str := Sprintf("%o|%o", str, pairs_str);
    Write(output, output_str);
  end while;
  T1 := Cputime();
  printf "Total time: %o seconds\n", T1-T0;
  return Sprintf("Data written to %o", output);
end function;
