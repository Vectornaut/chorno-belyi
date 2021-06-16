// translated from Python implementation in covering.py

function Cosh_ab(p,q,r : prec := 30)
  // returns cosh(C), where C is length of side opposite angle pi/r
  // see Prop 2.2 of KMSV
  CC<I> := ComplexField(prec);
  pi := Pi(CC);
  return (Cos(pi/p)*Cos(pi/q) + Cos(pi/r))/(Sin(pi/p)*Sin(pi/q));
end function;

function PlaneToDisc(z)
  // isomorphism HH -> DD
  CC<I> := Parent(z);
  return (z+I)/(z-I);
end function;

function DiscToPlane(w)
  // isomorphism DD -> HH
  CC<I> := Parent(w);
  return -I*(w+1)/(w-1);
end function;

// for plane 
function CrossVertex_Plane(p,q,r : prec := 30)
  // Compute the coordinates of the cross vertex in HH
  CC<I> := ComplexField(prec);
  pi := Pi(CC);
  lambda := Cosh_ab(p,q,r : prec := prec);
  mu := lambda + Sqrt(lambda^2-1);
  x_c := (mu^2-1)/(2*(Cot(pi/p) + mu*Cot(pi/q)));
  y_c := Sqrt(Cosec(pi/p)^2 - (x_c - Cot(pi/p))^2);
  return x_c + I*y_c;
end function;

// for disc
function TranslateOrigin(a)
  // translate origin to a
  CC<I> := Parent(a);
  return Matrix(CC,2,2,[[1, a],[ComplexConjugate(a), 1]]);
end function;

// for disc
function Translate(a,b)
  // translate a to b
  CC<I> := Parent(a);
  return TranslateOrigin(b)*TranslateOrigin(-a);
end function;

// for disc
function CrossVertex(p,q,r : prec := 30);
  // Compute the coordinates of the cross vertex in DD 
  return PlaneToDisc(CrossVertex_Plane(p,q,r : prec := prec));
end function;

// for disc
function SideShift_ba(p,q,r : prec := 30);
  // translate w_b to w_a
  lambda := Cosh_ab(p,q,r : prec := prec);
  mu := lambda + Sqrt(lambda^2-1);
  return Matrix(CC,2,2,[[1+mu, 1-mu],[1-mu, 1+mu]]);
end function;

// for disc
function SideShift_ca(p,q,r : prec := 30);
  // translate cross vertex to origin
  wc := CrossVertex(p,q,r : prec := prec);
  return Translate(wc,0);
end function;

// Functions relating to hypergeometric series

function HypergeometricParameters(p,q,r)
  // return hypergeometric parameters: a,b,c -> A,B,C
  return (1/2)*(1+1/p-1/q-1/r), (1/2)*(1+1/p-1/q+1/r), 1+1/p;
end function;

function MoebiusTransform(P,z)
  // Given an invertible 2 by 2 matrix P and a complex number z, return the image of z under the Moebius transform associated to P
  k := BaseRing(Parent(P));
  v := Matrix(k,2,1,[z,1]);
  prod := P*v;
  assert prod[2,1] ne 0;
  return prod[1,1]/prod[2,1];
end function;

function ScalingConstants(p,q,r : prec := 30);
  CC<I> := ComplexField(prec);
  A,B,C := HypergeometricParameters(p,q,r);
  shift_ba := SideShift_ba(p,q,r : prec := prec);
  shift_ca := SideShift_ba(p,q,r : prec := prec);
  w_b := MoebiusTransform(shift_ba^-1, 0);
  lambda := Cosh_ab(p,q,r : prec := prec);
  mu := lambda + Sqrt(lambda^2 -1);
  z_b := mu*I;
  z_c := CrossVertex_Plane(p,q,r : prec := prec);
  C_c := (z_b - z_c)/(z_b - ComplexConjugate(z_c));
  w_c := MoebiusTransform(shift_ca^-1, 0);
  return w_b*Gamma(2-C)*Gamma(C-A)*Gamma(C-B)/(Gamma(1-A)*Gamma(1-B)*Gamma(1-A-B+C)),
        -w_b*Gamma(1+A+B-C)*Gamma(C-A)*Gamma(C-B)/(Gamma(A)*Gamma(B)*Gamma(C-B)),
        C_c*Gamma(1-A)*Gamma(C-A)*Gamma(1+A-B)/(Gamma(1+B-A)*Gamma(1-B)*Gamma(C-B));
end function;

function LiftingSeries(p,q,r : prec := 30, bound := 0);

  CC<I> := ComplexField(prec);
  // make series with rings, with BigO bound, if specified
  if bound eq 0 then
    R<t> := PowerSeriesRing(CC);
    R0<t0> := PuiseuxSeriesRing(CC);
    R1<t1> := PuiseuxSeriesRing(CC);
    Roo<too> := PuiseuxSeriesRing(CC);
  else
    R<t> := PowerSeriesRing(CC,bound);
    R0<t0> := PuiseuxSeriesRing(CC,bound);
    R1<t1> := PuiseuxSeriesRing(CC,bound);
    Roo<too> := PuiseuxSeriesRing(CC,bound);
  end if;

  A,B,C := HypergeometricParameters(p,q,r);
  // series centered at 0
  // t0 = t
  F1_0 := (R0 ! HypergeometricSeries(CC!A,CC!B,CC!C,t));
  F2_0 := t0^(1-C)*(R0 ! HypergeometricSeries(CC!1+A-C,CC!1+B-C,CC!2-C,t));
  // series centered at 1
  // t1 = 1-t
  F1_1 := t1^(C-A-B)*(R1 ! HypergeometricSeries(CC!C-A,CC!C-B,CC!1-A-B+C,t));
  F2_1 := (R1 ! HypergeometricSeries(CC!A,CC!B,CC!1+A+B-C,t));
  // series centered at oo
  // too = 1/t
  F1_oo := too^(B-A)*(Roo ! HypergeometricSeries(CC!B,CC!1+B-C,CC!1+B-A,t));
  F2_oo := (Roo ! HypergeometricSeries(CC!A,CC!1+A-C,CC!1+A-B,t));

  //kappa0, kappa1, kappaoo := ScalingConstants(p,q,r : prec := prec);
  //return R0!kappa0*F1_0/F2_0, R1!kappa1*F1_1/F2_1, Roo!kappaoo*F1_oo/F2_oo;
  return R0!F1_0/F2_0, R1!F1_1/F2_1, Roo!F1_oo/F2_oo;
end function;

function CoveringSeries(p,q,r : prec := 30, bound := 0);
  psi0, psi1, psioo := LiftingSeries(p,q,r : prec := prec, bound := bound);
  return Reversion(psi0), Reversion(psi1), Reversion(psioo);
end function;
