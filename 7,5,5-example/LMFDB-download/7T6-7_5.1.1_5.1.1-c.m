
// Belyi maps downloaded from the LMFDB on 05 January 2021.
// Magma code for Belyi map with label 7T6-7_5.1.1_5.1.1-c


// Group theoretic data

d := 7;
i := 6;
G := TransitiveGroup(d,i);
sigmas := [[Sym(7) | [3, 1, 5, 6, 4, 7, 2], [3, 2, 7, 1, 5, 4, 6], [2, 3, 4, 5, 1, 6, 7]], [Sym(7) | [4, 1, 5, 3, 6, 7, 2], [3, 2, 7, 4, 1, 5, 6], [2, 3, 4, 5, 1, 6, 7]], [Sym(7) | [3, 5, 2, 6, 4, 7, 1], [2, 7, 3, 1, 5, 4, 6], [2, 3, 4, 5, 1, 6, 7]], [Sym(7) | [4, 6, 2, 3, 7, 5, 1], [6, 7, 3, 4, 1, 2, 5], [2, 3, 4, 5, 1, 6, 7]], [Sym(7) | [7, 5, 2, 3, 6, 1, 4], [2, 6, 3, 4, 7, 5, 1], [2, 3, 4, 5, 1, 6, 7]], [Sym(7) | [5, 4, 2, 7, 6, 3, 1], [1, 7, 3, 6, 2, 5, 4], [2, 3, 4, 5, 1, 6, 7]], [Sym(7) | [5, 4, 7, 3, 6, 2, 1], [1, 7, 6, 4, 2, 5, 3], [2, 3, 4, 5, 1, 6, 7]], [Sym(7) | [5, 1, 4, 6, 7, 2, 3], [1, 2, 6, 7, 3, 4, 5], [2, 3, 4, 5, 1, 6, 7]]];
embeddings := [ComplexField(15)![1.0154823303634, 0.0], ComplexField(15)![-0.8178968645798563, 1.094297620236517], ComplexField(15)![-0.8178968645798563, -1.094297620236517], ComplexField(15)![-0.07774705915910365, 1.695038798488431], ComplexField(15)![-0.07774705915910365, -1.695038798488431], ComplexField(15)![1.493183146619107, 2.14062674551009], ComplexField(15)![1.493183146619107, -2.14062674551009], ComplexField(15)![-1.210560776123694, 0.0]];

// Geometric data

// Define the base field
R<T> := PolynomialRing(Rationals());
K<nu> := NumberField(R![-45, -15, 10, 20, 17, 5, 5, -1, 1]);

// Define the curve
S<x> := PolynomialRing(K);
X := EllipticCurve(S!x^3 + (175984/317849525*nu^7 - 466672/953548575*nu^6 + 21024/9081415*nu^5 - 48592/317849525*nu^4 + 696656/190709715*nu^3 + 10720/12713981*nu^2 - 638352/63569905*nu + 835641/317849525)*x - 104244944/700858202625*nu^7 - 508117552/2102574607875*nu^6 - 20219104/233619400875*nu^5 + 155591216/700858202625*nu^4 + 2990291728/2102574607875*nu^3 + 894254944/700858202625*nu^2 - 55881712/46723880175*nu - 71179242/77873133625,S!0);
// Define the map
KX<x,y> := FunctionField(X);
phi := (1/3815783547625*(946418688*nu^7 - 6458514048*nu^6 - 12835992960*nu^5 - 19295458560*nu^4 - 26990619264*nu^3 + 472872960*nu^2 + 34006210560*nu + 29121068160)*x^2 + 1/133552424166875*(20114270208*nu^7 - 222845753088*nu^6 - 398448902400*nu^5 - 898139020800*nu^4 - 932229191424*nu^3 - 77322209280*nu^2 + 1079896688640*nu + 1502158728960)*x + 1/6544068784176875*(600501826560*nu^7 - 10489664325504*nu^6 - 21283696056448*nu^5 - 48645784203008*nu^4 - 45697303011200*nu^3 - 7508716269568*nu^2 + 56761943894016*nu + 83294289924480))/(x^7 + 1/3892035*(-16640*nu^7 + 58528*nu^6 - 132448*nu^5 + 83696*nu^4 - 291696*nu^3 + 199616*nu^2 + 759040*nu - 1390263)*x^6 + 1/953548575*(2600832*nu^7 - 3691664*nu^6 + 12873072*nu^5 - 12547920*nu^4 + 13625504*nu^3 - 40451904*nu^2 - 61703040*nu + 58069395)*x^5 + 1/420514921575*(624534528*nu^7 + 684436304*nu^6 + 837278928*nu^5 + 2403788640*nu^4 - 668385104*nu^3 + 8675531904*nu^2 + 1575440640*nu - 10776357855)*x^4 + 1/309078467357625*(228431715072*nu^7 + 777715031968*nu^6 + 1563200261280*nu^5 + 812769877920*nu^4 + 410493602624*nu^3 - 3331472188800*nu^2 - 2817359112960*nu + 2904101572635)*x^3 + 1/75724224502618125*(-17317673308416*nu^7 + 101822596179520*nu^6 + 182903621156928*nu^5 + 526136988276048*nu^4 + 497969914805648*nu^3 + 159796515870528*nu^2 - 503728492283136*nu - 1010409189657435)*x^2 + 1/166971915028272965625*(-27083092602088320*nu^7 - 17488370781301936*nu^6 - 32136535934554032*nu^5 + 51504847630623888*nu^4 + 144774056932086880*nu^3 + 151237017860271168*nu^2 - 32310495210139776*nu - 244395916241566905)*x + 1/859070502820464408140625*(-2893302146319273984*nu^7 - 27398996352040598096*nu^6 - 56429039878661353680*nu^5 - 102098551883884080000*nu^4 - 85281596678278333648*nu^3 + 7414955946811795200*nu^2 + 134386331768126887680*nu + 148038774206271202845))*y + (1/3815783547625*(-946418688*nu^7 + 6458514048*nu^6 + 12835992960*nu^5 + 19295458560*nu^4 + 26990619264*nu^3 - 472872960*nu^2 - 34006210560*nu - 29121068160)*x^3 + 1/934866969168125*(-67449977856*nu^7 + 1320083913600*nu^6 + 2593976243328*nu^5 + 6256034028288*nu^4 + 5768373333888*nu^3 + 997883486208*nu^2 - 6865734445056*nu - 10783543996800)*x^2 + 1/229042407446190625*(-584337392640*nu^7 + 6952154507136*nu^6 + 18042695164032*nu^5 + 28922784483072*nu^4 + 31383903561600*nu^3 + 4587293018112*nu^2 - 47372286726144*nu - 47203759912320)*x + 1/1178423186310650765625*(425900173667472384*nu^7 + 309455554126809216*nu^6 + 676894278272114560*nu^5 - 613299125377657600*nu^4 - 1965852922959866752*nu^3 - 2270651520137154560*nu^2 - 25812005042503680*nu + 3533329512842474880))/(x^7 + 1/3892035*(-16640*nu^7 + 58528*nu^6 - 132448*nu^5 + 83696*nu^4 - 291696*nu^3 + 199616*nu^2 + 759040*nu - 1390263)*x^6 + 1/953548575*(2600832*nu^7 - 3691664*nu^6 + 12873072*nu^5 - 12547920*nu^4 + 13625504*nu^3 - 40451904*nu^2 - 61703040*nu + 58069395)*x^5 + 1/420514921575*(624534528*nu^7 + 684436304*nu^6 + 837278928*nu^5 + 2403788640*nu^4 - 668385104*nu^3 + 8675531904*nu^2 + 1575440640*nu - 10776357855)*x^4 + 1/309078467357625*(228431715072*nu^7 + 777715031968*nu^6 + 1563200261280*nu^5 + 812769877920*nu^4 + 410493602624*nu^3 - 3331472188800*nu^2 - 2817359112960*nu + 2904101572635)*x^3 + 1/75724224502618125*(-17317673308416*nu^7 + 101822596179520*nu^6 + 182903621156928*nu^5 + 526136988276048*nu^4 + 497969914805648*nu^3 + 159796515870528*nu^2 - 503728492283136*nu - 1010409189657435)*x^2 + 1/166971915028272965625*(-27083092602088320*nu^7 - 17488370781301936*nu^6 - 32136535934554032*nu^5 + 51504847630623888*nu^4 + 144774056932086880*nu^3 + 151237017860271168*nu^2 - 32310495210139776*nu - 244395916241566905)*x + 1/859070502820464408140625*(-2893302146319273984*nu^7 - 27398996352040598096*nu^6 - 56429039878661353680*nu^5 - 102098551883884080000*nu^4 - 85281596678278333648*nu^3 + 7414955946811795200*nu^2 + 134386331768126887680*nu + 148038774206271202845));