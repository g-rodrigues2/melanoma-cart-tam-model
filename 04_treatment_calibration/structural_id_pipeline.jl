# Global structural identifiability of the T-M-C model in the calibrated configuration: alpha2, K2, kappa2 known
# (fixed at 0.1, 1000, 0.34 on the mm^3 scale). Question: does fixing the macrophage parameters make beta1 and mu1
# identifiable?
import Pkg
# the pinned environment of this directory (Project.toml, Manifest.toml) rather than whatever
# version the default environment happens to hold: the verdicts depend on the implementation
Pkg.activate(@__DIR__; io=devnull)
Pkg.instantiate(io=devnull)
using StructuralIdentifiability

println("="^60)
println("CALIBRATED CONFIGURATION: a2=0.1, K2=1000, k2=0.34 known | y = T")
println("="^60)
ode = @ODEmodel(
    T'(t) = a1*T(t)*(1 - T(t)/K1)*(1 + b1*M(t)) - u1*C(t)*T(t),
    M'(t) = (1//10)*M(t)*T(t)*(1 - M(t)//1000) - (34//100)*M(t),
    C'(t) = a3*C(t)*T(t)/(K3 + T(t)) - u3*C(t)*T(t) - k3*C(t),
    y(t) = T(t)
)
res = assess_identifiability(ode)
for (p, v) in res
    println("   ", p, "  =>  ", v)
end

println("\n--- same, with C(0)=1 also known (known_ic) ---")
res2 = assess_identifiability(ode; known_ic=[C])
for (p, v) in res2
    println("   ", p, "  =>  ", v)
end
println("\nDONE")
