# Global structural identifiability of the T-M-C model with StructuralIdentifiability.jl. Output: y = T.
#
# (A) full T-M-C model: all parameters and initial conditions assessed.
# (C) T-M subsystem (_mac, no CAR T-cells).
# The calibrated configuration (a2, K2, k2 known) is in structural_id_pipeline.jl.

import Pkg
# the pinned environment of this directory (Project.toml, Manifest.toml) rather than whatever
# version the default environment happens to hold: the verdicts depend on the implementation
Pkg.activate(@__DIR__; io=devnull)
Pkg.instantiate(io=devnull)
using StructuralIdentifiability

println("="^60)
println("(A) FULL T-M-C MODEL  |  y = T")
println("="^60)
# a2, K2 and k2 enter as unknown parameters here; the verdicts of interest are those of
# a1, K1, b1, u1, a3, K3, k3, u3.
ode = @ODEmodel(
    T'(t) = a1*T(t)*(1 - T(t)/K1)*(1 + b1*M(t)) - u1*C(t)*T(t),
    M'(t) = a2*M(t)*T(t)*(1 - M(t)/K2) - k2*M(t),
    C'(t) = a3*C(t)*T(t)/(K3 + T(t)) - u3*C(t)*T(t) - k3*C(t),
    y(t) = T(t)
)
res = assess_identifiability(ode)
for (p, verdict) in res
    println("   ", p, "  =>  ", verdict)
end

println()
println("="^60)
println("(C) T-M SUBSYSTEM (_mac, no CAR T-cells)  |  y = T")
println("="^60)
ode2 = @ODEmodel(
    T'(t) = a1*T(t)*(1 - T(t)/K1)*(1 + b1*M(t)),
    M'(t) = a2*M(t)*T(t)*(1 - M(t)/K2) - k2*M(t),
    y(t) = T(t)
)
res2 = assess_identifiability(ode2)
for (p, verdict) in res2
    println("   ", p, "  =>  ", verdict)
end
println("\nDONE")
