using Bicor
using Base.Test

tests = ["colwise",
         ]

println("Running tests:")

for t in tests
    println(" * $(t)")
    include("$(t).jl")
end
