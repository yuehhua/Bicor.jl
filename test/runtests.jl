using Bicor
using Missings
using Base.Test

x1 = [1.0, 2.0, 3.0, 4.0, 5.0, missing]
ans1 = [-0.6885895075239652, -0.3442947537619826, 0.0, 0.3442947537619826, 0.6885895075239652, 0.0]

res, nNAentries, NAmed, zeroMAD = prepareColBicor(x1, 1.0, 1, false)
@test res == ans1
@test nNAentries == 1
@test NAmed == false
@test zeroMAD == false



x2 = [0.0, 0.0, 0.0, 0.0, 0.0, missing]
ans2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

@time res, nNAentries, NAmed, zeroMAD = prepareColBicor(x2, 1.0, 1, false)
@test res == ans2
@test nNAentries == 1
@test NAmed == true
@test zeroMAD == true



x3 = [missing, missing, missing, missing, missing, missing]
ans3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

@time res, nNAentries, NAmed, zeroMAD = prepareColBicor(x3, 1.0, 1, false)
@test res == ans3
@test nNAentries == 1
@test NAmed == true
@test zeroMAD == true
