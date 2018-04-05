module PenalizedRegression

using Compat
using Reexport, Compat.LinearAlgebra
@reexport using LearnBase, LossFunctions, PenaltyFunctions, LearningStrategies

import LearnBase: nobs
import LearningStrategies: setup!, update!, hook, finished, cleanup!, learn!

abstract type MLModel end

abstract type ClosedFormSolver <: LearningStrategy end
finished(strat::ClosedFormSolver, model) = true


include("data.jl")
include("models/linreg.jl")
include("spec.jl")


# include("algorithms/gradientbased.jl")
# include("algorithms/dwd.jl")

end # module
