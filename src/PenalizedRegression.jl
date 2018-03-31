module PenalizedRegression

using Reexport, LinearAlgebra
@reexport using LearnBase, LossFunctions, PenaltyFunctions, LearningStrategies

include("data.jl")
include("model.jl")

end # module
