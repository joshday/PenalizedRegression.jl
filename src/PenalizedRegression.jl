module PenalizedRegression

using Compat
using Reexport, Compat.LinearAlgebra
@reexport using LearnBase, LossFunctions, PenaltyFunctions, LearningStrategies

include("data.jl")
include("model.jl")
include("algorithms/gradientbased.jl")
# include("algorithms/dwd.jl")

end # module
