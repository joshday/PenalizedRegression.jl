module PenalizedRegression

using Compat
using Reexport, Compat.LinearAlgebra
@reexport using LearnBase, LossFunctions, PenaltyFunctions, LearningStrategies

import LearnBase: nobs
import LearningStrategies: setup!, update!, hook, finished, cleanup!, learn!

#-----------------------------------------------------------------------# Abstract types
abstract type AbstractMLModel end
function Base.show(io::IO, o::AbstractMLModel)
    println(io, typeof(o))
    println(io, "  > β0: ", o.β0)
    print(io,   "  > β : ", o.β')
end
_modelsetup!(o::AbstractMLModel, data) = (o.β = zeros(npredictors(data)))


abstract type MLAlgorithm <: LearningStrategy end 
Base.show(io::IO, o::MLAlgorithm) = print(io, name(o, false, false))
function name(o, withmodule = false, withparams = true)
    s = string(typeof(o))
    if !withmodule
        s = replace(s, r"([a-zA-Z]*\.)" => "")  # remove text that ends in period
    end
    if !withparams
        s = replace(s, r"\{(.*)" => "")  # remove "{" to the end of the string
    end
    s
end

abstract type ClosedFormSolution <: MLAlgorithm end
finished(strat::ClosedFormSolution, model) = true

#-----------------------------------------------------------------------# includes
include("data.jl")
include("models/regerm.jl")
include("algorithms/gradientbased.jl")
# include("models/linreg.jl")
include("spec.jl")

end # module
