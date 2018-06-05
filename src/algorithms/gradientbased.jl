
abstract type GradientAlgorithm <: MLAlgorithm end

# function gradient!(alg::GradientAlgorithm, model, data)
#     A_mul_B!(alg.nvec, data.x, model.β)          # nvec ← x * β
#     deriv!(alg.nvec, model.loss, data.y, alg.nvec) # nvec ← deriv(L, y, x * β)
#     multiply_by_weights!(alg.nvec, data.w)   # nvec .*= w ./ n
#     At_mul_B!(alg.pvec, o.x, alg.nvec)      # pvec ← x'nvec
# end

# multiply_by_weights!(nvec, w::Void) = scale!(nvec, 1 / length(nvec))
# function multiply_by_weights!(nvec, w)
#     wt = inv(length(nvec))
#     for i in eachindex(nvec)
#         @inbounds nvec[i] *= w[i] * wt
#     end
# end


mutable struct AdaptiveProxGrad <: GradientAlgorithm
    nvec::Vector{Float64}
    pvec::Vector{Float64}
    step::Vector{Float64}
    divisor::Float64
    function AdaptiveProxGrad(initstep = 1.0, divisor = 1.5)
        divisor > 1  || error("divisor must be > 1")
        initstep > 0 || error("step size must be > 0")
        new(zeros(0), zeros(0), fill(initstep, 1), divisor)
    end
end


function setup!(alg::AdaptiveProxGrad, mod::AbstractMLModel, data::MLData)
    _modelsetup!(mod, data)
    alg.nvec = zeros(nobs(data))
    alg.pvec = zeros(npredictors(data))
    alg.step = fill(alg.step[1], npredictors(data))
end
