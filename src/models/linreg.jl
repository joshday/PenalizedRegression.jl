mutable struct LinearRegression{T} <: MLModel 
    β0::T
    β::Vector{T}
    A::Matrix{T}
end
LinearRegression(T = Float64) = LinearRegression(zero(T), zeros(T, 0), zeros(T, 0, 0))

algorithm(::LinearRegression{T}) where {T} = Sweep(T)

mutable struct Sweep{T} <: ClosedFormSolver 
    A::Matrix{T}  # [x 1 y]' * [x 1 y] = 
end
Sweep(T=Float64) = Sweep(zeros(T, 0, 0))

# function update!(model::LinearRegression{T}, strat::Sweep{T}, data::MLData{X,Y,Nothing,:rows}) where {T,X,Y}
#     x, y = xy(data)
#     n = nobs(data)
#     p = npredictors(data)
#     strat.A = zeros(T, p + 2, p + 2)
#     model.β = zeros(T, p)
#     strat.A[1:end-2, 1:end-2] = x'x / n
#     strat.A[1:end-2, end-1] = vec(mean(x, 1))
#     strat.A[end-1, ]
#     strat.A[end] = one(T)
# end

# function learn!(model, strat::LearningStrategy, data)
#     setup!(strat, model[, data])
#     for (i, item) in enumerate(data)
#         update!(model, strat[, i], item)
#         hook(strat, model[, data], i)
#         finished(strat, model[, data], i) && break
#     end
#     cleanup!(strat, model)
#     model
# end