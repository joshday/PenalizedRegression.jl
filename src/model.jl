struct StatsModel{L, P, D, T}
    β::Vector{T}
    λ::Vector{T}
    loss::L 
    penalty::P 
    data::D
end
function StatsModel(data::D, loss::L, penalty::P) where {D<:StatsModelData,L,P}
    T = eltype(data.x)
    p = nvars(data)
    StatsModel{L,P,D,T}(zeros(T, p), zeros(T, p), loss, penalty, data)
end
function Base.show(io::IO, o::StatsModel)
    println(io, "StatsModel")
    println(io, "  > β       : ", o.β')
    println(io, "  > λ       : ", o.λ')
    println(io, "  > loss    : ", o.loss)
    println(io, "  > penalty : ", o.penalty)
    print(io,   "  > data    : ", o.data)
end

algorithm(o::StatsModel{L, NoPenalty}) where {N, L<:Union{L2DistLoss, ScaledDistanceLoss{L2DistLoss, N}}}

# learn!(o::StatsModel, args...) = learn!(o, strategy(LassoSolver(), args...))

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