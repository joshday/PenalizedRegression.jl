struct StatsModel{L, P, D, T}
    β0::T
    β::Vector{T}
    λ::Vector{T}
    loss::L 
    penalty::P 
    data::D
end
function StatsModel(data::D, loss::L, penalty::P; kw...) where {D<:StatsModelData,L,P}
    T = eltype(data.x)
    p = nvars(data)
    StatsModel{L,P,D,T}(zero(T), zeros(T, p), zeros(T, p), loss, penalty, data)
end
function Base.show(io::IO, o::StatsModel)
    println(io, "StatsModel")
    println(io, "  > β0      : ", o.β0)
    println(io, "  > β       : ", o.β)
    println(io, "  > λ       : ", o.λ)
    println(io, "  > loss    : ", o.loss)
    println(io, "  > penalty : ", o.penalty)
    print(io,   "  > data    : ", o.data)
end

function StatsModel(x, y, loss::Loss = L2DistLoss(), penalty::Penalty = NoPenalty())
    StatsModel(StatsModelData(x, y), loss, penalty)
end