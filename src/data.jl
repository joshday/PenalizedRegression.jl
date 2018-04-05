#-----------------------------------------------------------------------# MLData 
"""
    mldata(;x = nothing, y = nothing, w = nothing, obs = :rows)

Create a structure holding cleaned-up data.
"""
struct MLData{
        X <: Union{Nothing, AbstractArray}, 
        Y <: Union{Nothing, AbstractArray},
        W <: Union{Nothing, AbstractVector},
        Obs
    } 
    x::X 
    y::Y 
    w::W
    function MLData(x::X, y::Y, w::W, obs::Symbol = :rows) where {X,Y,W}
        obs == :rows || obs == :cols || Compat.@error("obs must be :rows or :cols. Found :$obs")
        new{X,Y,W,obs}(x, y, w)
    end
end
# TODO: check that data sizes are correct in constructor
mldata(;x=nothing, y=nothing, w=nothing, obs=:rows) = MLData(x, y, w, obs)

function Base.show(io::IO, o::MLData)
    print(io, typeof(o))
    o.x != nothing && print(io, "\n    - x: ", summary(o.x))
    o.y != nothing && print(io, "\n    - y: ", summary(o.y))
    o.w != nothing && print(io, "\n    - w: ", summary(o.w))
end

xy(o::MLData) = o.x, o.y

nobs(o::MLData{X,Y,W,:rows}) where {X, Y<:AbstractVector, W} = length(o.y)

npredictors(o::MLData{X,Y,W,:rows}) where {X,Y,W} = size(o.x, 2)
npredictors(o::MLData{X,Y,W,:cols}) where {X,Y,W} = size(o.x, 1)

function xtx_n(o::MLData{X,Y,Nothing,:rows}) where {X,Y}
    o.x'o.x
end