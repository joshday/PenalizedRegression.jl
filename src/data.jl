#-----------------------------------------------------------------------# MLData 
# something or nothing
const _N{T} = Union{Nothing, T} where T

struct MLData{X < _N{AbstractArray}, Y < _N{AbstractArray}, W <: _N{AbstractVector}} 
    x::X 
    y::Y 
    w::W
    function MLData(x::X, y::Y, w::W) where {X,Y,W}
        new{X,Y,W}(x, y, w)
    end
end
# TODO: check that data sizes are correct in constructor
mldata(;x=nothing, y=nothing, w=nothing) = MLData(x, y, w)

function Base.show(io::IO, o::MLData)
    print(io, typeof(o))
    o.x != nothing && print(io, "\n    - x: ", summary(o.x))
    o.y != nothing && print(io, "\n    - y: ", summary(o.y))
    o.w != nothing && print(io, "\n    - w: ", summary(o.w))
end

xy(o::MLData) = o.x, o.y

nobs(o::MLData{<:AbstractArray}) = size(o.x, 1)
nobs(o::MLData{Nothing, <:AbstractArray}) = size(o,y, 1)

npredictors(o::MLData{<:AbstractArray}) = size(o.x, 2)
npredictors(o::MLData{Nothing}) = 0
