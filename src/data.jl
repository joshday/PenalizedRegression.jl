struct StatsModelData{X,Y,W,Obs}
    x::X 
    y::Y 
    w::W
end
function Base.show(io::IO, o::StatsModelData{X,Y,W,Obs}) where {X,Y,W,Obs}
    print(io, "StatsModelData (observations in $Obs):")
    print(io, "\n    - x : ", summary(o.x))
    print(io, "\n    - y : ", summary(o.y))
    o.w != nothing && print(io, "\n  > w : ", summary(o.w))
end


function StatsModelData(x::X, y::Y; obs = :rows) where {X<:AbstractMatrix,Y<:AbstractVector}
    if obs == :rows 
        size(x, 1) == length(y) || error("Incompatible x and y sizes")
    elseif obs == :cols 
        size(x, 2) == length(y) || error("Incompatible x and y sizes")
    else
        Compat.@error("obs must be :rows or :cols.  Found :$obs")
    end
    StatsModelData{X,Y,Nothing,obs}(x, y, nothing)
end

nvars(o::StatsModelData{X,Y,W,:rows}) where {X,Y,W} = size(o.x, 2) 
nvars(o::StatsModelData{X,Y,W,:cols}) where {X,Y,W} = size(o.x, 1) 