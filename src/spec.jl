struct ModelSpec{M <: MLModel, D <: MLData, S <: LearningStrategy}
    model::M
    data::D 
    strat::S
end

function Base.show(io::IO, o::ModelSpec)
    println(io, "ModelSpec:")
    println(io, "  > model: ", o.model)
    println(io, "  > data : ", o.data)
    print(  io, "  > strat: ", o.strat)
end

function learn!(o::ModelSpec, strats::LearningStrategy...) 
    s = strategy(o.strat, strats...)
    learn!(o.model, s, Iterators.repeated(o.data))
end