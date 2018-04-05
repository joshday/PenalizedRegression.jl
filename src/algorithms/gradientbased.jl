function learn!(o::StatsModel{LogitMarginLoss, NoPenalty};
        maxit = 100
    )
    for i in 1:maxit 

    end
end

# function learn!(model, strat::LearningStrategy)
#     setup!(strat, model)
#     for (i, item) in enumerate(InfiniteNothing())
#         update!(model, strat[, i])
#         hook(strat, model, i)
#         finished(strat, model, i) && break
#     end
#     cleanup!(strat, model)
#     model
# end