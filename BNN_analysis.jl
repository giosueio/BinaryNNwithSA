using BNN, Plots, Random, Statistics

function solution_landscape(X, y, Weight₀)
    X = standardize(X)
    ŷ = BNN_classifier(X, Weight₀).ŷ
    initial_accuracy = sum(ŷ .== y)/length(y)
    T = []
    proportions = []
    for seed in 1:prod(size(Weight₀.W))
        Random.seed!(seed)
        train_accuracy = []
        l = length(Weight₀.W)
        for i in 1:l
            Weight = deepcopy(Weight₀)
            if seed == 1
                push!(proportions, i/l)
            end
            for j in 1:i
                s = size(Weight.W)
                r₁,r₂ = rand(1:s[1]), rand(1:s[2]) # Random index for each coordinate
                Weight.W[r₁,r₂] = - Weight.W[r₁,r₂]
            end
            ŷ = BNN_classifier(X, Weight).ŷ
            push!(train_accuracy, sum(ŷ .== y)/length(y))
        end
        push!(T,train_accuracy)
    end
    T = mean(T)
    pushfirst!(T,initial_accuracy)
    pushfirst!(proportions,0)
    display(scatter((proportions, T), title = "Solution Landscape", xlabel = "Proportion of weights changed", ylabel="Training accuracy", ylim=(0,1), legend=false))
end

function test_layer_elements(X,y,H,max_elements,f = "Training error")
    stats = []
    elmnts = [i for i in 1:max_elements]
    for seed in 1:10
        Random.seed!(seed)
        st = []
        for m in 1:max_elements
            Ŵ = Weights(X,y,m,H)
            L = 30
            T₀ = 10
            steps = min(200, prod(size(Ŵ.W)))
            T = cooling(L, steps, T₀)
            W = train(X, y, Ŵ, T)
            if f == "Cross-entropy"
                push!(st, energy(X,y,W))
            elseif f == "Training error"
                ŷ = BNN_predict(X,W)
                push!(st, misclass_error(ŷ,y))
            end
        end
        push!(stats, st)
    end
    stats = mean(stats)
    conv = convergence_time(stats)
    display(scatter((elmnts, stats), xlabel = "Number of elements in each layer", ylabel=f, ylim=(0,1), legend=false))
    println("For ",H, " layers, ",f," stabilizes at ", conv, " elements in each layer")
    return conv
end

function test_number_layers(X,y,max_layers, M,f = "Training error")
    stats = []
    lyrs = [i for i in 1:max_layers]
    for seed in 1:10
        Random.seed!(seed)
        st = []
        for h in 1:max_layers
            Ŵ = Weights(X,y,h,M)
            L = 30
            T₀ = 10
            steps = min(200, prod(size(Ŵ.W)))
            T = cooling(L, steps, T₀)
            W = train(X, y, Ŵ, T)
            if f == "Cross-entropy"
                push!(st, energy(X,y,W))
            elseif f == "Training error"
                ŷ = BNN_predict(X,W)
                push!(st, misclass_error(ŷ,y))
            end
        end
        push!(stats, st)
    end
    stats = mean(stats)
    conv = convergence_time(stats)
    display(scatter((lyrs,stats),xlabel = "Number of layers", ylabel=f, ylim=(0,1), legend=false))
    println("For ",M, " elements in each layer, ",f," stabilizes at ", conv, " layers")
    return conv
end

