module

using BNN, BNN_operators, Plots, Random, Statistics

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
    display(scatter((proportions, T), title = "Solution Landscape", xlabel = "Proportion of weights changed", ylabel="Training accuracy", legend=false))
end

function test_layer_elements(X,y,H,max_elements)
    CE = []
    for seed in 1:10
        Random.seed!(seed)
        cross_entropy = []
        for m in 1:max_elements
            Ŵ = Weights(X,y,m,H)
            L = 30
            T₀ = 10
            steps = min(200, prod(size(Ŵ.W)))
            T = cooling(L, steps, T₀)
            W = train(X, y, Ŵ, T)
            push!(cross_entropy, energy(X,y,W))
        end
        push!(CE, cross_entropy)
    end
    CE = mean(CE)
    display(scatter(CE, xlabel = "Number of elements in each layer", ylabel="Cross-Entropy", legend=false))
end

function test_number_layers(X,y,max_layers, M)
    CE = []
    for seed in 1:10
        Random.seed!(seed)
        cross_entropy = []
        for h in 1:max_layers
            Ŵ = Weights(X,y,h,M)
            L = 30
            T₀ = 10
            steps = min(200, prod(size(Ŵ.W)))
            T = cooling(L, steps, T₀)
            W = train(X, y, Ŵ, T)
            push!(cross_entropy, energy(X,y,W))
        end
        push!(CE, cross_entropy)
    end
    CE = mean(CE)
    display(scatter(CE,xlabel = "Number of layers", ylabel="Cross-Entropy", legend=false))
end
                    
end
