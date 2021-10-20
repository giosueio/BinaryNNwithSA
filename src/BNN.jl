module BNN

export Weights, BNN_classifier
export energy, SimulatedAnnealing, lin_cooling, exp_cooling, convergence_time, train, BNN_predict, misclass_error
export binarize, xnor, xnordotproduct, standardize

using Plots, Random, Statistics
include("BNN_operators.jl")

mutable struct Weights
    W::Matrix
    H::Int
    M::Int
    K::Int
    function Weights(X::Matrix, y::Vector{Int}, M::Int, H::Int) # Initialization of random weights 
        p = size(X)[2]
        K = length(unique(y)) # Number of classes
        W = binarize(rand(0:1,(p+M*(H-1)+K,M))) # the last K rows are weights for the hidden layer of class probabilities
        return new(W, H, M, K)
    end
end

mutable struct BNN_classifier
    ŷ ::Vector{Int}
    class_prob ::Matrix
    function BNN_classifier(X::Matrix, Weight::Weights) # Using new multiplication method
        Z = X
        W = Weight.W
        H = Weight.H
        K = Weight.K
        N = size(Z)[1]
        f = 1
        for i in 1:H
            t = f + size(Z)[2]-1
            if i == 1
                Z = binarize(Z * W[f:t,:])
            else
                Z = xnordotproduct(Z, W[f:t,:])
            end
            f = t + 1
        end
        Z = Z * W[end - K + 1:end,:]'
        class_prob = reshape(
            [exp.(z)/sum(exp.(row)) for row in eachrow(Z) for z in row],
            (K, N))'
        ŷ = argmax.(eachrow(class_prob))
        return new(ŷ, class_prob)
    end
end

function energy(X::Matrix, y::Vector{Int}, Weight::Weights)
    BNN_predictions = BNN_classifier(X, Weight)
    ŷ = BNN_predictions.ŷ
    class_prob = BNN_predictions.class_prob
    energy = 0
    for i in 1:size(y)[1]
        energy += -log(class_prob[i,:][y[i]]) # y vector has to be encoded st classes are ordered from 1 to K for indexing to work
    end  
    return energy
end

function SimulatedAnnealing(X::Matrix, y::Vector{Int}, Weight::Weights, T::Vector{Float64})
    energies, train_accuracy= [], []
    function Metropolis_Step(Weight,t)
        E₀ = energy(X, y, Weight)
        s = size(Weight.W)
        r₁,r₂ = rand(1:s[1]), rand(1:s[2]) # Random index for each coordinate
        Weight.W[r₁,r₂] = - Weight.W[r₁,r₂] # Switch to new configuration
        E₁ = energy(X, y, Weight) # Energy is recomputed after sign switch
        ΔE = E₁ - E₀
        pr = exp(-ΔE/t)
        r = rand()
        if r < min(1,pr)
            return Weight # Keep the switch
        else
            Weight.W[r₁,r₂] = - Weight.W[r₁,r₂] # Switch back to original configuration
            return Weight
        end
    end
    for t in T
        Weight = Metropolis_Step(Weight,t)
        push!(energies,energy(X,y,Weight))
        ŷ = BNN_classifier(X, Weight).ŷ 
        push!(train_accuracy, sum(ŷ .== y)/length(y))
    end
    return Weight, energies, train_accuracy
end

function lin_cooling(a::Float64, steps::Int, T₀::Int) # Linear cooling schedule
    T = []
    t = T₀
    while t >= 0
        t_a = repeat([t],steps)
        push!(T,t_a)
        t -= a
    end
    T = collect(Iterators.flatten(T))
    return T
end
                                        
function exp_cooling(L::Int, steps::Int, T₀::Int) # Exponential cooling schedule
    a = 0.85
    T = repeat([T₀*(a^k) for k in 0:(L-1)],inner = steps)
    return T
end

function convergence_time(energies)
    l = length(energies)
    for i in 1:l
        if energies[i] - minimum(energies) < 1e-6
            return i
        end
    end
end

function train(X::Matrix, y::Vector{Int}, Weight::Weights, T::Vector{Float64}, disp = false)
    X = standardize(X)
    if disp==true
        println("Initial energy (log cross-entropy of the training set):",energy(X, y, Weight))
    end
    SA = SimulatedAnnealing(X, y, Weight, T)
    Weight = SA[1]
    energies = SA[2]
    train_accuracy = SA[3]
    ŷ = BNN_classifier(X,Weight).ŷ
    if disp==true
        println("Final energy (log cross-entropy of the training set):", energies[end])
        println("Accuracy on the training set: ", round(train_accuracy[end],digits=4))
        plot(energies, title = "Simulated Annealing", xlabel = "Iterations", ylabel="Cross-Entropy", legend=false, color=:darkblue)
        display(plot!(twinx(), T, label="Temperature", color=:orange))
        display(plot(train_accuracy, xlabel = "Iterations", ylabel="Training Accuracy", legend=false, color=:darkblue))
    end
    return Weight
end

function BNN_predict(X,Weight)
    X = standardize(X)
    ŷ = BNN_classifier(X,Weight).ŷ
    return ŷ
end

function misclass_error(ŷ, y)
    return sum(ŷ.!=y)/length(y)
end

end
