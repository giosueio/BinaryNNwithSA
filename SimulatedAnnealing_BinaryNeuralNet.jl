using Plots, Random, Statistics

function binarize(A)
    return floor.(Int8,2*(A.>0) .-1)
end

function xnor(a::Bool,b::Bool)
    return ~(a⊻b)
end

function binarizing_dot(A::Matrix{Int8},B::Matrix{Int8}) # Dot product is performed converting the arrays to BitArray to save memory
    C = BitArray(A.>0)
    D = BitArray(B.>0)
    rowsx, colsx = size(A)
    rowsy, colsy = size(B)
    innerprod = zeros(Int8, rowsx, colsy)
    for row in 1 : rowsx
      for col in 1 : colsy
        s = zero(Int8)
        for k in 1 : colsx
          s+=xnor(C[row, k], D[k, col])
        end
        innerprod[row, col] = binarize(2*s - colsx)
      end
    end
    return innerprod
end

function standardize(Input::Matrix)
    Input = (Input .- mean(Input,dims=1)) ./ std(Input, dims=1)
    return Input
end

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
                Z = binarizing_dot(Z, W[f:t,:])
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

function energy(X::Matrix, y::Vector{Int}, Weight::Weights, f = "CrossEntropy"::String)
    BNN_predictions = BNN_classifier(X, Weight)
    ŷ = BNN_predictions.ŷ
    class_prob = BNN_predictions.class_prob
    if f == "RSS"
        energy = sum((y .- ŷ).^2)
    elseif f == "Misclass"
        energy = sum(ŷ .!= y)
    elseif f == "CrossEntropy" 
        energy = 0
        for i in 1:size(y)[1]
            energy += -log(class_prob[i,:][y[i]]) # y vector has to be encoded st classes are ordered from 1 to K
        end  
    end
    return energy
end

function SimulatedAnnealing(X::Matrix, y::Vector{Int}, Weight::Weights, T::Vector{Float64}) # REMEMBER SPECIFYING T TYPE PROBABLY ::Vector{Float}
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

function cooling(L::Int, steps::Int, T₀::Int, f = "exp"::String)
    if f == "exp" # Exponential multiplicative cooling schedule
        a = 0.85
        T = repeat([T₀*(a^k) for k in 0:(L-1)],inner = steps)
        return T
    elseif f == "lin_mult" # Linear multiplicative
        a = 1.1
        T = [T₀/(a^k) for k in 0:(L-1)]
        T = repeat(T,inner = steps)
        return T
    elseif f == "lin_add" # Linear additive
        T = [T₀*(L - k)/L for k in 0:(L-1)]
        T = repeat(T,inner = steps)
        return T
    end
end

function convergence_time(energies)
    l = length(energies)
    for i in 1:l
        if energies[i] - minimum(energies) < 1e-10
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
        println("Convegence to optimum reached at ", convergence_time(energies), " iterations of Simulated Annealing")
        display(plot(energies, title = "Simulated Annealing", xlabel = "Iterations", ylabel="Cross-Entropy", legend=false))
        display(plot(train_accuracy, xlabel = "Iterations", ylabel="Training Accuracy", legend=false))
    end
    return Weight
end

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

function predict(X,Weight)
    X = standardize(X)
    ŷ = BNN_classifier(X,Weight).ŷ
    return ŷ
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
