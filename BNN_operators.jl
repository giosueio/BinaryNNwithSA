using Statistics

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