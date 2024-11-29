# Holdout

using Random
function holdOut(N::Int, P::Real)
    v = randperm(N)
    ind = round(Int, N*(1-P))

    return (v[1:ind], v[ind+1:end])
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    # calculate test index cutoff
    testInd = round(Int,N*(1-Ptest))

    trainI, valI = holdOut(testInd, Ptest)
    testI, _ = holdOut(round(Int,N*Ptest), 0)

    # assure disjoint sets
    testI .+= testInd

    @assert(size(trainI, 1) + size(valI, 1) + size(testI, 1) == N);
    return (trainI, valI, testI)
end;

# One-hot encoding
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}=unique(feature))
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]));

    # Second defensive statement, check the number of classes
    numClasses = length(classes);
    @assert(numClasses>1)

    if (numClasses==2)
        # Case with only two classes
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        #Case with more than two clases
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;

# Normalization

using Statistics;

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # eliminate any atribute that do not add information
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Remove any atribute that do not have information
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset;
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset , calculateZeroMeanNormalizationParameters(dataset));
end;
