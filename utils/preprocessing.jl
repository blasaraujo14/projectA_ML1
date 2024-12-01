#################################
# Data cleaning with DataFrames #
#################################

# Returns columns with more than 35% of the data missing
function getMissingColumns(df::DataFrame, percentage::Float64)
    summary = describe(df)
    feats_out = summary[summary.nmissing .> floor(nrow(df)*percentage), "variable"]
    return feats_out;
end;

# Function to OHE categorical features with a dataframe
function dfOneHotEncoding!(df::DataFrame, feature::String, classes::AbstractArray{<:Any,1})
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in df[:,feature]]));

    # Second defensive statement, check the number of classes
    numClasses = length(classes);
    @assert(numClasses>1)
    #Case with more than two clases
    nrows = length(df[:,feature])
    
    for numClass = 1:numClasses
        df[:, feature*"_"*classes[numClass]] .= (df[:,feature].==classes[numClass]) .* 1;
    end;

    df = select(df, Not(feature))

    return df;
end;

#Equivalent to function dfOneHotEncoding(feature::AbstractArray{<:Any,1})
dfOneHotEncoding!(df::DataFrame, feature::String) = dfOneHotEncoding!(df, feature, unique(df[:,feature]));

# Function to map ordinal string features to int values given in a Dict.
function dfOrdinalEncoding!(df::DataFrame, feature::String, ordDict::Dict{})

    # Map values of dict, if not found returns missing
    df[:,feature*"_ord"] = map(x -> get(ordDict, x, missing), df[:, feature])

    df = select(df, Not(feature))

    return df
end;


############
# Holdout  #
############

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

####################
# One-hot encoding #
####################

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

#################
# Normalization #
#################

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

using ScikitLearn;
@sk_import feature_selection: SelectKBest; # Feature Selection
@sk_import feature_selection: f_classif; # Used with SelectKBest

function prepareDataForFitting(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
                                testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
                                validationRatio::Float64 = 0.)

    # split training into training and validation sets
    if validationRatio > 0.0
        indicesT, indicesV = holdOut(size(trainingDataset[1], 1), validationRatio)
        (inputsAux, targetsAux) = trainingDataset;
        validationDataset = inputsAux[indicesV,:], targetsAux[indicesV]
        trainingDataset = inputsAux[indicesT,:], targetsAux[indicesT]

        normParams = calculateZeroMeanNormalizationParameters(trainingDataset[1])
        normalizeZeroMean!(validationDataset[1], normParams)
    else
        # empty validation set
        validationDataset = (Array{Float64}(undef, 0, 0), Array{Any}(undef, 0))

        normParams = calculateZeroMeanNormalizationParameters(trainingDataset[1])
    end;

    normalizeZeroMean!(trainingDataset[1], normParams)
    normalizeZeroMean!(testDataset[1], normParams)

    # reducer = PCA(0.85)
    reducer = SelectKBest(f_classif, k=10);

    #Once it is ajusted it can be used to transform the data
    trainingDataset = (fit_transform!(reducer, trainingDataset[1]), trainingDataset[2]);
    testDataset = (reducer.transform(testDataset[1]), testDataset[2]);

    if validationRatio > 0.0
        validationDataset = selectK.transform(validationDataset);
    end;

    return trainingDataset, validationDataset, testDataset;
end;
