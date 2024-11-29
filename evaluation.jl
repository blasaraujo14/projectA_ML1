# Cross validation
function crossvalidation(N::Int64, k::Int64)
    oneToK = collect(1:k)
    repeated = repeat(oneToK, ceil(Int64, N/k))
    return shuffle!(repeated[1:N])
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    @assert(all(sum(targets, dims=1) .>= k))

    indices = zeros(Int64, size(targets, 1))

    numClasses = size(targets, 2)
    for numClass = 1:numClasses
        numClassesOfType = sum(targets[:,numClass])
        subsetsOfRows = crossvalidation(numClassesOfType, k)

        indicesOfClass = findall(==(true), targets[:, numClass])
        indices[indicesOfClass] .= subsetsOfRows
    end

    return indices
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #return crossvalidation(oneHotEncoding(targets), k)

    classes = unique(targets)
    # build a vector of numbers of class occurrences with list comprehension
    @assert(all([length(findall(==(class), targets)) for class in classes] .>= k))

    indices = zeros(Int64, length(targets))

    for class in classes
        numClassesOfType = length(findall(==(class), targets))
        subsetsOfRows = crossvalidation(numClassesOfType, k)

        indicesOfClass = findall(==(class), targets)
        indices[indicesOfClass] .= subsetsOfRows
    end

    return indices
end

# Accuracy calculation
function classifyOutputs(outputs::AbstractArray{<:Real,2};
                        threshold::Real=0.5)
   numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Look for the maximum value using the findmax funtion
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Set up then boolean matrix to everything false while max values aretrue.
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    mean(outputs.==targets);
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;

# Confusion matrix calculation

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # check length is the same
    @assert(size(outputs,1)==size(targets,1));

    # calculate confusion matrix cells
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    confusionMatrix = [TN FP; FN TP]

    # calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    errorRate = 1 - accuracy

    if FP + FN + TP == 0 # every pattern is a TN
        sensitivity = 1
        positivePredictiveValue = 1
    else # check if we can calculate, else 0
        sensitivity = TP + FN == 0 ? 0 : TP / (TP + FN)
        positivePredictiveValue = TP + FP == 0 ? 0 : TP / (TP + FP)
    end

    if FP + FN + TN == 0 # every pattern is a TP
        specificity = 1
        negativePredictiveValue = 1
    else # check if we can calculate, else 0
        specificity = TN + FP == 0 ? 0 : TN / (TN + FP)
        negativePredictiveValue = TN + FN == 0 ? 0 : TN / (TN + FN)
    end

    # F-score = 0 if positivePredictiveValue and sensitivity equal to zero
    fScore = (positivePredictiveValue + sensitivity) == 0 ? 0 :
              2 * (positivePredictiveValue * sensitivity) / (positivePredictiveValue + sensitivity)

    return accuracy, errorRate, sensitivity, specificity, positivePredictiveValue,
            negativePredictiveValue, fScore, confusionMatrix
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # check if num of columns is equal and different from 2
    numClasses = size(targets,2)
    @assert(size(outputs,2) == numClasses);
    @assert(size(outputs,2) != 2);

    sensitivities = zeros(numClasses)
    specificities = zeros(numClasses)
    ppvs = zeros(numClasses)
    npvs = zeros(numClasses)
    fScores = zeros(numClasses)

    for numClass = 1:numClasses
        binOutputs = outputs[:,numClass]
        if (1 in binOutputs) # check if there is at least of pattern of this class
            _, _, sensitivities[numClass], specificities[numClass], ppvs[numClass],
            npvs[numClass], fScores[numClass], _ = confusionMatrix(binOutputs, targets[:,numClass]);
        end;outputs
    end;

    confMat = zeros(Int64, numClasses, numClasses)

    for predClass = 1:numClasses
        for targetClass = 1:numClasses
            # we go cell by cell in the matrix, calculating the sum of model classifications
            # into predClass that were supposed to be targetClass

            # get all row indices in the model output where predClass appears
            predClassI = findall(==(1), outputs[:, predClass])

            # the corresponding target classifications are in the same row indices in 'targets'
            # so we take into account all targetClasses that were actually classified as predClass
            # by summing all 1s across the column of index targetClass
            confMat[predClass,targetClass] = sum(targets[predClassI, targetClass])
        end;
    end;

    function weightedMean(valuesPerClass::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,2})
        numPatterns = size(targets, 1)
        accum = 0

        for numClass = 1:numClasses
            numPatternsOfClass = sum(targets[:, numClass])
            accum += valuesPerClass[numClass] * numPatternsOfClass
        end;

        return accum/numPatterns
    end;

    # choose between arithmetic mean and weighted average for aggregating the class metrics
    aggFunc(valuesPerClass) = (weighted == true) ? weightedMean(valuesPerClass, targets) : mean(valuesPerClass);

    sensitivity = aggFunc(sensitivities); specificity = aggFunc(specificities);
    ppv = aggFunc(ppvs); npv = aggFunc(npvs); fScore = aggFunc(fScores)

    accur = accuracy(outputs, targets)
    errorRate = 1 - accur

    return accur, errorRate, sensitivity, specificity, ppv, npv, fScore, confMat
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    numOutputs = size(outputs, 2);
    classifOuts = classifyOutputs(outputs)

    if numOutputs==1
        return confusionMatrix(classifOuts[:,1], targets[:,1])
    else
        return confusionMatrix(classifOuts, targets; weighted)
    end
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all(x -> x in unique(targets), unique(outputs)));
    return confusionMatrix(oneHotEncoding(outputs, unique(targets)), oneHotEncoding(targets); weighted)
end
