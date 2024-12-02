
# Trains models using the julia wrapper for scikit-learn
function fitScikitModel(modelType::Symbol, modelHyperparameters::Dict, 
                        (inputsTrain, targetsTrain)::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}})
    
    if modelType == :SVM
        model = SVC(kernel=modelHyperparameters["kernel"], 
                    degree=modelHyperparameters["degree"], 
                    gamma=modelHyperparameters["gamma"], 
                    C=modelHyperparameters["C"]);
    elseif modelType == :DTree
        model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], 
                                        random_state=1);
    elseif modelType == :KNN
        model = KNeighborsClassifier(modelHyperparameters["k"]);
    else
        println("Unknown model type ", modelType)
        return -1
    end;
    
    fit!(model, inputsTrain, targetsTrain);
    
    return model
end

# Trains a model and obtains its confusion matrix in test
function fitAndConfusion(modelType::Symbol, modelHyperparameters::Dict, 
                        (inputsTrain, targetsTrain)::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
                        (inputsVal, targetsVal)::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
                        (inputsTest, targetsTest)::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
                        classes::AbstractArray{<:Any,1})
    
    if modelType == :ANN
        targetsTrain = oneHotEncoding(targetsTrain, classes)
        targetsVal = oneHotEncoding(targetsVal, classes)
        targetsTest = oneHotEncoding(targetsTest, classes)
        
        (ann, _) = trainClassANN(modelHyperparameters["topology"], (inputsTrain, targetsTrain), 
                                (inputsVal, targetsVal), (inputsTest, targetsTest);
                                maxEpochs=modelHyperparameters["maxEpochs"], 
                                minLoss=modelHyperparameters["minLoss"],
                                learningRate=modelHyperparameters["learningRate"],
                                maxEpochsVal=modelHyperparameters["maxEpochsVal"]);
        
        outputsTest = ann(inputsTest')';
    else
        model = fitScikitModel(modelType, modelHyperparameters, (inputsTrain, targetsTrain)) 
        outputsTest = predict(model, inputsTest);
    end
    
    return confusionMatrix(outputsTest, targetsTest; weighted=true)
end

# Applies model cross validation to a given model
function modelCrossValidation(modelType::Symbol,
        modelHyperparameters::Dict,
        inputs::AbstractArray{<:Real,2},
        targets::AbstractArray{<:Any,1},
        crossValidationIndices::Array{Int64,1};
        reduceDimensions::Bool = false)
    classes = unique(targets)
    trainingDataset = (inputs, targets);
    repetitionsTraining = 1

    if modelType == :ANN
        # set to more than one because ANNs are non-deterministic
        repetitionsTraining = 10
    end;
    
    k = kFoldIndices[argmax(crossValidationIndices)]
    results = zeros(k, 2)
    
    for testGroup=1:k
        metrics = []
        
        # computing training, test
        indicesTraining = findall(!=(testGroup), kFoldIndices)
        indicesTest = findall(==(testGroup), kFoldIndices)

        trainingDataset = inputs[indicesTraining,:], targets[indicesTraining]
        testDataset = inputs[indicesTest,:], targets[indicesTest]

        # standardization and validation set computation if needed
        valRatio = if modelType != :ANN 0. else modelHyperparameters["validationRatio"] end
        trainingDataset, validationDataset, testDataset = prepareDataForFitting(trainingDataset, testDataset, valRatio; reduceDimensions=reduceDimensions)
        
        for _ = 1:repetitionsTraining
            (accur, _, _, _, _, _, fScore) = fitAndConfusion(modelType, modelHyperparameters, 
                                                    trainingDataset, validationDataset, testDataset, classes);
            push!(metrics, [accur, fScore])
        end;
        
        results[testGroup, :] = mean(metrics)
    end;
    
    return mean(results, dims=1), std(results, dims=1)
end;

# Applies cross validation to an ensemble model
function trainClassEnsemble(estimators::AbstractArray{Symbol,1},
        modelsHyperParameters:: AbstractArray{Dict, 1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        kFoldIndices::     Array{Int64,1};
        reduceDimensions::Bool = false)
    (inputs, targets) = trainingDataset;
    classes = unique(targets)
    
    k = kFoldIndices[argmax(kFoldIndices)]
    results = zeros(k, 2)
    
    for testGroup=1:k
        # computing training and test sets
        indicesTraining = findall(!=(testGroup), kFoldIndices)
        indicesTest = findall(==(testGroup), kFoldIndices)
        trainingDataset = inputs[indicesTraining,:], targets[indicesTraining]
        testDataset = inputs[indicesTest,:], targets[indicesTest]

        # standardization is applied
        trainingDataset, _, testDataset = prepareDataForFitting(trainingDataset, testDataset; reduceDimensions=reduceDimensions)

        trainInputs, trainTargets = trainingDataset
        testInputs, testTargets = testDataset

        ensemble = fitEnsemble(trainingDataset, estimators, modelsHyperParameters)
        outputsTest = predict(ensemble, testInputs);

        (accur, _, _, _, _, _, fScore) = confusionMatrix(outputsTest, testTargets; weighted=true)
        results[testGroup, :] = [accur, fScore]
    end;
    return mean(results, dims=1), std(results, dims=1)
end;

# Trains an ensemble model
function fitEnsemble(trainingDataset, estimators, modelsHyperParameters)
    # each individual model is fitted
    modelsEnsemble = []
    for i = 1:length(estimators)
        modelType = estimators[i]
        modelHyperparameters = modelsHyperParameters[i]

        model = fitScikitModel(modelType, modelHyperparameters, trainingDataset);

        push!(modelsEnsemble, (string(modelType)*string(i), model))
    end;

    trainInputs, trainTargets = trainingDataset
    # the ensemble is built, trained and tested
    ensemble = StackingClassifier(estimators=[(name,model) for (name, model) in modelsEnsemble],
                final_estimator = SVC(probability=true), n_jobs=-1)
    fit!(ensemble, trainInputs, trainTargets)

    return ensemble
end;

# Function to print better the results of cross validation
function printCrossValOutput(((accur, fScore), (stdAccur, stdFScore)))
    accPerc, fScorePerc =  @sprintf("%.2f%%", accur * 100), @sprintf("%.2f%%", fScore * 100);
    accStdPerc, fScoreStdPerc =  @sprintf("%.2f%%", stdAccur * 100), @sprintf("%.2f%%", stdFScore * 100);
    println("Accuracy:", accur, " (", accPerc, "), stdDev:", stdAccur, " (", accStdPerc, ")");
    println("F1-Score:", fScore, " (", fScorePerc, "), stdDev:", stdFScore, " (", fScoreStdPerc, ")");
    println();

    return accur, fScore
end;

# Applies crosvalidation for given models and return best configuration
function findBestModel(models, trainInputs, trainTargets, kFoldIndices; reduceDimensions::Bool = false)
    bestMetricYet = 0;
    bestModel = -1;
    
    for (modelType, paramList) in models
        cnt = 1;
        println();
        println("Training ", modelType, " models:");
        for params in paramList
            println("Configuration ", cnt, ": ", params);
            accur, fScore = printCrossValOutput(modelCrossValidation(modelType, params, trainInputs,
                                        trainTargets, kFoldIndices; reduceDimensions=reduceDimensions));
            
            cnt += 1;
            #println(modelType, (accur, fScore), (stdAccur, stdFScore))
            if accur * fScore > bestMetricYet
                bestMetricYet = accur * fScore;
                bestModel = (modelType, params);
            end;
        end;
    end;
    
    return bestModel
end;
