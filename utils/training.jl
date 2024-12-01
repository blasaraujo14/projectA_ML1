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

function modelCrossValidation(modelType::Symbol,
        modelHyperparameters::Dict,
        inputs::AbstractArray{<:Real,2},
        targets::AbstractArray{<:Any,1},
        crossValidationIndices::Array{Int64,1})
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

        # normalization and validation set computation if needed
        valRatio = if modelType != :ANN 0. else modelHyperparameters["validationRatio"] end
        trainingDataset, validationDataset, testDataset = prepareDataForFitting(trainingDataset, testDataset, valRatio)
        
        for _ = 1:repetitionsTraining
            (accur, _, _, _, _, _, fScore) = fitAndConfusion(modelType, modelHyperparameters, 
                                                    trainingDataset, validationDataset, testDataset, classes);
            push!(metrics, [accur, fScore])
        end;
        
        results[testGroup, :] = mean(metrics)
    end;
    
    return mean(results, dims=1), std(results, dims=1)
end;