# Libraries
using DataFrames;
using CSV;
using Random;
using Statistics;
using ScikitLearn;
using Flux.Losses;
using Flux;
using Plots;
using StatsPlots;
using Printf;

@sk_import impute: KNNImputer; # Imputation of missing values.
@sk_import svm: SVC;
@sk_import decomposition: PCA;
@sk_import tree: DecisionTreeClassifier;
@sk_import neighbors: KNeighborsClassifier;
@sk_import feature_selection: SelectKBest; # Feature Selection
@sk_import feature_selection: f_classif; # Used with SelectKBest
@sk_import ensemble:StackingClassifier;

include("utils/fluxANNs.jl"); # for ANN training with Flux
include("utils/evaluation.jl"); # for confusion matrix calculation and cross validation set partitioning
include("utils/preprocessing.jl"); # for normalization, one-hot encoding, holdout division
include("utils/training.jl"); # for crossvalidation methodology
include("utils/visualizations.jl"); # for plots

# Set seed
Random.seed!(10);

#############
# Load data #
#############

println("Preparing data...");
# loading the dataset
support2 = CSV.read("datasets/support2_cleaned.csv", DataFrame, delim = ',');
# fate: 0 = recovery, 1 = death at home, 2 = death at the hospital
support2[:, "fate"] = support2[:,"death"] + support2[:,"hospdead"];
target_cols = ["death", "hospdead", "fate"];

#####################
# HoldOut partition #
#####################

trainIndex, testIndex = holdOut(nrow(support2), 0.2);

trainInputs = Array(support2[trainIndex, Not(target_cols)]);
testInputs = Array(support2[testIndex, Not(target_cols)]);

# Targets depend on the approach selected
trainTargets = Array(support2[trainIndex, "death"]);
testTargets = Array(support2[testIndex, "death"]);

# Imputation of missing data
imputer = KNNImputer(n_neighbors = 5);
trainInputs[:,1:32] = fit_transform!(imputer, trainInputs[:,1:32]);
testInputs[:,1:32] = imputer.transform(testInputs[:,1:32]);

##################
# Visualizations #
##################

approachName = "binary"
# ----------------------
# Correlations
# ----------------------
meanTrain, stdTrain = calculateZeroMeanNormalizationParameters(trainInputs[:,1:32]);
numInputs = normalizeZeroMean!(trainInputs[:,1:32], (meanTrain, stdTrain));
plot_correlations(numInputs);
savefig("plots/" * approachName * "/correlations.png");

# ----------------------
# PCA
# ----------------------
pca = PCA(2);
pcaInputs = fit_transform!(pca, numInputs);
draw_results(pcaInputs, trainTargets; colors=[:green,:red], target_names=["Died", "Survived"]);

savefig("plots/" * approachName * "/pcaVisualization.png");
println("Done");

####################
# Cross-validation #
####################
println();
println("####################");
println("# Cross-validation #");
println("####################");

kFoldIndices = crossvalidation(trainTargets, 5);

# ----------------------
# ANN
# ----------------------
ANNparams = [Dict("topology" => [2], "maxEpochs" => 500, "minLoss" => 0., "learningRate" => 0.01,
                "maxEpochsVal" => 20, "validationRatio" => 0.1),
            Dict("topology" => [4], "maxEpochs" => 500, "minLoss" => 0., "learningRate" => 0.01,
                "maxEpochsVal" => 20, "validationRatio" => 0.1),
            Dict("topology" => [8], "maxEpochs" => 500, "minLoss" => 0., "learningRate" => 0.01,
                "maxEpochsVal" => 20, "validationRatio" => 0.1),
            Dict("topology" => [16], "maxEpochs" => 500, "minLoss" => 0., "learningRate" => 0.01,
                "maxEpochsVal" => 20, "validationRatio" => 0.1),
            Dict("topology" => [2, 2], "maxEpochs" => 500, "minLoss" => 0., "learningRate" => 0.01,
                "maxEpochsVal" => 20, "validationRatio" => 0.1),
            Dict("topology" => [4, 4], "maxEpochs" => 500, "minLoss" => 0., "learningRate" => 0.01,
                "maxEpochsVal" => 20, "validationRatio" => 0.1),
            Dict("topology" => [8, 8], "maxEpochs" => 500, "minLoss" => 0., "learningRate" => 0.01,
                "maxEpochsVal" => 20, "validationRatio" => 0.1),
            Dict("topology" => [16, 16], "maxEpochs" => 500, "minLoss" => 0., "learningRate" => 0.01,
                "maxEpochsVal" => 20, "validationRatio" => 0.1)]

models = [(:ANN, ANNparams)];

(modelType, params) = findBestModel(models, trainInputs, trainTargets, kFoldIndices; reduceDimensions=true);
println("Best model is ", modelType, " with hyperparameters:");
println(params);

bestANNparams = params

# ----------------------
# SVM
# ----------------------
SVMparams = [Dict("kernel" => "linear", "degree" => 0, "gamma" => "scale", "C" => 0.1),
            Dict("kernel" => "linear", "degree" => 0, "gamma" => "scale", "C" => 1),
            Dict("kernel" => "rbf", "degree" => 0, "gamma" => "scale", "C" => 0.1),
            Dict("kernel" => "rbf", "degree" => 0, "gamma" => "scale", "C" => 1),
            Dict("kernel" => "sigmoid", "degree" => 0, "gamma" => "scale", "C" => 0.1),
            Dict("kernel" => "sigmoid", "degree" => 0, "gamma" => "scale", "C" => 1),
            Dict("kernel" => "poly", "degree" => 3, "gamma" => "scale", "C" => 0.1),
            Dict("kernel" => "poly", "degree" => 3, "gamma" => "scale", "C" => 1)]

models = [(:SVM, SVMparams)];

(modelType, params) = findBestModel(models, trainInputs, trainTargets, kFoldIndices; reduceDimensions=false);
println("Best model is ", modelType, " with hyperparameters:");
println(params);

bestSVMparams = params

# ----------------------
# DTree
# ----------------------
DTreeParams = [Dict("maxDepth" => 4), Dict("maxDepth" => 8), Dict("maxDepth" => 16),
                Dict("maxDepth" => 32), Dict("maxDepth" => 64), Dict("maxDepth" => 128)]

models = [(:DTree, DTreeParams)];

(modelType, params) = findBestModel(models, trainInputs, trainTargets, kFoldIndices; reduceDimensions=false);
println("Best model is ", modelType, " with hyperparameters:");
println(params);

bestDTreeParams = params


# ----------------------
# KNN
# ----------------------
KNNparams = [Dict("k" => 3), Dict("k" => 6), Dict("k" => 12),
            Dict("k" => 24), Dict("k" => 48), Dict("k" => 96)]

models = [(:KNN, KNNparams)];

(modelType, params) = findBestModel(models, trainInputs, trainTargets, kFoldIndices; reduceDimensions=false);
println("Best model is ", modelType, " with hyperparameters:");
println(params);

bestKNNparams = params


# ----------------------
# Ensemble
# ----------------------

# An ANN flux model cannot be included in a Scikit ensemble due to PyCall wrapping issues
estimators = [:KNN, :DTree, :SVM]


#=
bestDTreeParams = Dict("maxDepth" => 4)
bestKNNparams = Dict("k" => 24)
bestSVMparams = Dict("C" => 1, "kernel" => "rbf", "gamma" => "scale", "degree" => 0)
=#

params = Vector{Dict}([bestKNNparams, bestDTreeParams, bestSVMparams])

println("Training ensemble consisting of best performing models:")
printCrossValOutput(trainClassEnsemble(estimators, params, (trainInputs, trainTargets),
                   crossvalidation(trainTargets, 5); reduceDimensions=false));


################
# Test results #
################
println();
println("Saving confusion matrix of best models and ensemble");

# Ensemble of best models
train = (trainInputs, trainTargets); test = (testInputs, testTargets);
# standardization is applied
trainNorm, _, testNorm = prepareDataForFitting(train, test; reduceDimensions=false);
ensemble = fitEnsemble(trainNorm, estimators, params);
classes = unique(trainTargets);

matAndMetrics = confusionMatrix(predict(ensemble, testNorm[1]), testNorm[2]; weighted=true);
confMat = matAndMetrics[8];

# Plot confusion matrix and save it.

class_map = Dict(0 => "Recovery", 1 => "Death");
classNames = [class_map[class] for class in classes];
displayConfMat(confMat, classNames);
savefig("plots/" * approachName * "/ensembleConfusionMatrix.png");

estimators = [:KNN, :DTree, :SVM, :ANN]
params = Vector{Dict}([bestKNNparams, bestDTreeParams, bestSVMparams, bestANNparams])

# Best models
for i = 1:length(estimators)
    train = (trainInputs, trainTargets); test = (testInputs, testTargets);

    local modelType = estimators[i]
    hyperParams = params[i]
    # normalization and validation set computation if needed
    valRatio = if modelType != :ANN 0. else hyperParams["validationRatio"] end
    train, val, test = prepareDataForFitting(train, test, valRatio)

    classes = unique(trainTargets)
    matAndMetrics = fitAndConfusion(modelType, hyperParams, train, val, test, classes)
    confMat = matAndMetrics[8]

    class_map = Dict(0 => "Recovery", 1 => "Death")
    classNames = [class_map[class] for class in classes]
    displayConfMat(confMat, classNames)
    savefig("plots/" * approachName * "/" * string(modelType) * "ConfusionMatrix.png");
end;

println("Done");

