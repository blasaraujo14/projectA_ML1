# Libraries used
using DataFrames;
using CSV;
using Statistics;
using Random;
using StatsPlots;
using ScikitLearn;
@sk_import decomposition:PCA;
@sk_import impute: KNNImputer;
include("utils/visualizations.jl");
include("utils/preprocessing.jl");

#############
# Load data #
#############

println("Loading data...");
support2 = CSV.read("datasets/support2_cleaned.csv", DataFrame, delim = ',');
support2[:,"hospdeath"] = support2[:,"death"] + support2[:,"hospdead"];
target_cols = ["death", "hospdead", "hospdeath"];
println("Done.");
println();

##############
# Imputation #
##############

# Plot using imputed data.
println("Imputing missing data...");
inputs = Array(support2[!,Not(target_cols)]);
targets = convert(Array{Bool, 1}, Array(support2[!,:death]));

imputer = KNNImputer(n_neighbors = 2);
inputs[:,1:32] = fit_transform!(imputer, inputs[:,1:32]);
println("Done.");
println();

##########
# Scaler #
##########

println("Normalizing data...");
meanTrain, stdTrain = calculateZeroMeanNormalizationParameters(inputs[:,1:32]);
inputs[:,1:32] = normalizeZeroMean!(inputs[:,1:32], (meanTrain, stdTrain));
println("Done.");
println();

##################
# Visualizations #
##################

plot_correlations(inputs[:,1:32]);
savefig("plots/correlations.png");

# Train PCA on training dataset.
pca = PCA(2);
pcaInputs = fit_transform!(pca, inputs[:,1:32]);

# Draw results
draw_results(pcaInputs, targets; colors=[:green,:red], target_names=["Survived", "Died"]);
savefig("plots/pca.png");

##########
# Models #
##########
println("Obtaining model results...");
println("Done.");