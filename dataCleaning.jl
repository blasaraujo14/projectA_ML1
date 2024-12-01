#############
# Libraries #
#############
using DataFrames;
using CSV;
include("utils/preprocessing.jl");

#############
# Load data #
#############

# Load dataset
dataset = CSV.read("datasets/support2.csv", DataFrame, delim = ',');
dataset = select(dataset, Not("id"))
support2 = copy(dataset);

#################
# Data Cleaning #
#################

# Eliminate features with more than 35% of data missing.
featsOut = getMissingColumns(support2, 0.35);
support2 = select(support2, Not(featsOut))

# Eliminate missings in race and dnr columns (only categoricals with missing and few missing values).
support2 = dropmissing(support2, [:dnr,:race])

println("Features original dataset: ", ncol(dataset))
println("Intances original dataset: ", nrow(dataset))
println();
println("Features with more than 35% missing data: ", featsOut);
println("Features after transform: ", ncol(support2))
println("Instances after transform: ", nrow(support2))
println();

#################
# Preprocessing #
#################

# Partition by data type to perform preprocessing
catNames= ["sex", "dzgroup", "dzclass", "race", "dnr", "dementia", "diabetes"]
targetName = ["death", "hospdead"]
numNames = num_feats = names(select(support2, Not(catNames, targetName)), Union{Missing, Number});

support2Target = select(support2, targetName);
support2Cat = select(support2, catNames);
support2Num = select(support2, Not(targetName, catNames));

#-------------------------
# Categorical
#-------------------------

# Dementia and diabetes are already in OHE format
for cat in catNames[1:end-2]
    global support2Cat = dfOneHotEncoding!(support2Cat, cat)
end;

catNames = names(support2Cat)

#-------------------------
# Numerical
#-------------------------

# Some ordinal features do not have appropiate format, so we correct it
ordinalDict = Dict("income" => Dict(missing => missing,
                                   "under \$11k" => 1,
                                    "\$11-\$25k" => 2,
                                    "\$25-\$50k" => 3,
                                    ">\$50k" => 4),
                    "sfdm2" => Dict(missing => missing,
                                    "no(M2 and SIP pres)" => 1,
                                    "adl>=4 (>=5 if sur)" => 2,
                                    "SIP>=30" => 3,
                                    "Coma or Intub" => 4,
                                    "<2 mo. follow-up" => 5),
                    "ca" => Dict("no" => 1,
                                 "yes" => 2,
                                 "metastatic" => 3))

for (feat, ordDict) in ordinalDict
    global support2Num = dfOrdinalEncoding!(support2Num, feat, ordDict)
end;

# Change missing by NaN for using imputer.
support2Num = coalesce.(support2Num, NaN);

# Save results in csv.
support2Clean = hcat(support2Num, support2Cat, support2Target);
CSV.write("datasets/support2_cleaned.csv", support2Clean);
println("Saved cleaned file at datasets/support2_cleaned.csv");