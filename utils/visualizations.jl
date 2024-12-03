
# Obtains correlation matrix between numerical inputs and plots it
function plot_correlations(inputs)
    M = cor(Matrix(inputs))
    (n,m) = size(M)
    heatmap(M, fc=cgrad([:white,:dodgerblue4]), xticks=(1:m,m), xrot=90, yticks=(1:m,m), yflip=true)
end;

# Function to plot the proportion of classes in target variable
function pie_feature(dataset::DataFrame, feature::Symbol, labels::Vector{String}, title::String)
    value_counts = combine(groupby(dataset, feature), nrow => :count);
    values = value_counts.count;   
    pie(labels, values, title="Proportion of deaths", autopct="%1.1f%%")
end;

# Function to plot number of deaths by a categorical variable
function plotCountDeaths(dataset::DataFrame, feature::Symbol, label::Matrix{String}, title::String)
    value_counts = combine(groupby(dataset, feature), nrow => :count,
                                                :death => sum => :sum_dead,
                                                :hospdead => sum => :sum_hosp_dead)
    labels = value_counts[:,feature];  
    total = value_counts.count;   
    deaths = value_counts.sum_dead;
    deaths_hosp = value_counts.sum_hosp_dead;
    groupedbar(labels, [total deaths deaths_hosp], xlabel = feature, ylabel = "Count",
        title = title, label = label, legend = true)
end;

# Function to plot results from PCA with 2 components
function draw_results(x, y; colors=nothing, target_names=nothing)
    num_classes = length(unique(colors))

    if !isnothing(target_names)
        @assert num_classes == length(target_names)
        label = target_names
    else
        label = [string(i) for i in 1:num_classes]
    end

    fig = plot()
    if (num_classes == 2)
        possitive_class = y[:,1].==1
        scatter!(fig, x[possitive_class,1], x[possitive_class,2], markercolor=colors[1], label=label[1])
        scatter!(fig, x[.!possitive_class,1], x[.!possitive_class,2], makercolor=colors[2], label=label[2])
    else
        for i in 1:num_classes
            index_class = y[:,i].==1
            scatter!(fig, x[index_class, 1], x[index_class, 2], markercolor=colors[i], label=label[i])
        end
    end
end;

# Function to display the confusion matrix of a model
function displayConfMat(confMat, classes)
    (n,m) = size(confMat)
    heatmap(confMat, fc=cgrad([:white,:dodgerblue4]), xticks=(1:m,classes), yrot=90, yticks=(1:m,classes), 
            title="Confussion matrix", xlabel="Target", ylabel="Prediction", yflip=true)
    # add number of coincidences in each cell
    annotate!([(j, i, text(round(confMat[i,j]), 8,"Computer Modern",:black)) for i in 1:n for j in 1:m])
end;

