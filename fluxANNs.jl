using Flux.Losses
using Flux

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;


function trainClassANN(topology::AbstractArray{<:Int,1},
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0));
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
            maxEpochsVal::Int=20, showText::Bool=false)

    #(inputs, targets) = dataset;
    (inputsTrain, targetsTrain) = trainingDataset;
    (inputsTest, targetsTest) = testDataset;
    (inputsVal, targetsVal) = validationDataset;

    # This function assumes that each sample is in a row
    # we are going to check the number of samples to have same inputs and targets
    @assert(size(inputsTrain,1)==size(targetsTrain,1));
    @assert(size(inputsTest,1)==size(targetsTest,1));
    @assert(size(inputsVal,1)==size(targetsVal,1));

    isValEmpty = 0 == size(inputsVal, 1);

    # We define the ANN
    ann = buildClassANN(size(inputsTrain,2), topology, size(targetsTrain,2));

    # Setting up the loss funtion to reduce the error
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    # This vectos is going to contain the losses and precission on each training epoch
    trainingLosses = Float32[];
    validationLosses = Float32[];
    testLosses = Float32[];

    # Inicialize the counter to 0
    numEpoch = 0;
    # Calcualte the loss without training
    trainingLoss = loss(ann, inputsTrain', targetsTrain');
    push!(trainingLosses, trainingLoss);

    testLoss = loss(ann, inputsTest', targetsTest');
    push!(testLosses, testLoss);

    #  and give some feedback on the screen
    if(showText)
        print("Epoch ", numEpoch, ": training loss: ", trainingLoss);
    end
    if (!isValEmpty)
        validationLoss = loss(ann, inputsVal', targetsVal');
        push!(validationLosses, validationLoss);
        if(showText)
            print(", Validation loss: ", validationLoss);
        end
    end;
    if(showText)
        println()
    end

    # Define the optimazer for the network
    opt_state = Flux.setup(Adam(learningRate), ann);

    valLossCounter = 0;
    bestValLoss = Inf;

    # Start the training until it reaches one of the stop critteria
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && ((valLossCounter < maxEpochsVal) | isValEmpty)

        # For each epoch, we habve to train and consequently traspose the pattern to have then in columns
        Flux.train!(loss, ann, [(inputsTrain', targetsTrain')], opt_state);

        numEpoch += 1;
        # calculate the loss for this epoch
        trainingLoss = loss(ann, inputsTrain', targetsTrain');
        testLoss = loss(ann, inputsTest', targetsTest');
        # store it
        push!(trainingLosses, trainingLoss);
        push!(testLosses, testLoss);
        if(showText)
            print("Epoch ", numEpoch, ": training loss: ", trainingLoss);
        end

        if (!isValEmpty)
            validationLoss = loss(ann, inputsVal', targetsVal');
            push!(validationLosses, validationLoss);
            if (validationLoss < bestValLoss)
                bestValLoss = validationLoss;
                valLossCounter = 0;
                bestAnn = deepcopy(ann);
            else
                valLossCounter += 1;
            end;
            if(showText)
                print( ", validation loss: ", validationLoss);
            end
        end;
        # shown it
        if(showText)
            println();
        end
    end;
    # return the network and the evolution of the error
    if (isValEmpty)
        bestAnn = deepcopy(ann);
    end;
    return (bestAnn, trainingLosses, validationLosses, testLosses)
end;
