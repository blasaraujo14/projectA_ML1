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