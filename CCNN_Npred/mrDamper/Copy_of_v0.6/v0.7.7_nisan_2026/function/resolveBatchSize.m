function batchSize = resolveBatchSize(config, fieldName, numSamples)
    batchSize = numSamples;
    if isfield(config, 'training') && isfield(config.training, fieldName)
        candidateSize = config.training.(fieldName);
        if isnumeric(candidateSize) && candidateSize > 0
            batchSize = min(numSamples, max(1, round(candidateSize)));
        end
    end
end
