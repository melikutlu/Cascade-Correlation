function batches = buildMiniBatchOrder(numSamples, batchSize)
    if batchSize >= numSamples
        batches = {1:numSamples};
        return;
    end
    order = randperm(numSamples);
    numBatches = ceil(numSamples / batchSize);
    batches = cell(numBatches,1);
    for k=1:numBatches
        idxStart = (k-1)*batchSize + 1;
        idxEnd = min(k*batchSize, numSamples);
        batches{k} = order(idxStart:idxEnd);
    end
end
