function maxLag = getMaxLagFromRegressors(regressors)
    ulags = regressors.u(:)';
    ylags = regressors.y(:)';
    maxLag = 0;
    posULags = ulags(ulags>0);
    if ~isempty(posULags)
        maxLag = max(maxLag, max(posULags));
    end
    if ~isempty(ylags)
        maxLag = max(maxLag, max(ylags));
    end
end
