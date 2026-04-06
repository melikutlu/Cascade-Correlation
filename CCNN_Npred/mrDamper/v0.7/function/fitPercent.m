function fit = fitPercent(y, yhat)
    fit = 100 * (1 - norm(y - yhat) / norm(y - mean(y)));
end
