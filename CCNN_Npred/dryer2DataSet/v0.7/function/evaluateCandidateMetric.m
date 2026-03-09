function metricVal = evaluateCandidateMetric(w_h, X0, U, T, W_hidden, w_o, g, config, candOrder)
    if nargin < 9
        candOrder = 1;
    end
    metric = candidateCorrelationMetric(w_h, X0, U, T, W_hidden, w_o, g, config, candOrder);
    metricVal = gather(extractdata(metric));
end
