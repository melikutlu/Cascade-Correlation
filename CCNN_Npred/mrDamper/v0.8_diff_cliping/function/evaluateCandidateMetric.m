function metricVal = evaluateCandidateMetric(w_h, X0, U, T, W_hidden, w_o, g, config)
    metric = candidateCorrelationMetric(w_h, X0, U, T, W_hidden, w_o, g, config);
    metricVal = gather(extractdata(metric));
end
