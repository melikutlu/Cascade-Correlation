function [L, metric, grad] = loss_candidate_corr(w_h, X0, U, T, W_hidden, w_o, g, config, candOrder)
    if nargin < 9
        candOrder = 1;
    end
    metric = candidateCorrelationMetric(w_h, X0, U, T, W_hidden, w_o, g, config, candOrder);
    %(Karesini alarak her iki yöndeki büyümeyi de ödüllendiriyoruz):
    L = -(metric^2);
    grad = dlgradient(L, w_h);
end
