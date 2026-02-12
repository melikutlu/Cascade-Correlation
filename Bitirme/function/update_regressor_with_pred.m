function next_x = update_regressor_with_pred(current_x, y_pred, config)
    % current_x: [1 (bias), u(k), y(k-1)...]
    next_x = current_x;
    na = config.regressors.na;
    nb = config.regressors.nb;
    
    % Bias kontrolü (config'de false olsa bile matriste olabilir)
    % createRegressorsDynamic çıktısına göre ayarlanmalı
    has_bias = (size(current_x, 2) > (na + nb + 1)); 
    
    % y(k-1)'in başladığı indeks: Bias + Giriş Gecikmeleri
    start_idx_y = double(has_bias) + (nb + 1); 
    
    if na > 0
        % Eğer birden fazla gecikme varsa kaydır: y(k-2) -> y(k-1)
        if na > 1
            next_x(start_idx_y + 1 : start_idx_y + na - 1) = ...
                next_x(start_idx_y : start_idx_y + na - 2);
        end
        % Yeni tahmini y(k-1) yerine yaz
        next_x(start_idx_y) = y_pred;
    end
end