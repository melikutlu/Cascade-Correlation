function [w_c_trained, v_c_final] = trainCandidateUnit_NStep_Autograd(X_bias, T, E_residual, max_epochs, params, config,g,eta_candidate)
    % 1. Ağırlıkları Başlat
    [~, num_features] = size(X_bias);
    w_c_init = randn(num_features, 1) * 0.01;
    w_c_dl = dlarray(w_c_init);
    
    E_residual_dl = dlarray(E_residual);

    % 3. Eğitim Döngüsü
    for epoch = 1:max_epochs
        % Gradyan hesapla
        [neg_corr, grad] = dlfeval(@calculate_n_step_correlation, w_c_dl, X_bias, T, E_residual_dl, config, g);
        
        % Gradient ASCENT yapıyoruz (Korelasyonu maksimize et)
        % neg_corr'u minimize etmek korelasyonu maksimize eder
        w_c_dl = w_c_dl - eta_candidate * grad;
        
        if mod(epoch, 50) == 0
            fprintf('  Aday Epoch %d | Skor (S^2): %.6f\n', epoch, -extractdata(neg_corr));
        end
    end
    
    w_c_trained = extractdata(w_c_dl);
    v_c_final = g(X_bias * w_c_trained);
end

function [neg_total_corr, grad] = calculate_n_step_correlation(w_c_dl, X_bias, T, E_residual, config, g)
    N = config.model.n_step;
    batch_size = config.model.batch_size;
    [num_samples, ~] = size(X_bias);
    
    idx = randi([1, num_samples - N], batch_size, 1);
    total_corr_sq = dlarray(0);
    
    for i = 1:batch_size
        start_idx = idx(i);
        curr_x = dlarray(X_bias(start_idx, :)); 
        
        step_correlation = dlarray(0);
        
        for step = 1:N
            % Aday birim çıktısı
            v_c = g(curr_x * w_c_dl);
            
            % O anki adımdaki hata
            e_step = E_residual(start_idx + step - 1, :);
            
            % Korelasyonu biriktir (v * error)
            step_correlation = step_correlation + sum(v_c * e_step);
            
            % Regresör Güncelle (N-step feedback simülasyonu)
            if step < N
                next_raw_x = X_bias(start_idx + step, :);
                % Aday eğitiminde kararlılık için gerçek T ile besleme (Teacher Forcing)
                curr_x = update_regressor_with_pred(next_raw_x, T(start_idx + step - 1, :), config);
            end
        end
        % Adımların toplam korelasyonunun karesini al (Maksimize edilecek hedef)
        total_corr_sq = total_corr_sq + step_correlation^2;
    end
    
    % Ortalama negatif korelasyon (Minimize edilince gerçek korelasyon artar)
    neg_total_corr = - (total_corr_sq / batch_size);
    grad = dlgradient(neg_total_corr, w_c_dl);
end