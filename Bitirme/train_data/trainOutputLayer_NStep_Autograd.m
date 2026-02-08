function [w_o_trained, E_final, current_mse] = trainOutputLayer_NStep_Autograd(X_bias, T, w_o_init, max_epochs, params, config)
    w_dl = dlarray(w_o_init);
    
    % N-step için gerekli parametreler
    N = config.model.n_step;
    
    for epoch = 1:max_epochs
        % Gradyan hesapla (Argümanları config üzerinden iç fonksiyona pasla)
        [loss, grad] = dlfeval(@calculate_n_step_loss, w_dl, X_bias, T, config);
        
        % Ağırlık güncelleme
        w_dl = w_dl - params.eta_output_gd * grad;
        
        if mod(epoch, 50) == 0 || epoch == 1
            fprintf('Epoch %d | N-Step MSE: %.6f\n', epoch, extractdata(loss));
        end
    end
    
    w_o_trained = extractdata(w_dl);
    current_mse = extractdata(loss);
    E_final = T - (X_bias * w_o_trained);
end

function [total_loss, grad] = calculate_n_step_loss(w_dl, X_bias, T, config)
    N = config.model.n_step;
    batch_size = config.model.batch_size;
    [num_samples, num_features] = size(X_bias);
    
    % N adım ileri bakacağımız için sınırları belirle
    idx = randi([1, num_samples - N], batch_size, 1);
    sample_loss = dlarray(0);
    
    for i = 1:batch_size
        start_idx = idx(i);
        curr_x = X_bias(start_idx, :); 
        
        for step = 1:N
            % 1. Tahmin
            y_pred = curr_x * w_dl;
            
            % 2. Hata biriktir
            target = T(start_idx + step - 1, :);
            sample_loss = sample_loss + sum((y_pred - target).^2);
            
            % 3. Regresör Güncelle (Feedback)
            if step < N
                % Bir sonraki adımın u(k+1) bilgisini GERÇEK veriden al
                next_raw_x = X_bias(start_idx + step, :);
                
                % Kendi tahminimizi (y_pred) regresöre yerleştir
                curr_x = update_regressor_with_pred(next_raw_x, y_pred, config);
            end
        end
    end
    
    total_loss = sample_loss / (batch_size * N);
    grad = dlgradient(total_loss, w_dl);
end