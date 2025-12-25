function [w_o_trained, E_final, current_mse, Y_pred] = trainOutputLayer_NStep_Autograd( ...
    U, Y_actual, w_initial, max_epochs, eta, config, W_hidden, g_func)

    % 1) Veriyi dlarray'e çevir
    U_dl = dlarray(U);
    Y_actual_dl = dlarray(Y_actual);
    w = dlarray(w_initial);
    
    W_hidden_dl = cellfun(@(m) dlarray(m), W_hidden, 'UniformOutput', false);

    % 2) ADAM Parametreleri
    averageGrad = []; averageSqGrad = []; iteration = 0;
    gradDecay = 0.9; sqGradDecay = 0.999; epsilon = 1e-8;

    % 3) Eğitim Döngüsü
    for epoch = 1:max_epochs
        iteration = iteration + 1;
        
        % Loss ve gradyanı hesapla
        [loss, grad_w] = dlfeval(@loss_and_grad_nstep, w, U_dl, Y_actual_dl, config, W_hidden_dl, g_func);
        
        % ADAM GÜNCELLEMESİ
        [w, averageGrad, averageSqGrad] = adamupdate(w, grad_w, ...
            averageGrad, averageSqGrad, iteration, ...
            eta, gradDecay, sqGradDecay, epsilon);
    end

    % 4) Sonuçlar
    w_o_trained = extractdata(w);
    [~, Y_pred_dl] = loss_and_grad_nstep(w, U_dl, Y_actual_dl, config, W_hidden_dl, g_func);
    Y_pred = extractdata(Y_pred_dl);
    
    % Hata hesaplama (boyut uyumu için transpose kontrolü)
    max_lag = max(config.regressors.na, config.regressors.nb + config.regressors.nk);
    target_real = Y_actual(max_lag+1:end, :);
    E_final = target_real - Y_pred;
    current_mse = mean(E_final.^2);
end

function [loss, y_sim_steps] = loss_and_grad_nstep(w, U, Y, config, W_hidden, g)
    na = config.regressors.na;
    nb = config.regressors.nb;
    nk = config.regressors.nk;
    max_lag = max(na, nb+nk);
    N = size(U, 1);
    
    % Y'nin sütun vektörü olduğundan emin ol (N x 1)
    y_sim = Y(1:max_lag, :); 
    
    for k = (max_lag+1):N
        % Girişleri topla
        x_u = U(k-nk:-1:k-nk-nb+1, :)'; % (1 x nb)
        x_y = y_sim(k-1:-1:k-na, :)';   % (1 x na)
        
        % BIAS KONTROLÜ (Kritik Düzeltme)
        if config.regressors.include_bias
            curr_in = [dlarray(1), x_u, x_y];
        else
            curr_in = [x_u, x_y];
        end
        
        % Gizli birimler
        for h = 1:length(W_hidden)
            v = g(curr_in * W_hidden{h});
            curr_in = [curr_in, v];
        end
        
        % Tahmin
        y_next = curr_in * w;
        y_sim = [y_sim; y_next];
    end
    
    y_sim_steps = y_sim(max_lag+1:end, :);
    target = Y(max_lag+1:end, :);
    
    % Boyut uyumsuzluğunu önlemek için (target - y_sim_steps)
    loss = mean((target - y_sim_steps).^2);
end