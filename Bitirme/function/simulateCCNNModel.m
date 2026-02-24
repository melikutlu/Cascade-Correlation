function [y_sim, fit_sim] = simulateCCNNModel(U_val, Y_val, w_o, W_hidden, g_func, config)
    % Free-run simülasyonu
    N = size(U_val, 1);
    num_inputs = config.model.num_inputs;
    num_outputs = config.model.num_outputs;
    
    % Gecikmeleri belirle
    if strcmpi(config.regressors.type, 'narx')
        na = config.regressors.na;
        nb = config.regressors.nb;
        nk = config.regressors.nk;
        input_lags = nk:(nk+nb-1);
        output_lags = 1:na;
    else
        input_lags = config.regressors.input_lags;
        output_lags = config.regressors.output_lags;
    end
    
    max_lag = max([input_lags, output_lags]);
    num_hidden = length(W_hidden);
    
    % Başlangıç değerleri (ilk max_lag değeri gerçek veriden)
    y_sim = zeros(N, num_outputs);
    y_sim(1:max_lag, :) = Y_val(1:max_lag, :);
    
    % Simülasyon döngüsü
    for k = (max_lag+1):N
        % Giriş vektörünü oluştur
        curr_in = [];
        
        % Bias
        if config.regressors.include_bias
            curr_in = [curr_in, 1];
        end
        
        % Giriş gecikmeleri
        for input_idx = 1:num_inputs
            for lag = input_lags
                if k-lag > 0
                    curr_in = [curr_in, U_val(k-lag, input_idx)];
                else
                    curr_in = [curr_in, 0]; % Başlangıçta yoksa 0
                end
            end
        end
        
        % Çıkış gecikmeleri (simüle edilmiş değerler)
        for output_idx = 1:num_outputs
            for lag = output_lags
                if k-lag > 0
                    curr_in = [curr_in, y_sim(k-lag, output_idx)];
                else
                    curr_in = [curr_in, Y_val(k-lag, output_idx)]; % Başlangıçta gerçek
                end
            end
        end
        
        % Gizli katman çıktıları
        for h = 1:num_hidden
            v = g_func(curr_in * W_hidden{h});
            curr_in = [curr_in, v];
        end
        
        % Çıkış hesapla
        y_sim(k, :) = curr_in * w_o;
    end
    
    % Fit hesapla (sadece simülasyon bölgesi)
    y_sim = y_sim(max_lag+1:end, :);
    y_val = Y_val(max_lag+1:end, :);
    
    if size(y_val, 1) == size(y_sim, 1)
        fit_sim = (1 - (norm(y_val - y_sim) / norm(y_val - mean(y_val)))) * 100;
    else
        fit_sim = NaN;
    end
end
