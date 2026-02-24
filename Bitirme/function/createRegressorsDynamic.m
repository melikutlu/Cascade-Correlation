
function [X_bias, T, reg_info] = createRegressorsDynamic(U, Y, config)
    % Dinamik regresör matrisi oluşturur (NARX veya custom)
    
    N = size(U, 1);
    num_inputs = config.model.num_inputs;
    num_outputs = config.model.num_outputs;
    
    % 1. Gecikmeleri belirle
    if strcmpi(config.regressors.type, 'narx')
        % NARX tarzı gecikmeler
        na = config.regressors.na;
        nb = config.regressors.nb;
        nk = config.regressors.nk;
        
        input_lags = nk:(nk+nb-1);
        output_lags = 1:na;
        
    elseif strcmpi(config.regressors.type, 'custom')
        % Özel gecikmeler
        input_lags = config.regressors.input_lags;
        output_lags = config.regressors.output_lags;
        
    else
        error('Bilinmeyen regresör tipi: %s', config.regressors.type);
    end
    
    % 2. En büyük gecikme
    max_lag = max([input_lags, output_lags]);
    
    % 3. Regresör matrisini oluştur
    start_idx = max_lag + 1;
    X = [];
    
    % 3.1 Giriş gecikmeleri
    for input_idx = 1:num_inputs
        for lag = input_lags
            col_data = U(start_idx-lag:end-lag, input_idx);
            X = [X, col_data];
        end
    end
    
    % 3.2 Çıkış gecikmeleri
    for output_idx = 1:num_outputs
        for lag = output_lags
            col_data = Y(start_idx-lag:end-lag, output_idx);
            X = [X, col_data];
        end
    end
    
    % 4. Bias ekle
    if config.regressors.include_bias
        X_bias = [ones(size(X,1), 1), X];
    else
        X_bias = X;
    end
    
    % 5. Hedef matrisi (ileri bir adım)
    T = Y(start_idx:end, :);
    
    % 6. Bilgileri kaydet
    reg_info.input_lags = input_lags;
    reg_info.output_lags = output_lags;
    reg_info.max_lag = max_lag;
    reg_info.num_features = size(X_bias, 2);
    reg_info.num_samples = size(X_bias, 1);
    
    % Debug bilgisi
    fprintf('  Giriş gecikmeleri: %s\n', mat2str(input_lags));
    fprintf('  Çıkış gecikmeleri: %s\n', mat2str(output_lags));
    fprintf('  Maksimum gecikme: %d\n', max_lag);
end
