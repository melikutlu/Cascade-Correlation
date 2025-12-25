% Cascade Correlation (CCNN) - n-Step Prediction Odaklı Master Versiyon
clear; clc; close all; rng(0);

%% 0. KONFİGÜRASYON
config = struct();
% Veri Kaynağı
config.data.source = 'twotankdata'; 
config.data.train_ratio = 0.5;
config.data.val_ratio = 0.5;
config.data.twotank.sampling_time = 0.2;

% N-STEP KRİTİK AYARI
% Model her n adımda bir gerçek veriyi görecek, aradaki adımlarda kendi tahminini kullanacak.
config.n_predict = 10; 

% Regresör Ayarları (NARX Yapısı)
config.regressors.type = 'narx';
config.regressors.na = 2; % Çıkış gecikmesi
config.regressors.nb = 2; % Giriş gecikmesi
config.regressors.nk = 1; % Gecikme (delay)
config.regressors.include_bias = true;

% Model Hiperparametreleri
config.model.max_hidden_units = 15;
config.model.target_mse = 1e-6;
config.model.output_trainer = 'GD_Auto_nstep';
config.model.activation = 'tanh';
config.norm_method = 'ZScore';

%% 1. VERİ HAZIRLAMA
fprintf('\n=== VERİ HAZIRLANIYOR ===\n');
[U_raw, Y_raw, U_v_raw, Y_v_raw] = loadDataByConfig(config);

% Normalizasyon
[U_tr, Y_tr, U_val, Y_val, norm_stats] = normalizeData(config.norm_method, U_raw, Y_raw, U_v_raw, Y_v_raw);

% Eğitim için temel regresör matrisi
[X_tr_base, T_tr_base, reg_info] = createRegressorsDynamic(U_tr, Y_tr, config);

% Aktivasyon Tanımı
g = @(a) tanh(a);

%% 2. CCNN EĞİTİMİ (One-Step Optimization)
% Not: Gradyan hesapları için eğitim "one-step" üzerinden yapılır.
[w_o_final, W_hidden_list, num_h] = trainCCNNCore(X_tr_base, T_tr_base, config, g);

%% 3. N-STEP PERFORMANS DEĞERLENDİRME
fprintf('\n=== N-STEP TAHMİN ANALİZİ (n=%d) ===\n', config.n_predict);

% Eğitim ve Validasyon setleri için n-step tahmini çalıştır
[y_tr_n, fit_tr] = runNStepPrediction(U_tr, Y_tr, w_o_final, W_hidden_list, g, config);
[y_val_n, fit_val] = runNStepPrediction(U_val, Y_val, w_o_final, W_hidden_list, g, config);

%% 4. SONUÇLAR VE GÖRSELLEŞTİRME
% Gerçek birimlere dönüş
y_val_n_real = denormalizeData(y_val_n, config.norm_method, norm_stats.y);
y_val_true_real = Y_v_raw(reg_info.max_lag+1:end, :);

fprintf('Eğitim n-Step Fit: %.2f%%\n', fit_tr);
fprintf('Validasyon n-Step Fit: %.2f%%\n', fit_val);

% Grafik: Tahmin vs Gerçek
figure('Name', sprintf('CCNN %d-Step Prediction Analysis', config.n_predict), 'Color', 'w');
subplot(2,1,1);
plot(y_val_true_real, 'k', 'LineWidth', 1.5, 'DisplayName', 'Gerçek Veri'); hold on;
plot(y_val_n_real, 'r--', 'LineWidth', 1.2, 'DisplayName', sprintf('%d-Step Tahmin', config.n_predict));
title(sprintf('Validasyon Verisi %d-Adımlı Tahmin (Fit: %.2f%%)', config.n_predict, fit_val));
legend('show'); grid on; ylabel('Çıktı');

% Grafik: Hata Dağılımı
subplot(2,1,2);
plot(y_val_true_real - y_val_n_real, 'b');
title('Tahmin Hatası (Residuals)');
xlabel('Zaman Adımı'); ylabel('Hata'); grid on;

%% === TEMEL N-STEP FONKSİYONU ===

function [y_pred, fit_score] = runNStepPrediction(U, Y, w_o, W_h, g_func, config)
    % Bu fonksiyon n-adımlı özyinelemeli tahmini gerçekleştirir.
    N = size(U, 1);
    n_step = config.n_predict;
    max_lag = max([config.regressors.na, config.regressors.nb + config.regressors.nk]);
    
    y_pred = zeros(N, size(Y, 2));
    y_pred(1:max_lag, :) = Y(1:max_lag, :); % Başlangıç için gerçek veri
    
    for k = (max_lag + 1):N
        % Reset mekanizması: Her n adımda bir gerçek geçmişi kullan
        is_reset_point = (mod(k - max_lag - 1, n_step) == 0);
        
        % 1. Regresör oluştur (u biliniyor, y tahmini veya gerçek)
        curr_x = [];
        if config.regressors.include_bias, curr_x = [curr_x, 1]; end
        
        % Giriş u gecikmeleri
        for lag = config.regressors.nk : (config.regressors.nk + config.regressors.nb - 1)
            curr_x = [curr_x, U(k-lag, :)];
        end
        
        % Çıkış y gecikmeleri (Kritik n-step mantığı)
        for lag = 1:config.regressors.na
            if is_reset_point
                curr_x = [curr_x, Y(k-lag, :)]; % Yeni pencere başı: Gerçek veri
            else
                curr_x = [curr_x, y_pred(k-lag, :)]; % Pencere içi: Kendi tahmini
            end
        end
        
        % 2. Gizli Birimlerden Geçir (CCNN Yapısı)
        temp_feat = curr_x;
        for i = 1:length(W_h)
            v = g_func(temp_feat * W_h{i});
            temp_feat = [temp_feat, v];
        end
        
        % 3. Çıkış Hesapla
        y_pred(k, :) = temp_feat * w_o;
    end
    
    % Kırpma ve Fit Hesaplama
    y_pred = y_pred(max_lag+1:end, :);
    y_true = Y(max_lag+1:end, :);
    fit_score = (1 - (norm(y_true - y_pred)/norm(y_true - mean(y_true)))) * 100;
end

function [w_o, W_hidden, h_idx] = trainCCNNCore(X, T, config, g)
    % CCNN çekirdek eğitim algoritması
    fprintf('CCNN Eğitimi: One-step optimizasyon başlıyor...\n');
    
    % Başlangıç: Gizli birimsiz ağırlıklar
    w_o = randn(size(X,2), size(T,2)) * 0.01;
    [w_o, E_res, cur_mse] = runOutputTraining(config.model.output_trainer, X, T, w_o, config.model.max_epochs_output, config.model);
    
    W_hidden = {}; h_idx = 0; prev_mse = cur_mse;
    
    while cur_mse > config.model.target_mse && h_idx < config.model.max_hidden_units
        h_idx = h_idx + 1;
        
        % Aday Birim Eğitimi
        [w_new, v_new] = trainCandidateUnit(X, E_res, config.model.max_epochs_candidate, config.model.eta_candidate, g, @(v) 1-v.^2);
        
        W_hidden{h_idx} = w_new;
        X = [X, v_new]; % Ağı genişlet
        
        % Çıkış Katmanı Güncelleme
        w_init = [w_o; randn(1, size(T,2)) * 0.01];
        [w_o, E_res, cur_mse] = runOutputTraining(config.model.output_trainer, X, T, w_init, config.model.max_epochs_output, config.model);
        
        fprintf('Birim %d eklendi | MSE: %.6f\n', h_idx, cur_mse);
        if (prev_mse - cur_mse) < 1e-8, break; end
        prev_mse = cur_mse;
    end
end



%% YARDIMCI FONKSİYONLAR
% =========================================================================
function [U_train, Y_train, U_val, Y_val, U_test, Y_test] = loadDataByConfig(config)
    % Farklı veri kaynaklarını yükleyen ana fonksiyon
    
    U_test = [];
    Y_test = [];
    
    switch lower(config.data.source)
        case 'twotankdata'
            [U_train, Y_train, U_val, Y_val] = loadTwotankData(config);
            
        case 'csv'
            [U_train, Y_train, U_val, Y_val] = loadCSVData(config);
            
        case 'mat'
            [U_train, Y_train, U_val, Y_val] = loadMATData(config);
            
        case 'workspace'
            [U_train, Y_train, U_val, Y_val] = loadFromWorkspace(config);
            
        otherwise
            error('Desteklenmeyen veri kaynağı: %s', config.data.source);
    end
end

function [U_train, Y_train, U_val, Y_val] = loadTwotankData(config)
    % Twotankdata için özel yükleyici
    
    load dryer2;
    z_full = iddata(y2, u2, config.data.twotank.sampling_time);
    
    % Veriyi böl
    N_total = length(z_full.y);
    train_end = floor(N_total * config.data.train_ratio);
    val_end = train_end + floor(N_total * config.data.val_ratio);
    
    % Eğitim verisi
    if config.data.train_ratio > 0
        z1 = z_full(1:train_end);
        z1f = detrend(z1);
        U_train = z1f.u;
        Y_train = z1f.y;
    else
        U_train = [];
        Y_train = [];
    end
    
    % Doğrulama verisi
    if config.data.val_ratio > 0
        z2 = z_full(train_end+1:val_end);
        z2f = detrend(z2);
        U_val = z2f.u;
        Y_val = z2f.y;
    else
        U_val = [];
        Y_val = [];
    end
end

function [U_train, Y_train, U_val, Y_val] = loadCSVData(config)
    % CSV dosyasından veri yükleme
    
    if isempty(config.data.filepath)
        error('CSV dosya yolu belirtilmeli: config.data.filepath');
    end
    
    data = readtable(config.data.filepath);
    
    % Sütun belirleme
    if isfield(config.data, 'input_columns') && ~isempty(config.data.input_columns)
        input_cols = config.data.input_columns;
    else
        % Varsayılan: ilk num_inputs sütun
        input_cols = 1:config.model.num_inputs;
    end
    
    if isfield(config.data, 'output_columns') && ~isempty(config.data.output_columns)
        output_cols = config.data.output_columns;
    else
        % Varsayılan: sonraki num_outputs sütun
        start_idx = max(input_cols) + 1;
        output_cols = start_idx:(start_idx + config.model.num_outputs - 1);
    end
    
    U_all = table2array(data(:, input_cols));
    Y_all = table2array(data(:, output_cols));
    
    % Veriyi böl
    N = size(U_all, 1);
    train_end = floor(N * config.data.train_ratio);
    val_end = train_end + floor(N * config.data.val_ratio);
    
    U_train = U_all(1:train_end, :);
    Y_train = Y_all(1:train_end, :);
    U_val = U_all(train_end+1:val_end, :);
    Y_val = Y_all(train_end+1:val_end, :);
end

function [U_train, Y_train, U_val, Y_val] = loadMATData(config)
    % .mat dosyasından veri yükleme
    
    if isempty(config.data.filepath)
        error('MAT dosya yolu belirtilmeli: config.data.filepath');
    end
    
    data = load(config.data.filepath);
    
    % Değişken isimlerini belirle
    if isfield(config.data, 'input_var')
        U_all = data.(config.data.input_var);
    else
        % Varsayılan değişken isimleri
        if isfield(data, 'U')
            U_all = data.U;
        elseif isfield(data, 'u')
            U_all = data.u;
        elseif isfield(data, 'input')
            U_all = data.input;
        else
            error('Giriş verisi değişkeni bulunamadı');
        end
    end
    
    if isfield(config.data, 'output_var')
        Y_all = data.(config.data.output_var);
    else
        % Varsayılan değişken isimleri
        if isfield(data, 'Y')
            Y_all = data.Y;
        elseif isfield(data, 'y')
            Y_all = data.y;
        elseif isfield(data, 'output')
            Y_all = data.output;
        else
            error('Çıkış verisi değişkeni bulunamadı');
        end
    end
    
    % Veriyi böl
    N = size(U_all, 1);
    train_end = floor(N * config.data.train_ratio);
    val_end = train_end + floor(N * config.data.val_ratio);
    
    U_train = U_all(1:train_end, :);
    Y_train = Y_all(1:train_end, :);
    U_val = U_all(train_end+1:val_end, :);
    Y_val = Y_all(train_end+1:val_end, :);
end

function [U_train, Y_train, U_val, Y_val] = loadFromWorkspace(config)
    % Workspace'ten değişkenleri yükleme
    
    fprintf('Workspace''ten yükleme:\n');
    
    % Giriş değişkenleri
    if isfield(config.data, 'input_var')
        U_all = evalin('base', config.data.input_var);
    else
        error('Workspace için config.data.input_var belirtilmeli');
    end
    
    % Çıkış değişkenleri
    if isfield(config.data, 'output_var')
        Y_all = evalin('base', config.data.output_var);
    else
        error('Workspace için config.data.output_var belirtilmeli');
    end
    
    % Veriyi böl
    N = size(U_all, 1);
    train_end = floor(N * config.data.train_ratio);
    val_end = train_end + floor(N * config.data.val_ratio);
    
    U_train = U_all(1:train_end, :);
    Y_train = Y_all(1:train_end, :);
    U_val = U_all(train_end+1:val_end, :);
    Y_val = Y_all(train_end+1:val_end, :);
end

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

function evaluateModelOptimized(X_val, T_val, w_stage1, w_final, W_hidden, g, plot_title, config,num_hidden_units)
    % Model değerlendirme fonksiyonu
    
    num_hidden = length(W_hidden);
    
    % 1. Aşama 1 (Gizli Katmansız) Tahmin
    Y_stage1 = X_val * w_stage1;
    fit_stage1 = (1 - (sum((T_val - Y_stage1).^2) / sum((T_val - mean(T_val)).^2))) * 100;
    
    % 2. Aşama 2 (Tam Model) Tahmin
    X_curr = X_val;
    X_cand = X_val;
    for k = 1:num_hidden
        V_h = g(X_cand * W_hidden{k});
        X_curr = [X_curr, V_h];
        X_cand = [X_cand, V_h];
    end
    
    Y_final = X_curr * w_final;
    fit_final = (1 - (sum((T_val - Y_final).^2) / sum((T_val - mean(T_val)).^2))) * 100;
    
    fprintf('%s -> Başlangıç Fit: %.2f%% | Final Fit: %.2f%%\n', ...
        plot_title, fit_stage1, fit_final);
    
    if config.plotting.enabled
        figure('Name', plot_title, 'Color', 'w');
        plot(T_val, 'k', 'LineWidth', 1.5); hold on;
        plot(Y_stage1, 'r--', 'DisplayName', sprintf('Gizli Katman Yok - Fit: %.2f%%', fit_stage1));
        plot(Y_final, 'b-', 'DisplayName', sprintf('%d Gizli Katman - Fit: %.2f%%', num_hidden_units, fit_final));
        legend('show', 'Location', 'best');
        title(sprintf('%s (Fit: %.2f%%)', plot_title, fit_final));
        xlabel('Zaman Örneği');
        ylabel('Çıkış (Normalize)');
        grid on;
    end
end

function plotPerformanceSimple(T, Y_stage1, X_final_input, w_final, title_txt, config,num_hidden_units)
    % Performans grafiği
    
    Y_final = X_final_input * w_final;
    fit_final = (1 - (sum((T - Y_final).^2) / sum((T - mean(T)).^2))) * 100;
    fit_stage1 = (1 - (sum((T - Y_stage1).^2) / sum((T - mean(T)).^2))) * 100;
    
    % Gizli katman sayısı hesapla
    num_hidden_calc = size(X_final_input, 2) - size(Y_stage1, 2);
    
    if config.plotting.enabled
        figure('Name', title_txt, 'Color', 'w');
        plot(T, 'k', 'LineWidth', 1.5); hold on;
        plot(Y_stage1, 'r--', 'LineWidth', 1.2, ...
            'DisplayName', sprintf('Gizli Katman Yok - Fit: %.2f%%', fit_stage1));
        plot(Y_final, 'b-', 'LineWidth', 1.2, ...
            'DisplayName', sprintf('%d Gizli Katman - Fit: %.2f%%', num_hidden_units, fit_final));
        legend('show', 'Location', 'best');
        title(sprintf('%s (Fit: %.2f%%)', title_txt, fit_final));
        xlabel('Zaman Örneği');
        ylabel('Çıkış (Normalize)');
        grid on;
    end
end

