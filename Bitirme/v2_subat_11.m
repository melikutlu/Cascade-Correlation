% KASKAD KORELASYON (CCNN) - FULL SERİ N-STEP VERSİYONU
clear; clc; close all; rng(0);

addpath('../function');
addpath('../train_data');

%% 0. KONFİGÜRASYON - KULLANICI BURAYI DÜZENLER
% =========================================================================
config = struct();

% ==== VERİ KAYNAĞI AYARLARI ====
config.data.source = 'twotankdata';
config.data.filepath = '';

% ==== N-STEP AYARLARI (FULL SERİ) ====
config.model.n_step = 480;          % Tüm validasyon serisi uzunluğu
config.model.training_mode = 'n_step_full';  % FULL SERİ N-STEP

% Twotankdata özel ayarları
config.data.twotank.filter_cutoff = 0.066902;
config.data.twotank.warmup_samples = 20;
config.data.twotank.sampling_time = 0.2;

% Veri bölme oranları
config.data.train_ratio = 0.8;
config.data.val_ratio = 0.2;
config.data.test_ratio = 0;

% ==== NORMALİZASYON AYARLARI ====
config.norm_method = 'ZScore';

% ==== REGRESÖR AYARLARI (NARX STYLE) ====
config.regressors.type = 'narx';
config.regressors.na = 1;
config.regressors.nb = 1;
config.regressors.nk = 0;
config.regressors.include_bias = true;

% ==== GİRİŞ/ÇIKIŞ BOYUTLARI ====
config.model.num_inputs = 1;
config.model.num_outputs = 1;

% ==== MODEL HİPERPARAMETRELERİ ====
config.model.max_hidden_units = 20;
config.model.target_mse = 0.000001;
config.model.eta_output = 0.0001;
config.model.mu = 0.75;
config.model.max_epochs_output = 500;
config.model.min_mse_change = 1e-9;
config.model.epsilon = 1e-8;
config.model.eta_candidate = 0.002;
config.model.max_epochs_candidate = 500;
config.model.eta_output_gd = 0.002;
config.model.batch_size = 32;

% Aktivasyon fonksiyonu
config.model.activation = 'tanh';

% ==== GÖRSELLEŞTİRME AYARLARI ====
config.plotting.enabled = true;
config.plotting.show_simulation = true;

% =========================================================================
%% 1. VERİ YÜKLEME
fprintf('\n=== VERİ YÜKLENİYOR ===\n');

[U_train_raw, Y_train_raw, U_val_raw, Y_val_raw, U_test_raw, Y_test_raw] = ...
    loadDataByConfig(config);

fprintf('Eğitim verisi: %d örnek\n', size(U_train_raw, 1));
fprintf('Doğrulama verisi: %d örnek\n', size(U_val_raw, 1));

%% 2. NORMALİZASYON
fprintf('\n=== NORMALİZASYON İŞLEMİ ===\n');
fprintf('Normalizasyon Yöntemi: %s\n', config.norm_method);

[U_train_norm, Y_train_norm, U_val_norm, Y_val_norm, norm_stats] = ...
    normalizeData(config.norm_method, U_train_raw, Y_train_raw, U_val_raw, Y_val_raw);

%% 3. REGRESÖR MATRİSLERİNİ OLUŞTUR
fprintf('\n=== REGRESÖR MATRİSLERİ OLUŞTURULUYOR ===\n');

[X_train_bias, T_train, reg_info] = createRegressorsDynamic(...
    U_train_norm, Y_train_norm, config);

[X_val_bias, T_val, ~] = createRegressorsDynamic(...
    U_val_norm, Y_val_norm, config);

num_inputs_with_bias = size(X_train_bias, 2);
fprintf('Regresör matrisi: %d örnek, %d özellik\n', ...
    size(X_train_bias, 1), num_inputs_with_bias);
fprintf('Hedef matrisi: %d örnek, %d çıkış\n', size(T_train, 1), size(T_train, 2));
fprintf('N-Step değeri (Full seri): %d\n', config.model.n_step);

%% 4. AKTİVASYON FONKSİYONU SEÇ
fprintf('\n=== AKTİVASYON FONKSİYONU ===\n');
fprintf('Seçilen: %s\n', config.model.activation);

switch lower(config.model.activation)
    case 'tanh'
        g = @(a) tanh(a);
        g_prime = @(v) 1 - v.^2;
    case 'sigmoid'
        g = @(a) 1./(1 + exp(-a));
        g_prime = @(v) v .* (1 - v);
    case 'relu'
        g = @(a) max(0, a);
        g_prime = @(v) double(v > 0);
    otherwise
        error('Bilinmeyen aktivasyon: %s', config.model.activation);
end

%% 5. HİPERPARAMETRELERİ AYARLA
eta_output = config.model.eta_output;
mu = config.model.mu;
max_epochs_output = config.model.max_epochs_output;
min_mse_change = config.model.min_mse_change;
epsilon = config.model.epsilon;
eta_candidate = config.model.eta_candidate;
max_epochs_candidate = config.model.max_epochs_candidate;
target_mse = config.model.target_mse;
max_hidden_units = config.model.max_hidden_units;
eta_output_gd = config.model.eta_output_gd;
batch_size = config.model.batch_size;
N = config.model.n_step;  % Full seri uzunluğu

% Başlangıç değerleri
num_hidden_units = 0;
mse_history = [];
W_hidden = {};

%% AŞAMA 1: BAŞLANGIÇ AĞI EĞİTİMİ (FULL SERİ N-STEP)
fprintf('\n=== AŞAMA 1: GİZLİ KATMANSIZ EĞİTİM (FULL SERİ N-STEP) ===\n');

w_o_initial = randn(num_inputs_with_bias, config.model.num_outputs) * 0.01;

% Parametre yapısını oluştur
all_params = struct();
all_params.eta_output = eta_output;
all_params.mu = mu;
all_params.epsilon = epsilon;
all_params.eta_output_gd = eta_output_gd;
all_params.batch_size = batch_size;
all_params.U = U_train_norm;
all_params.Y = T_train;
all_params.g = g;
all_params.g_prime = g_prime;
all_params.norm_stats = norm_stats;

% FULL SERİ N-STEP ile çıktı katmanı eğitimi
[w_o_stage1_trained, E_residual, current_mse, Y_pred_history] = ...
    trainOutputLayer_FullNStep(U_train_norm, T_train, [], w_o_initial, ...
    max_epochs_output, all_params, config, g, {});

% Fit hesapla
T_variance_sum = sum((T_train - mean(T_train)).^2);
Y_pred_stage1 = predictOneStep(X_train_bias, w_o_stage1_trained, {}, g);
fit_percentage_train_stage1 = (1 - (sum((T_train - Y_pred_stage1).^2) / T_variance_sum)) * 100;

fprintf('Aşama 1 MSE (Full N-STEP): %f | One-Step Fit: %.2f%%\n', ...
    current_mse, fit_percentage_train_stage1);
mse_history(1) = current_mse;

%% AŞAMA 2: DİNAMİK BİRİM EKLEME DÖNGÜSÜ (FULL SERİ N-STEP)
w_o_trained = w_o_stage1_trained;

fprintf('\n=== GİZLİ BİRİM EKLEME DÖNGÜSÜ (FULL SERİ N-STEP) ===\n');
fprintf('Hedef MSE: %f\n', target_mse);
fprintf('Maksimum gizli birim: %d\n', max_hidden_units);

while current_mse > target_mse && num_hidden_units < max_hidden_units
    num_hidden_units = num_hidden_units + 1;
    fprintf('\n--- Gizli Birim #%d Ekleniyor ---\n', num_hidden_units);
    
    % ADAY BİRİM EĞİTİMİ - FULL SERİ N-STEP
    [w_new_hidden, v_new_hidden] = trainCandidateUnit_FullNStep(...
        X_train_bias, T_train, E_residual, max_epochs_candidate, ...
        all_params, config, g, g_prime, w_o_trained, W_hidden, N);
    
    W_hidden{num_hidden_units} = w_new_hidden;
    
    % Matrisleri güncelle (yeni nöron çıktısını ekle)
    X_output_input = [X_train_bias, v_new_hidden];
    X_candidate_input = [X_train_bias, v_new_hidden];
    
    % ÇIKTI KATMANI YENİDEN EĞİTİMİ - FULL SERİ N-STEP
    w_o_initial_new = [w_o_trained; randn(1, config.model.num_outputs) * 0.01];
    
    [w_o_trained, E_residual, current_mse, Y_pred_history] = ...
        trainOutputLayer_FullNStep(U_train_norm, T_train, X_output_input, ...
        w_o_initial_new, max_epochs_output, all_params, config, g, W_hidden);
    
    % Fit hesapla
    Y_pred_current = predictOneStep(X_output_input, w_o_trained, {}, g);
    current_fit = (1 - (sum((T_train - Y_pred_current).^2) / T_variance_sum)) * 100;
    
    % Durma kontrolü
    mse_history(num_hidden_units + 1) = current_mse;
    
    if length(mse_history) > 1 && (mse_history(end-1) - current_mse) < 1e-5
        fprintf('*** İyileşme yetersiz. Döngü sonlandırılıyor. ***\n');
        break;
    end
    
    fprintf('Gizli Birim #%d EKLENDİ | Full N-STEP MSE: %f | One-Step Fit: %.2f%%\n', ...
        num_hidden_units, current_mse, current_fit);
end

fprintf('\n=== EĞİTİM TAMAMLANDI ===\n');
fprintf('Toplam %d gizli birim eklendi.\n', num_hidden_units);
fprintf('Son Full N-STEP MSE: %f\n', current_mse);

%% 6. DOĞRULAMA - FULL SERİ N-STEP SİMÜLASYON
fprintf('\n=== DOĞRULAMA: FULL SERİ N-STEP SİMÜLASYON ===\n');

[y_val_simulation_norm, val_mse_nstep, val_fit_nstep] = ...
    simulateAndEvaluate_FullNStep(U_val_norm, Y_val_norm, ...
    w_o_trained, W_hidden, g, config, norm_stats, reg_info);

fprintf('Doğrulama Full N-STEP MSE: %f | Fit: %.2f%%\n', ...
    val_mse_nstep, val_fit_nstep);

%% 7. TEST - FULL SERİ N-STEP SİMÜLASYON (eğer test verisi varsa)
if ~isempty(U_test_raw)
    fprintf('\n=== TEST: FULL SERİ N-STEP SİMÜLASYON ===\n');
    
    [U_test_norm, Y_test_norm, ~, ~, ~] = ...
        normalizeData(config.norm_method, U_test_raw, Y_test_raw, [], []);
    
    [y_test_simulation_norm, test_mse_nstep, test_fit_nstep] = ...
        simulateAndEvaluate_FullNStep(U_test_norm, Y_test_norm, ...
        w_o_trained, W_hidden, g, config, norm_stats, reg_info);
    
    fprintf('Test Full N-STEP MSE: %f | Fit: %.2f%%\n', ...
        test_mse_nstep, test_fit_nstep);
end

%% 8. GÖRSELLEŞTİRME
if config.plotting.enabled
    % Eğitim MSE grafiği
    figure('Name', 'Eğitim MSE Geçmişi', 'Color', 'w');
    plot(0:length(mse_history)-1, mse_history, 'b-o', 'LineWidth', 1.5, ...
        'MarkerSize', 6, 'MarkerFaceColor', 'b');
    xlabel('Eklenen Gizli Birim Sayısı');
    ylabel('MSE (Full N-STEP)');
    title(sprintf('Eğitim Performansı - Full N-STEP (N=%d)', N));
    grid on;
    set(gca, 'YScale', 'log');
    
    % Doğrulama simülasyon grafiği
    figure('Name', 'Doğrulama - Full Seri Simülasyon', 'Color', 'w');
    
    % Gerçek veri (orijinal birimler)
    y_val_real = Y_val_raw(reg_info.max_lag+1:end, :);
    
    % Simülasyon çıktısı (orijinal birimler)
    y_val_simulation_real = y_val_simulation_norm * norm_stats.y.std + norm_stats.y.mean;
    
    % Boyut kontrolü
    min_len = min(length(y_val_real), length(y_val_simulation_real));
    y_val_real = y_val_real(1:min_len);
    y_val_simulation_real = y_val_simulation_real(1:min_len);
    
    plot(y_val_real, 'k-', 'LineWidth', 1.5); hold on;
    plot(y_val_simulation_real, 'r--', 'LineWidth', 1.2);
    title(sprintf('Doğrulama - Full Seri Simülasyon (N=%d) - Fit: %.2f%%', ...
        N, val_fit_nstep));
    legend('Gerçek Veri', 'Model Simülasyonu', 'Location', 'best');
    ylabel('Çıkış (Gerçek Birim)');
    xlabel('Zaman Adımı');
    grid on;
    
    % Hata grafiği
    figure('Name', 'Simülasyon Hatası', 'Color', 'w');
    error = y_val_real - y_val_simulation_real;
    plot(error, 'b-', 'LineWidth', 1);
    title(sprintf('Simülasyon Hatası (RMSE: %.4f)', sqrt(mean(error.^2))));
    xlabel('Zaman Adımı');
    ylabel('Hata');
    grid on;
    yline(0, 'k--');
end

%% 9. ÖZET
fprintf('\n========== ÖZET ==========\n');
fprintf('Veri Kaynağı: %s\n', config.data.source);
fprintf('N-Step Değeri (Full Seri): %d\n', N);
fprintf('Gizli Birim Sayısı: %d\n', num_hidden_units);
fprintf('Eğitim Full N-STEP MSE: %f\n', current_mse);
fprintf('Doğrulama Full N-STEP MSE: %f\n', val_mse_nstep);
fprintf('Doğrulama Full N-STEP Fit: %.2f%%\n', val_fit_nstep);
fprintf('Normalizasyon: %s\n', config.norm_method);
fprintf('===========================\n');

%% ========== FONKSİYONLAR ==========

function [Y_pred] = predictOneStep(X, w_o, W_hidden, g)
    % One-step prediction
    Y_pred = X * w_o(1:size(X,2), :);
    
    for h = 1:length(W_hidden)
        v_h = g(X * W_hidden{h});
        Y_pred = Y_pred + v_h * w_o(size(X,2)+h, :);
    end
end

function [y_pred] = predictOneStepSample(x, w_o, W_hidden, g)
    % Tek örnek için one-step prediction
    y_pred = x' * w_o(1:length(x), :);
    
    for h = 1:length(W_hidden)
        v_h = g(x' * W_hidden{h});
        y_pred = y_pred + v_h * w_o(length(x)+h, :);
    end
end

function [w_o, E_residual, final_mse, Y_pred_history] = trainOutputLayer_FullNStep(...
    U, Y, X_output, w_o_initial, max_epochs, params, config, g, W_hidden)
    
    % FULL SERİ N-STEP ile çıktı katmanı eğitimi
    N = config.model.n_step;
    eta = params.eta_output;
    Y = Y(:);  % Vektör formatına dönüştür
    
    w_o = w_o_initial;
    num_features = size(X_output, 2);
    best_mse = inf;
    best_w_o = w_o;
    Y_pred_history = [];
    
    for epoch = 1:max_epochs
        total_nstep_mse = 0;
        num_predictions = 0;
        
        % Her başlangıç noktası için N-step simülasyon
        for t = 1:length(Y)-N
            % N-step simülasyon yap
            y_sim = simulateNSteps(U, Y, w_o, W_hidden, g, t, N, num_features);
            
            % N adım sonraki gerçek değer
            y_target = Y(t + N);
            
            % Hata
            error = y_target - y_sim(end);
            total_nstep_mse = total_nstep_mse + error^2;
            num_predictions = num_predictions + 1;
            
            % Gradient hesapla (BPTT - basitleştirilmiş)
            if num_features > 0
                X_t = X_output(t, :)';
                grad = X_t * error * 0.01;  % Basitleştirilmiş gradient
                w_o(1:num_features, :) = w_o(1:num_features, :) + eta * grad;
            end
        end
        
        current_mse = total_nstep_mse / num_predictions;
        
        % En iyi modeli kaydet
        if current_mse < best_mse
            best_mse = current_mse;
            best_w_o = w_o;
        end
        
        if mod(epoch, 50) == 0 || epoch == 1
            fprintf('  Epoch %d | Full N-STEP MSE: %f\n', epoch, current_mse);
        end
        
        % Erken durdurma
        if epoch > 50 && abs(current_mse - best_mse) < 1e-6
            break;
        end
    end
    
    w_o = best_w_o;
    final_mse = best_mse;
    
    % Artık hata (residual) - one-step prediction ile
    if ~isempty(X_output)
        Y_pred_one_step = predictOneStep(X_output, w_o, W_hidden, g);
        E_residual = Y - Y_pred_one_step;
    else
        E_residual = Y;
    end
end

function [y_sim] = simulateNSteps(U, Y, w_o, W_hidden, g, start_idx, N, num_features)
    % N adımlık simülasyon yap
    y_sim = zeros(N, 1);
    
    % İlk adım: gerçek geçmiş değeri kullan
    if start_idx > 1
        x = [1; Y(start_idx); U(start_idx + 1)];
    
    end
    
    y_sim(1) = predictOneStepSample(x, w_o, W_hidden, g);
    
    % Sonraki adımlar: modelin kendi tahminlerini kullan
    for k = 2:N
        x = [1; y_sim(k-1); U(start_idx + k)];
        y_sim(k) = predictOneStepSample(x, w_o, W_hidden, g);
    end
end

function [w_new_hidden, v_new_hidden] = trainCandidateUnit_FullNStep(...
    X_input, T_train, E_residual, max_epochs, params, config, g, g_prime, ...
    w_o_current, W_hidden_current, N)
    
    % FULL SERİ N-STEP ile candidate birim eğitimi
    U = params.U;
    Y = T_train(:);
    eta = config.model.eta_candidate;
    num_features = size(X_input, 2);
    
    % Candidate ağırlıklarını başlangıç değerleri
    w_candidate = randn(num_features, 1) * 0.01;
    best_correlation = -inf;
    best_w = w_candidate;
    
    for epoch = 1:max_epochs
        total_correlation = 0;
        grad_sum = zeros(size(w_candidate));
        num_samples = 0;
        
        for t = 1:length(Y)-N
            % N-step simülasyon yap
            y_sim = simulateNSteps(U, Y, w_o_current, W_hidden_current, g, t, N, num_features);
            
            % Candidate çıktısı
            a_candidate = X_input(t, :) * w_candidate;
            v_candidate = g(a_candidate);
            
            % N-step sonrası hata
            error = Y(t + N) - y_sim(end);
            
            % Korelasyon: |cov(v, error)|
            if t == 1
                mean_v = v_candidate;
                mean_error = error;
            else
                mean_v = mean_v + (v_candidate - mean_v) / t;
                mean_error = mean_error + (error - mean_error) / t;
            end
            
            cov = (v_candidate - mean_v) * (error - mean_error);
            total_correlation = total_correlation + abs(cov);
            
            % Gradient (korelasyon maksimizasyonu)
            dv_dw = X_input(t, :)' * g_prime(a_candidate);
            grad = dv_dw * error * sign(cov);
            grad_sum = grad_sum + grad;
            num_samples = num_samples + 1;
        end
        
        % Ağırlık güncelleme
        w_candidate = w_candidate + eta * (grad_sum / num_samples);
        
        % Korelasyon değeri
        current_correlation = total_correlation / num_samples;
        
        if current_correlation > best_correlation
            best_correlation = current_correlation;
            best_w = w_candidate;
        end
        
        if mod(epoch, 50) == 0
            fprintf('  Aday Epoch %d | Korelasyon: %f\n', ...
                epoch, current_correlation);
        end
    end
    
    w_new_hidden = best_w;
    v_new_hidden = g(X_input * w_new_hidden);
end

function [y_simulation, mse_nstep, fit_nstep] = simulateAndEvaluate_FullNStep(...
    U_val, Y_val, w_o, W_hidden, g, config, norm_stats, reg_info)
    
    % FULL SERİ N-STEP simülasyon ve değerlendirme
    N = config.model.n_step;
    Y_val = Y_val(:);
    U_val = U_val(:);
    
    % Başlangıç indisleri
    start_idx = reg_info.max_lag + 1;
    Y_val_initial = Y_val(start_idx - reg_info.max_lag:end);
    U_val_aligned = U_val(start_idx:end);
    
    num_samples = length(Y_val_initial) - 1;
    y_simulation = zeros(num_samples, 1);
    
    % Warm-up: ilk örnek için gerçek değeri kullan
    if num_samples > 0
        x = [1; Y_val_initial(1); U_val_aligned(1)];
        y_simulation(1) = predictOneStepSample(x, w_o, W_hidden, g);
    end
    
    % Simülasyon
    for t = 2:num_samples
        x = [1; y_simulation(t-1); U_val_aligned(t)];
        y_simulation(t) = predictOneStepSample(x, w_o, W_hidden, g);
    end
    
    % Değerlendirme - N-STEP MSE
    mse_nstep = 0;
    num_pred = 0;
    
    for t = 1:length(y_simulation)-N
        y_target = Y_val_initial(t + N);
        mse_nstep = mse_nstep + (y_target - y_simulation(t + N - 1))^2;
        num_pred = num_pred + 1;
    end
    
    if num_pred > 0
        mse_nstep = mse_nstep / num_pred;
    end
    
    % Fit hesapla
    y_target_all = Y_val_initial(1:length(y_simulation));
    if length(y_target_all) == length(y_simulation)
        fit_nstep = (1 - (norm(y_target_all - y_simulation) / ...
            norm(y_target_all - mean(y_target_all)))) * 100;
    else
        min_len = min(length(y_target_all), length(y_simulation));
        fit_nstep = (1 - (norm(y_target_all(1:min_len) - y_simulation(1:min_len)) / ...
            norm(y_target_all(1:min_len) - mean(y_target_all(1:min_len))))) * 100;
    end
end