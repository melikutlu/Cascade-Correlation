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
config.model.output_trainer = 'GD_Autograd_1';
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