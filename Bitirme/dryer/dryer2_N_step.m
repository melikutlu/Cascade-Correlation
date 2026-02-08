% Kaskad Korelasyon (CCNN) - GENELLEŞTİRİLMİŞ VERSİYON
clear; clc; close all; rng(0);

% '..' demek bir üst klasöre çık demektir.
% Yani: dryer'dan çık, functions klasörüne gir.
addpath('../function');
addpath('../train_data');
%% 0. KONFİGÜRASYON - KULLANICI BURAYI DÜZENLER
% =========================================================================
config = struct();

% ==== VERİ KAYNAĞI AYARLARI ====
config.data.source = 'twotankdata';  % 'twotankdata', 'csv', 'mat', 'workspace'
config.data.filepath = '';           % CSV/MAT dosyası için yol (ör: 'data/mydata.csv')

% CSV/MAT için sütun ayarları (isteğe bağlı)
% config.data.input_columns = [1, 2, 3];    % Giriş sütun numaraları
% config.data.output_columns = [4, 5];      % Çıkış sütun numaraları

% ==== N-STEP AYARLARI ====
config.model.n_step = 5; % Kaç adım sonrasına kadar hata hesaplansın?
config.model.training_mode = 'n_step'; % 'one_step' veya 'n_step'

% Twotankdata özel ayarları
config.data.twotank.filter_cutoff = 0.066902;
config.data.twotank.warmup_samples = 20;
config.data.twotank.sampling_time = 0.2;

% Veri bölme oranları (0-1 arası, toplam 1 olmalı)
config.data.train_ratio = 0.8;    % Eğitim oranı
config.data.val_ratio = 0.2;      % Doğrulama oranı
config.data.test_ratio = 0;       % Test oranı (0 = kullanma)

% ==== NORMALİZASYON AYARLARI ====
% Seçenekler: 'ZScore', 'MinMax', 'None'
config.norm_method = 'ZScore';

% ==== REGRESÖR AYARLARI (NARX STYLE) ====
config.regressors.type = 'narx';     % 'narx' veya 'custom'
config.regressors.na = 1;            % Çıkış gecikme sayısı (y(k-1)...y(k-na))
config.regressors.nb = 1;            % Giriş gecikme sayısı (u(k-nk)...u(k-nk-nb+1))1
config.regressors.nk = 0;            % Giriş gecikmesi (delay)
config.regressors.include_bias = true;

% ==== ÖZEL GECİKMELER İÇİN (type='custom' ise) ====
% config.regressors.input_lags = [1, 2, 3];
% config.regressors.output_lags = [1, 2, 4, 6];

% ==== GİRİŞ/ÇIKIŞ BOYUTLARI ====
config.model.num_inputs = 1;      % Giriş değişkeni sayısı (u boyutu)
config.model.num_outputs = 1;     % Çıkış değişkeni sayısı (y boyutu)

% ==== MODEL HİPERPARAMETRELERİ ====
config.model.max_hidden_units = 100;
config.model.target_mse = 0.000001;
config.model.output_trainer = 'GD_Autograd_1';
config.model.eta_output = 0.0001;
config.model.mu = 0.75;
config.model.max_epochs_output = 300;
config.model.min_mse_change = 1e-9;
config.model.epsilon = 1e-8;
config.model.eta_candidate = 0.002;
config.model.max_epochs_candidate = 300;
config.model.eta_output_gd = 0.002;
config.model.batch_size = 32;

% Aktivasyon fonksiyonu
config.model.activation = 'tanh';  % 'tanh', 'sigmoid', 'relu'

% ==== GÖRSELLEŞTİRME AYARLARI ====
config.plotting.enabled = true;
config.plotting.show_simulation = true;

% =========================================================================
%% 1. VERİ YÜKLEME (Config'e göre)
fprintf('\n=== VERİ YÜKLENİYOR ===\n');

[U_train_raw, Y_train_raw, U_val_raw, Y_val_raw, U_test_raw, Y_test_raw] = ...
    loadDataByConfig(config);

% Boyut kontrolü
fprintf('Eğitim verisi: %d örnek, %d giriş, %d çıkış\n', ...
    size(U_train_raw, 1), size(U_train_raw, 2), size(Y_train_raw, 2));
if ~isempty(U_val_raw)
    fprintf('Doğrulama verisi: %d örnek\n', size(U_val_raw, 1));
end

%% 2. NORMALİZASYON
fprintf('\n=== NORMALİZASYON İŞLEMİ ===\n');
fprintf('Normalizasyon Yöntemi: %s\n', config.norm_method);

% Normalizasyon uygula 
[U_train_norm, Y_train_norm, U_val_norm, Y_val_norm, norm_stats] = ...
    normalizeData(config.norm_method, U_train_raw, Y_train_raw, U_val_raw, Y_val_raw);

%% 3. REGRESÖR MATRİSLERİNİ OLUŞTUR
fprintf('\n=== REGRESÖR MATRİSLERİ OLUŞTURULUYOR ===\n');

% Regresör oluştur (yeni fonksiyon)
[X_train_bias, T_train, reg_info] = createRegressorsDynamic(...
    U_train_norm, Y_train_norm, config);

[X_val_bias, T_val, ~] = createRegressorsDynamic(...
    U_val_norm, Y_val_norm, config);

% Model boyutlarını güncelle
num_inputs_with_bias = size(X_train_bias, 2);
fprintf('Regresör matrisi: %d örnek, %d özellik\n', ...
    size(X_train_bias, 1), num_inputs_with_bias);
fprintf('Hedef matrisi: %d örnek, %d çıkış\n', size(T_train, 1), size(T_train, 2));

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
% Config'den alınan parametreleri değişkenlere ata
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

% Başlangıç değerleri
num_hidden_units = 0;
mse_history = [];

%% AŞAMA 1: BAŞLANGIÇ AĞI EĞİTİMİ (Gizli Katmansız)
fprintf('\n=== AŞAMA 1: GİZLİ KATMANSIZ EĞİTİM ===\n');

w_o_initial = randn(num_inputs_with_bias, config.model.num_outputs) * 0.01;

% Başlangıçta giriş matrisleri, saf regresör matrisleridir
X_output_input = X_train_bias;
X_candidate_input = X_train_bias;

all_params.eta_output = eta_output;
all_params.mu = mu;
all_params.epsilon = epsilon;
all_params.eta_output_gd = eta_output_gd;
all_params.batch_size = batch_size;

[w_o_stage1_trained, E_residual, current_mse] = runOutputTraining(...
    config.model.output_trainer, X_output_input, T_train, w_o_initial, ...
    max_epochs_output, all_params,config);

T_variance_sum = sum((T_train - mean(T_train)).^2);
Y_pred_stage1 = X_output_input * w_o_stage1_trained;
fit_percentage_train_stage1 = (1 - (sum(E_residual.^2) / T_variance_sum)) * 100;

fprintf('Aşama 1 MSE: %f | Fit: %.2f%%\n', current_mse, fit_percentage_train_stage1);
mse_history(1) = current_mse;

%% AŞAMA 2: DİNAMİK BİRİM EKLEME DÖNGÜSÜ
W_hidden = {};
w_o_trained = w_o_stage1_trained;

fprintf('\n=== GİZLİ BİRİM EKLEME DÖNGÜSÜ ===\n');
fprintf('Hedef MSE: %f\n', target_mse);

while current_mse > target_mse && num_hidden_units < max_hidden_units
    num_hidden_units = num_hidden_units + 1;
    
    % A) Aday Birim Eğitimi
    [w_new_hidden, v_new_hidden] = trainCandidateUnit(...
        X_candidate_input, E_residual, max_epochs_candidate, eta_candidate, g, g_prime);
    
    W_hidden{num_hidden_units} = w_new_hidden;
    
    % B) Matrisleri Güncelle (Yeni nöron çıktısını ekle)
    X_output_input = [X_output_input, v_new_hidden];
    X_candidate_input = [X_candidate_input, v_new_hidden];
    
    % C) Çıktı Katmanı Yeniden Eğitimi
    w_o_initial_new = [w_o_trained; randn(1, config.model.num_outputs) * 0.01];
    
    [w_o_trained, E_residual, current_mse] = runOutputTraining(...
        config.model.output_trainer, X_output_input, T_train, w_o_initial_new, ...
        max_epochs_output, all_params,config);
    
    current_fit = (1 - (sum(E_residual.^2) / T_variance_sum)) * 100;
    
    % Durma Kontrolü
    mse_history(num_hidden_units + 1) = current_mse;
    
    if (mse_history(end-1) - current_mse) < 1e-5
        fprintf('*** İyileşme yetersiz. Döngü sonlandırılıyor. ***\n');
        break;
    end
    
    fprintf('Gizli Birim #%d | MSE: %f | Fit: %.2f%%\n', ...
        num_hidden_units, current_mse, current_fit);
end

fprintf('\nToplam %d gizli birim eklendi.\n', num_hidden_units);

%% 6. PERFORMANS ANALİZLERİ
fprintf('\n=== PERFORMANS ANALİZLERİ ===\n');

if config.plotting.enabled
    % --- A) EĞİTİM PERFORMANSI ---
    plotPerformanceSimple(T_train, Y_pred_stage1, X_output_input, ...
        w_o_trained, 'EĞİTİM Verisi Performansı (Normalize)', config,num_hidden_units);
    
    % --- B) DOĞRULAMA PERFORMANSI ---
    evaluateModelOptimized(X_val_bias, T_val, w_o_stage1_trained, ...
        w_o_trained, W_hidden, g, ...
        'DOĞRULAMA Verisi (One-Step Prediction - Normalize)', config,num_hidden_units);
end

%% 7. SİMÜLASYON MODU (Free Run)
if config.plotting.show_simulation
    fprintf('\n=== SİMÜLASYON (FREE RUN) MODU ===\n');
    
    % Simülasyona NORMALIZE verileri gönder
    [y_simulation_norm, ~] = simulateCCNNModel(U_val_norm, Y_val_norm, ...
        w_o_trained, W_hidden, g, config);
    
    % Geri normalizasyon
    y_simulation_real = denormalizeData(y_simulation_norm, ...
        config.norm_method, norm_stats.y);
    
    % Gerçek veri (orijinal)
    y_val_real = Y_val_raw(reg_info.max_lag+1:end, :);
    
    % Fit değerini gerçek veriler üzerinden hesapla
    if size(y_val_real, 1) == size(y_simulation_real, 1)
        fit_simulation_real = (1 - (norm(y_val_real - y_simulation_real) / ...
            norm(y_val_real - mean(y_val_real)))) * 100;
    else
        fit_simulation_real = NaN;
        fprintf('Uyarı: Simülasyon ve gerçek veri boyutları uyuşmuyor.\n');
    end
    
    % Simülasyon Grafiği (Gerçek Birimler)
    if ~isnan(fit_simulation_real)
        figure('Name', 'Simulation Result (Real World Units)', 'Color', 'w');
        plot(y_val_real, 'k', 'LineWidth', 1.5); hold on;
        plot(y_simulation_real, 'r--', 'LineWidth', 1.2);
        title(sprintf('Simülasyon (Gerçek Birimler) - Fit: %.2f%%', fit_simulation_real));
        legend('Gerçek Veri', 'Simülasyon');
        ylabel('Output (Real)');
        xlabel('Zaman Örneği');
        grid on;
    end
end




%% 8. ÖZET
fprintf('\n=== ÖZET ===\n');
fprintf('Veri Kaynağı: %s\n', config.data.source);
fprintf('Regresör Tipi: %s\n', config.regressors.type);
if strcmpi(config.regressors.type, 'narx')
    fprintf('  na=%d, nb=%d, nk=%d\n', config.regressors.na, ...
        config.regressors.nb, config.regressors.nk);
end
fprintf('Gizli Birim Sayısı: %d\n', num_hidden_units);
fprintf('Son MSE: %f\n', current_mse);
fprintf('Normalizasyon: %s\n', config.norm_method);

