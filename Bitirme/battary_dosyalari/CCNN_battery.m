% Kaskad Korelasyon (CCNN) - GENELLEŞTİRİLMİŞ VERSİYON
clear; clc; close all; rng(0);

%% 0. KONFİGÜRASYON - BATARYA VERİSİ İÇİN
% =========================================================================
config = struct();

% ==== VERİ KAYNAĞI AYARLARI ====
config.data.source = 'battery';  % 'twotankdata' yerine 'battery'
config.data.train_file = 'FTP75.mat';    % Eğitim verisi
config.data.val_file = 'LA92.mat';       % Doğrulama verisi

% ==== NORMALİZASYON AYARLARI ====
config.norm_method = 'ZScore';

% ==== REGRESÖR AYARLARI (NARX STYLE) ====
config.regressors.type = 'narx';
config.regressors.na = 5;    % 5 çıkış gecikmesi (temperature)
config.regressors.nb = 3;    % 3 giriş gecikmesi (her giriş için)
config.regressors.nk = 1;    % 1 gecikme (delay)
config.regressors.include_bias = true;

% ==== GİRİŞ/ÇIKIŞ BOYUTLARI ====
config.model.num_inputs = 2;      % 2 giriş: current ve SOC
config.model.num_outputs = 1;     % 1 çıkış: temperature

% ==== MODEL HİPERPARAMETRELERİ ====
config.model.max_hidden_units = 50;
config.model.target_mse = 0.001;
config.model.output_trainer = 'GD_Autograd';
config.model.eta_output = 0.001;
config.model.mu = 0.75;
config.model.max_epochs_output = 300;
config.model.min_mse_change = 1e-9;
config.model.epsilon = 1e-8;
config.model.eta_candidate = 0.003;
config.model.max_epochs_candidate = 100;
config.model.eta_output_gd = 0.002;
config.model.batch_size = 32;

% Aktivasyon fonksiyonu
config.model.activation = 'tanh';

% ==== GÖRSELLEŞTİRME AYARLARI ====
config.plotting.enabled = true;
config.plotting.show_simulation = true;
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

% Normalizasyon uygula (arkadaşının fonksiyonu)
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
    max_epochs_output, all_params);

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
        max_epochs_output, all_params);
    
    current_fit = (1 - (sum(E_residual.^2) / T_variance_sum)) * 100;
    
    % Durma Kontrolü
    mse_history(num_hidden_units + 1) = current_mse;
    
    if (mse_history(end-1) - current_mse) < 1e-2 && num_hidden_units > 1
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
        w_o_trained, 'EĞİTİM Verisi Performansı (Normalize)', config);
    
    % --- B) DOĞRULAMA PERFORMANSI ---
    evaluateModelOptimized(X_val_bias, T_val, w_o_stage1_trained, ...
        w_o_trained, W_hidden, g, ...
        'DOĞRULAMA Verisi (One-Step Prediction - Normalize)', config);
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

%% YARDIMCI FONKSİYONLAR
% =========================================================================
function [U_train, Y_train, U_val, Y_val, U_test, Y_test] = loadDataByConfig(config)
    % Farklı veri kaynaklarını yükleyen ana fonksiyon
    
    U_test = [];
    Y_test = [];
    
    switch lower(config.data.source)
        case 'twotankdata'
            [U_train, Y_train, U_val, Y_val] = loadTwotankData(config);
            
        case 'battery'
            [U_train, Y_train, U_val, Y_val] = loadBatteryData(config);
            
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

function [U_train, Y_train, U_val, Y_val] = loadBatteryData(config)
    % Batarya verisi için özel yükleyici
    % MATLAB'ın resmi batarya veri seti formatına uygun
    
    fprintf('   Batarya verisi yükleniyor...\n');
    
    % 1. Eğitim verisini yükle (FTP75)
    if isfield(config.data, 'train_file') && ~isempty(config.data.train_file)
        train_data = load(config.data.train_file);
        fprintf('   Eğitim dosyası: %s\n', config.data.train_file);
    else
        train_data = load('FTP75.mat');
        fprintf('   Varsayılan eğitim: FTP75.mat\n');
    end
    
    % 2. Doğrulama verisini yükle (LA92)
    if isfield(config.data, 'val_file') && ~isempty(config.data.val_file)
        val_data = load(config.data.val_file);
        fprintf('   Doğrulama dosyası: %s\n', config.data.val_file);
    else
        val_data = load('LA92.mat');
        fprintf('   Varsayılan doğrulama: LA92.mat\n');
    end
    
    % 3. MATLAB'ın resmi formatını kontrol et (iddata formatı)
    % Eğer iddata formatında ise
    if isfield(train_data, 'eData') || isfield(val_data, 'vData')
        fprintf('   iddata formatı kullanılıyor\n');
        
        if isfield(train_data, 'eData')
            eData = train_data.eData;
        else
            % Alternatif: direkt değişkenlerden oluştur
            eData = iddata(train_data.y_temp, [train_data.u_curr, train_data.u_soc], 0.1);
        end
        
        if isfield(val_data, 'vData')
            vData = val_data.vData;
        else
            vData = iddata(val_data.y_temp, [val_data.u_curr, val_data.u_soc], 0.1);
        end
        
        U_train = eData.u;
        Y_train = eData.y;
        U_val = vData.u;
        Y_val = vData.y;
        
    else
        % 4. Standart değişken isimlerini kontrol et
        fprintf('   Standart değişken isimleri aranıyor...\n');
        
        % Eğitim verisi için değişken isimleri
        train_fields = fieldnames(train_data);
        fprintf('   Eğitim dosyası alanları: %s\n', strjoin(train_fields', ', '));
        
        % Çıkış değişkenini bul (y_temp veya temperature)
        if isfield(train_data, 'y_temp')
            Y_train = train_data.y_temp;
        elseif isfield(train_data, 'temperature')
            Y_train = train_data.temperature;
        elseif isfield(train_data, 'datae') && isfield(train_data.datae, 'y_temp')
            Y_train = train_data.datae.y_temp;
        else
            error('Eğitim verisinde çıkış değişkeni (y_temp/temperature) bulunamadı');
        end
        
        % Giriş değişkenlerini bul
        if isfield(train_data, 'u_curr') && isfield(train_data, 'u_soc')
            U_train = [train_data.u_curr, train_data.u_soc];
        elseif isfield(train_data, 'current') && isfield(train_data, 'SOC')
            U_train = [train_data.current, train_data.SOC];
        elseif isfield(train_data, 'datae') && isfield(train_data.datae, 'u_curr')
            U_train = [train_data.datae.u_curr, train_data.datae.u_soc];
        else
            % İlk 2 sütunu giriş olarak kullan
            all_vars = struct2cell(train_data);
            if length(all_vars) >= 3
                U_train = [all_vars{1}, all_vars{2}];
                Y_train = all_vars{3};
            else
                error('Eğitim verisinde giriş değişkenleri (u_curr, u_soc) bulunamadı');
            end
        end
        
        % Doğrulama verisi için
        val_fields = fieldnames(val_data);
        fprintf('   Doğrulama dosyası alanları: %s\n', strjoin(val_fields', ', '));
        
        if isfield(val_data, 'y_temp')
            Y_val = val_data.y_temp;
        elseif isfield(val_data, 'temperature')
            Y_val = val_data.temperature;
        elseif isfield(val_data, 'datav') && isfield(val_data.datav, 'y_temp')
            Y_val = val_data.datav.y_temp;
        else
            error('Doğrulama verisinde çıkış değişkeni (y_temp/temperature) bulunamadı');
        end
        
        if isfield(val_data, 'u_curr') && isfield(val_data, 'u_soc')
            U_val = [val_data.u_curr, val_data.u_soc];
        elseif isfield(val_data, 'current') && isfield(val_data, 'SOC')
            U_val = [val_data.current, val_data.SOC];
        elseif isfield(val_data, 'datav') && isfield(val_data.datav, 'u_curr')
            U_val = [val_data.datav.u_curr, val_data.datav.u_soc];
        else
            all_vars = struct2cell(val_data);
            if length(all_vars) >= 3
                U_val = [all_vars{1}, all_vars{2}];
                Y_val = all_vars{3};
            else
                error('Doğrulama verisinde giriş değişkenleri (u_curr, u_soc) bulunamadı');
            end
        end
    end
    
    % 5. Veriyi kısalt (isteğe bağlı, daha hızlı eğitim için)
    max_samples = 4000;
    if size(U_train, 1) > max_samples
        fprintf('   Eğitim verisi %d örneğe indiriliyor\n', max_samples);
        U_train = U_train(1:max_samples, :);
        Y_train = Y_train(1:max_samples, :);
    end
    
    if size(U_val, 1) > max_samples
        fprintf('   Doğrulama verisi %d örneğe indiriliyor\n', max_samples);
        U_val = U_val(1:max_samples, :);
        Y_val = Y_val(1:max_samples, :);
    end
    
    % 6. Boyut kontrolü
    fprintf('   Eğitim: U=%dx%d, Y=%dx%d\n', size(U_train,1), size(U_train,2), size(Y_train,1), size(Y_train,2));
    fprintf('   Doğrulama: U=%dx%d, Y=%dx%d\n', size(U_val,1), size(U_val,2), size(Y_val,1), size(Y_val,2));
end

function [U_train, Y_train, U_val, Y_val] = loadTwotankData(config)
    % Twotankdata için özel yükleyici
    
    load twotankdata;
    z_full = iddata(y, u, config.data.twotank.sampling_time);
    
    % Veriyi böl
    N_total = length(z_full.y);
    train_end = floor(N_total * config.data.train_ratio);
    val_end = train_end + floor(N_total * config.data.val_ratio);
    
    % Eğitim verisi
    if config.data.train_ratio > 0
        z1 = z_full(1:train_end);
        z1f = idfilt(z1, 3, config.data.twotank.filter_cutoff);
        z1f = z1f(config.data.twotank.warmup_samples:end);
        U_train = z1f.u;
        Y_train = z1f.y;
    else
        U_train = [];
        Y_train = [];
    end
    
    % Doğrulama verisi
    if config.data.val_ratio > 0
        z2 = z_full(train_end+1:val_end);
        z2f = idfilt(z2, 3, config.data.twotank.filter_cutoff);
        z2f = z2f(config.data.twotank.warmup_samples:end);
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
        
        % nb ve nk skaler mi vektör mü?
        if isscalar(nb)
            nb_vec = repmat(nb, 1, num_inputs);
        else
            nb_vec = nb;
        end
        
        if isscalar(nk)
            nk_vec = repmat(nk, 1, num_inputs);
        else
            nk_vec = nk;
        end
        
        % Her giriş için gecikme listesi oluştur
        input_lags_all = cell(1, num_inputs);
        for i = 1:num_inputs
            input_lags_all{i} = nk_vec(i):(nk_vec(i) + nb_vec(i) - 1);
        end
        
        % Çıkış gecikmeleri
        output_lags = 1:na;
        
    elseif strcmpi(config.regressors.type, 'custom')
        % Özel gecikmeler (tüm girişler aynı gecikmeleri kullanır)
        input_lags = config.regressors.input_lags;
        output_lags = config.regressors.output_lags;
        
        % Tüm girişler için aynı gecikmeleri kullan
        input_lags_all = cell(1, num_inputs);
        for i = 1:num_inputs
            input_lags_all{i} = input_lags;
        end
        
    else
        error('Bilinmeyen regresör tipi: %s', config.regressors.type);
    end
    
    % 2. En büyük gecikmeyi bul
    all_lags = [];
    for i = 1:num_inputs
        all_lags = [all_lags, input_lags_all{i}];
    end
    all_lags = [all_lags, output_lags];
    max_lag = max(all_lags);
    
    % 3. Regresör matrisini oluştur
    start_idx = max_lag + 1;
    X = [];
    
    % 3.1 Giriş gecikmeleri (her giriş için kendi gecikmeleri)
    for input_idx = 1:num_inputs
        input_lags = input_lags_all{input_idx};
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
    reg_info.input_lags_all = input_lags_all;
    reg_info.output_lags = output_lags;
    reg_info.max_lag = max_lag;
    reg_info.num_features = size(X_bias, 2);
    reg_info.num_samples = size(X_bias, 1);
    
    % Debug bilgisi
    fprintf('  Giriş sayısı: %d\n', num_inputs);
    fprintf('  Çıkış sayısı: %d\n', num_outputs);
    if strcmpi(config.regressors.type, 'narx')
        fprintf('  NARX: na=%d, nb=%s, nk=%s\n', na, mat2str(nb_vec), mat2str(nk_vec));
    end
    fprintf('  Maksimum gecikme: %d\n', max_lag);
    fprintf('  Regresör matrisi: %d örnek, %d özellik\n', reg_info.num_samples, reg_info.num_features);
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

function evaluateModelOptimized(X_val, T_val, w_stage1, w_final, W_hidden, g, plot_title, config)
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
        plot(Y_final, 'b-', 'DisplayName', sprintf('%d Gizli Katman - Fit: %.2f%%', num_hidden, fit_final));
        legend('show', 'Location', 'best');
        title(sprintf('%s (Fit: %.2f%%)', plot_title, fit_final));
        xlabel('Zaman Örneği');
        ylabel('Çıkış (Normalize)');
        grid on;
    end
end

function plotPerformanceSimple(T, Y_stage1, X_final_input, w_final, title_txt, config)
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
            'DisplayName', sprintf('%d Gizli Katman - Fit: %.2f%%', num_hidden_calc, fit_final));
        legend('show', 'Location', 'best');
        title(sprintf('%s (Fit: %.2f%%)', title_txt, fit_final));
        xlabel('Zaman Örneği');
        ylabel('Çıkış (Normalize)');
        grid on;
    end
end