% Kaskad Korelasyon (CCNN) - GENELLEŞTİRİLMİŞ VERSİYON
clear; clc; close all; rng(0);

%% 0. KONFİGÜRASYON - KULLANICI BURAYI DÜZENLER
% =========================================================================
config = struct();

% ==== VERİ KAYNAĞI AYARLARI ====
config.data.source = 'twotankdata';  % 'twotankdata', 'csv', 'mat', 'workspace'
config.data.filepath = '';           % CSV/MAT dosyası için yol (ör: 'data/mydata.csv')

% CSV/MAT için sütun ayarları (isteğe bağlı)
% config.data.input_columns = [1, 2, 3];    % Giriş sütun numaraları
% config.data.output_columns = [4, 5];      % Çıkış sütun numaraları

% Twotankdata özel ayarları
config.data.twotank.filter_cutoff = 0.066902;
config.data.twotank.warmup_samples = 20;
config.data.twotank.sampling_time = 0.2;

% Veri bölme oranları (0-1 arası, toplam 1 olmalı)
config.data.train_ratio = 0.5;    % Eğitim oranı
config.data.val_ratio = 0.5;      % Doğrulama oranı
config.data.test_ratio = 0;       % Test oranı (0 = kullanma)

% ==== NORMALİZASYON AYARLARI ====
% Seçenekler: 'ZScore', 'MinMax', 'None'
config.norm_method = 'ZScore';

% ==== REGRESÖR AYARLARI (NARX STYLE) ====
config.regressors.type = 'narx';     % 'narx' veya 'custom'
config.regressors.na = 1;            % Çıkış gecikme sayısı (y(k-1)...y(k-na))
config.regressors.nb = 1;            % Giriş gecikme sayısı (u(k-nk)...u(k-nk-nb+1))
config.regressors.nk = 0;            % Giriş gecikmesi (delay)
config.regressors.include_bias = false;

% ==== ÖZEL GECİKMELER İÇİN (type='custom' ise) ====
% config.regressors.input_lags = [1, 2, 3];
% config.regressors.output_lags = [1, 2, 4, 6];

% ==== GİRİŞ/ÇIKIŞ BOYUTLARI ====
config.model.num_inputs = 1;      % Giriş değişkeni sayısı (u boyutu)
config.model.num_outputs = 1;     % Çıkış değişkeni sayısı (y boyutu)

% ==== MODEL HİPERPARAMETRELERİ ====
config.model.max_hidden_units = 100;
config.model.target_mse = 0.00005;
config.model.output_trainer = 'GD_Autograd';
config.model.eta_output = 0.001;
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
    
    load SIEngineData IOData

    eData = IOData(1:6e4,:);     % portion used for estimation
    vData = IOData(6e4+1:end,:); % portion used for validation

    [eDataN, C, S] = normalize(eData);
    s_x = table2array(S);
    c_x = table2array(C);

    vDataN = (vData - c_x) ./ s_x;

    % Downsample datasets 10 times
    eDataD = idresamp(eDataN,[1 10]);
    vDataD = idresamp(vDataN,[1 10]);
    
    xTrain = table2array(eDataD);
    out = xTrain(:, end);
    xTrain(:, end) = [];
    xTrain = [out xTrain];

    U_train = xTrain(:,2:end);
    Y_train = xTrain(:,1);

    xTest = table2array(vDataD);
    out_val = xTest(:, end);
    xTest(:, end) = [];
    xTest = [out_val xTest];

    U_val = xTest(:,2:end);
    Y_val = xTest(:,1);

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