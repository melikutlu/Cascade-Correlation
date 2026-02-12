% Kaskad Korelasyon (CCNN) - BATARYA VERİSİ (FTP75 & LA92)
clear; clc; close all; rng(0);

%% 1. VERİ SETİNİ YÜKLEME VE HAZIRLIK
fprintf('Batarya verileri aranıyor ve yükleniyor...\n');

% 1. Dosyalar mevcut klasörde mi diye kontrol et
if exist('FTP75.mat', 'file') ~= 2
    fprintf('Dosyalar mevcut klasörde bulunamadı. MATLAB Toolbox dizinleri taranıyor...\n');
    
    % System Identification Toolbox varsayılan veri yolu
    toolboxDataPath = fullfile(matlabroot, 'toolbox', 'ident', 'iddemos', 'data');
    
    if exist(toolboxDataPath, 'dir')
        addpath(toolboxDataPath);
        fprintf('Veri dosyaları bulundu ve yol eklendi: %s\n', toolboxDataPath);
    else
        % Eğer toolbox yolu bulunamazsa, kullanıcıya uyarı ver ve rastgele veri üret (Test için)
        warning('FTP75.mat dosyası MATLAB dizinlerinde bulunamadı!');
        fprintf('DİKKAT: Veri seti bulunamadığı için TEST amaçlı RASTGELE veri üretiliyor.\n');
        
        % Rastgele Batarya Benzeri Veri Üretimi (Sadece kodun çalışırlığını test etmek için)
        N_sim = 3000;
        datae.u_curr = randn(N_sim, 1) + 0.5 * sin(0.01*(1:N_sim)'); % Akım
        datae.y_temp = 25 + filter(0.1, [1 -0.9], abs(datae.u_curr)) + 0.1*randn(N_sim,1); % Sıcaklık
        
        datav.u_curr = randn(N_sim, 1);
        datav.y_temp = 25 + filter(0.1, [1 -0.9], abs(datav.u_curr));
        
        % Dosyaları "bulunmuş" gibi davranması için struct oluşturuldu
        fprintf('Rastgele veri seti oluşturuldu.\n');
    end
end

% Dosyaları yüklemeyi tekrar dene (veya yukarıda oluşturulanı kullan)
if ~exist('datae', 'var') % Eğer rastgele üretilmediyse yükle
    try
        datae = load('FTP75.mat'); 
        datav = load('LA92.mat');  
        fprintf('Veri setleri başarıyla yüklendi.\n');
    catch
        error('Veri dosyaları yüklenemedi. Lütfen "FTP75.mat" ve "LA92.mat" dosyalarını script ile aynı klasöre kopyalayın.');
    end
end

% --- A) EĞİTİM VERİSİ (FTP75) ---
% Not: CCNN kodun şu an SISO (Tek Giriş) olduğu için sadece 'Current' alıyoruz.
% Eğer SOC da eklenecekse prepareRegressors fonksiyonu değiştirilmeli.
u_train_raw = double(datae.u_curr); 
y_train_raw = double(datae.y_temp);

% --- B) DOĞRULAMA VERİSİ (LA92) ---
u_val_raw = double(datav.u_curr);
y_val_raw = double(datav.y_temp);

% Veri boyutlarını ve örnekleme zamanını kontrol et
Ts = 0.1; % Batarya verisi örnekleme zamanı
fprintf('Eğitim Verisi Boyutu: %d | Doğrulama Verisi Boyutu: %d\n', length(u_train_raw), length(u_val_raw));

%% 2. NORMALİZASYON YÖNTEMİ SEÇİMİ VE UYGULAMA
% Seçenekler: 'ZScore', 'MinMax', 'None'
config.norm_method = 'ZScore'; 
fprintf('Normalizasyon Yöntemi: %s\n', config.norm_method);

% Normalizasyon Fonksiyonu Çağrısı
[u_train_norm, y_train_norm, u_val_norm, y_val_norm, norm_stats] = ...
    normalizeData(config.norm_method, u_train_raw, y_train_raw, u_val_raw, y_val_raw);

% Regresör Matrislerini Hazırla (Lag = 2)
[X_train_bias, T_train] = prepareRegressors(u_train_norm, y_train_norm);
[X_val_bias, T_val]     = prepareRegressors(u_val_norm, y_val_norm);

[N_train, num_inputs] = size(X_train_bias);
num_outputs = 1;
disp('Veri hazırlığı tamamlandı.');

% --- EĞİTİM AYARLARI ---
config.output_trainer = 'GD_Autograd'; 

%% 3. HİPERPARAMETRELER
eta_output = 0.001;
mu = 0.75;
max_epochs_output = 300;
min_mse_change = 1e-9;
epsilon = 1e-8;
eta_candidate = 0.005; 
max_epochs_candidate = 100;
g = @(a) tanh(a);
g_prime = @(v) 1 - v.^2;
target_mse = 0.00005;
max_hidden_units = 50; % Batarya verisi karmaşık olabilir, birim sayısı artırılabilir
num_hidden_units = 0; 
eta_output_gd = 0.002; 
batch_size = 64; % Veri seti büyükse batch size artırılabilir
mse_history = [];

%% AŞAMA 1: BAŞLANGIÇ AĞI EĞİTİMİ (Gizli Katmansız)
fprintf('Aşama 1: Başlangıç ağı eğitiliyor...\n');
w_o_initial = randn(num_inputs, num_outputs)*0.01; 

X_output_input = X_train_bias; 
X_candidate_input = X_train_bias; 

all_params.eta_output = eta_output;
all_params.mu = mu;
all_params.epsilon = epsilon;
all_params.eta_output_gd = eta_output_gd;
all_params.batch_size = batch_size;

% runOutputTraining fonksiyonunun çalışma klasöründe olduğundan emin ol
[w_o_stage1_trained, E_residual, current_mse] = runOutputTraining(...
    config.output_trainer, X_output_input, T_train, w_o_initial, ...
    max_epochs_output, all_params);

T_variance_sum = sum((T_train - mean(T_train)).^2);
Y_pred_stage1 = X_output_input * w_o_stage1_trained;
fit_percentage_train_stage1 = (1 - (sum(E_residual.^2) / T_variance_sum)) * 100;
                                    
fprintf('Aşama 1 MSE: %f | Fit: %.2f%%\n', current_mse, fit_percentage_train_stage1);
mse_history(1) = current_mse; 

%% AŞAMA 2: DİNAMİK BİRİM EKLEME DÖNGÜSÜ
W_hidden = {}; 
w_o_trained = w_o_stage1_trained;

fprintf('\n--- GİZLİ BİRİM EKLEME DÖNGÜSÜ ---\n');
while current_mse > target_mse && num_hidden_units < max_hidden_units
    num_hidden_units = num_hidden_units + 1;
    
    % A) Aday Birim Eğitimi
    [w_new_hidden, v_new_hidden] = trainCandidateUnit(...
        X_candidate_input, E_residual, max_epochs_candidate, eta_candidate, g, g_prime);
    
    W_hidden{num_hidden_units} = w_new_hidden;
    
    % B) Matrisleri Güncelle
    X_output_input = [X_output_input, v_new_hidden];
    X_candidate_input = [X_candidate_input, v_new_hidden];
    
    % C) Çıktı Katmanı Yeniden Eğitimi
    w_o_initial_new = [w_o_trained; randn(1, num_outputs) * 0.01];
    
    [w_o_trained, E_residual, current_mse] = runOutputTraining(...
        config.output_trainer, X_output_input, T_train, w_o_initial_new, ...
        max_epochs_output, all_params);
    
    current_fit = (1 - (sum(E_residual.^2) / T_variance_sum)) * 100;
    
    mse_history(num_hidden_units + 1) = current_mse;
    
    % Durma koşulu (İyileşme çok azsa dur)
    if (mse_history(end-1) - current_mse) < 1e-10 
        fprintf('*** İyileşme yetersiz. Döngü sonlandırılıyor. ***\n');
        break; 
    end
        
    fprintf('Gizli Birim #%d | MSE: %f | Fit: %.2f%%\n', num_hidden_units, current_mse, current_fit);
end

%% 3. PERFORMANS ANALİZİ
% --- EĞİTİM PERFORMANSI ---
plotPerformanceSimple(T_train, Y_pred_stage1, X_output_input, w_o_trained, ...
    'EĞİTİM Verisi Performansı (Normalize)');

% --- DOĞRULAMA PERFORMANSI ---
evaluateModelOptimized(X_val_bias, T_val, w_o_stage1_trained, w_o_trained, W_hidden, g, ...
    'DOĞRULAMA Verisi (One-Step Prediction - Normalize)');

%% 5. ADIM: SİMÜLASYON MODU (Free Run) ve GERİ NORMALİZASYON
fprintf('\n--- Simülasyon (Free Run) Modu ---\n');

% Simülasyon (Normalize verilerle)
[y_simulation_norm, ~] = simulateCCNNModel(u_val_norm, y_val_norm, w_o_trained, W_hidden, g);

% --- GERİ NORMALİZASYON (DENORMALIZATION) ---
y_simulation_real = denormalizeData(y_simulation_norm, config.norm_method, norm_stats.y);

% Gerçek validation çıktısı (Regresör uzunluğuna göre kırpılmalı)
% Simülasyon fonksiyonu ilk 2 adımı kopyaladığı için boyutu tamdır.
y_val_real = y_val_raw; 

% Fit Hesapla
fit_simulation_real = (1 - (norm(y_val_real - y_simulation_real) / norm(y_val_real - mean(y_val_real)))) * 100;

% Simülasyon Grafiği
figure('Name', 'Simulation Result (Real World Units)', 'Color', 'w');
plot(y_val_real, 'k', 'LineWidth', 1.5); hold on;
plot(y_simulation_real, 'r--', 'LineWidth', 1.2);
title(['Simülasyon (Gerçek Birimler - Temp) - Fit: ' num2str(fit_simulation_real, '%.2f') '%']);
legend('Gerçek (LA92)', 'Model Tahmini'); 
xlabel('Zaman (Örnek)'); ylabel('Sıcaklık (\circC)'); grid on;

%% --- YARDIMCI FONKSİYONLAR ---

% 1. Regresör Hazırlama (SISO: u ve y)
function [X_bias, T] = prepareRegressors(u, y)
    L = length(u);
    % Lag = 2 olarak ayarlı: u(k-1), u(k-2), y(k-1), y(k-2)
    X = [u(2:L-1), u(1:L-2), y(2:L-1), y(1:L-2)];
    T = y(3:L);
    X_bias = [ones(size(X, 1), 1), X];
end

% 2. Normalizasyon Fonksiyonu
function [u_tr, y_tr, u_val, y_val, stats] = normalizeData(method, u_tr_raw, y_tr_raw, u_val_raw, y_val_raw)
    stats.method = method;
    if strcmp(method, 'ZScore')
        stats.u.mean = mean(u_tr_raw); stats.u.std = std(u_tr_raw);
        stats.y.mean = mean(y_tr_raw); stats.y.std = std(y_tr_raw);
        
        u_tr = (u_tr_raw - stats.u.mean) / stats.u.std;
        y_tr = (y_tr_raw - stats.y.mean) / stats.y.std;
        
        u_val = (u_val_raw - stats.u.mean) / stats.u.std;
        y_val = (y_val_raw - stats.y.mean) / stats.y.std;
    elseif strcmp(method, 'MinMax')
        stats.u.min = min(u_tr_raw); stats.u.max = max(u_tr_raw);
        stats.y.min = min(y_tr_raw); stats.y.max = max(y_tr_raw);
        
        u_tr = (u_tr_raw - stats.u.min) / (stats.u.max - stats.u.min);
        y_tr = (y_tr_raw - stats.y.min) / (stats.y.max - stats.y.min);
        
        u_val = (u_val_raw - stats.u.min) / (stats.u.max - stats.u.min);
        y_val = (y_val_raw - stats.y.min) / (stats.y.max - stats.y.min);
    else
        u_tr = u_tr_raw; y_tr = y_tr_raw;
        u_val = u_val_raw; y_val = y_val_raw;
    end
end

% 3. Denormalizasyon Fonksiyonu
function y_real = denormalizeData(y_norm, method, stats_y)
    if strcmp(method, 'ZScore')
        y_real = (y_norm * stats_y.std) + stats_y.mean;
    elseif strcmp(method, 'MinMax')
        y_real = y_norm * (stats_y.max - stats_y.min) + stats_y.min;
    else
        y_real = y_norm;
    end
end

function evaluateModelOptimized(X_val, T_val, w_stage1, w_final, W_hidden, g, plot_title)
    num_hidden = length(W_hidden);
    X_curr = X_val;
    
    % Hızlı hesaplama için sadece final tahmin
    for k = 1:num_hidden
        V_h = g(X_curr * W_hidden{k}); % Basitleştirilmiş, candidate ayrımı yok
        X_curr = [X_curr, V_h];
    end
    Y_final = X_curr * w_final;
    fit_final = (1 - (sum((T_val - Y_final).^2) / sum((T_val - mean(T_val)).^2))) * 100;
    
    figure('Name', plot_title, 'Color', 'w');
    plot(T_val, 'k'); hold on;
    plot(Y_final, 'b--');
    title([plot_title ' Fit: ' num2str(fit_final, '%.2f') '%']); legend('Gerçek', 'Tahmin'); grid on;
end

function plotPerformanceSimple(T, Y_stage1, X_final_input, w_final, title_txt)
    Y_final = X_final_input * w_final;
    fit_final = (1 - (sum((T - Y_final).^2) / sum((T - mean(T)).^2))) * 100;
    figure('Name', title_txt, 'Color', 'w');
    plot(T, 'k'); hold on;
    plot(Y_final, 'b--');
    title([title_txt ' Fit: ' num2str(fit_final, '%.2f') '%']); grid on;
end

function [y_sim, fit_sim] = simulateCCNNModel(u_val, y_real, w_o, W_hidden, g_func)
    N = length(u_val);
    y_sim = zeros(N, 1);
    y_sim(1:2) = y_real(1:2); 
    num_hidden = length(W_hidden);
    
    for k = 3:N
        curr_in = [1, u_val(k-1), u_val(k-2), y_sim(k-1), y_sim(k-2)];
        for h = 1:num_hidden
            v = g_func(curr_in * W_hidden{h});
            curr_in = [curr_in, v];
        end
        y_sim(k) = curr_in * w_o;
    end
    fit_sim = (1 - (norm(y_real - y_sim) / norm(y_real - mean(y_real)))) * 100;
end