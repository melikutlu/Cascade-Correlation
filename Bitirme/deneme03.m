% Kaskad Korelasyon (CCNN) - OPTİMİZE EDİLMİŞ VERSİYON
clear; clc; close all; rng(0);

%% 1. VERİ SETİNİ YÜKLEME, FİLTRELEME VE MATRİS HAZIRLIĞI
load twotankdata;
z_full = iddata(y, u, 0.2, 'Name', 'Two-tank system');

% --- A) EĞİTİM VERİSİ HAZIRLIĞI ---
z1 = z_full(1:1500);
z1f = idfilt(z1, 3, 0.066902); % Filtreleme
z1f = z1f(20:end);             % Warm-up atma
% Regresör Matrisini Oluştur (Fonksiyon aşağıda tanımlı)
[X_train_bias, T_train] = prepareRegressors(z1f.u, z1f.y);
[N_train, num_inputs] = size(X_train_bias);
num_outputs = 1;

% --- B) DOĞRULAMA (VALIDATION) VERİSİ HAZIRLIĞI ---
z2 = z_full(1501:3000);
z2f = idfilt(z2, 3, 0.066902); % Aynı filtre
z2f = z2f(20:end);             % Warm-up atma
% Regresör Matrisini Oluştur (Fonksiyon aşağıda tanımlı)
% DİKKAT: Artık fonksiyon içinde tekrar hesaplamayacağız, buradan göndereceğiz.
[X_val_bias, T_val] = prepareRegressors(z2f.u, z2f.y);

disp('Veri setleri ve Regresör Matrisleri hazırlandı. (Lag=2)');

%% 2. HİPERPARAMETRELER
config.output_trainer = 'GD_MiniBatch'; 
eta_output = 0.001;
mu = 1.75;
max_epochs_output = 300;
min_mse_change = 1e-7;
epsilon = 1e-8;

eta_candidate = 0.00005;
max_epochs_candidate = 100;

g = @(a) tanh(a);
g_prime = @(v) 1 - v.^2;

target_mse = 0.00005;
max_hidden_units = 100; 
num_hidden_units = 0; 
eta_output_gd = 0.005; 
batch_size = 32;
mse_history = [];

%% AŞAMA 1: BAŞLANGIÇ AĞI EĞİTİMİ (Gizli Katmansız)
fprintf('Aşama 1: Başlangıç ağı eğitiliyor...\n');
w_o_initial = randn(num_inputs, num_outputs)*0.01; 

% Başlangıçta giriş matrisleri, saf regresör matrisleridir
X_output_input = X_train_bias; 
X_candidate_input = X_train_bias; 

all_params.eta_output = eta_output;
all_params.mu = mu;
all_params.epsilon = epsilon;
all_params.eta_output_gd = eta_output_gd;
all_params.batch_size = batch_size;

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
    
    % B) Matrisleri Güncelle (Yeni nöron çıktısını ekle)
    X_output_input = [X_output_input, v_new_hidden];
    X_candidate_input = [X_candidate_input, v_new_hidden];
    
    % C) Çıktı Katmanı Yeniden Eğitimi
    w_o_initial_new = [w_o_trained; randn(1, num_outputs) * 0.01];
    
    [w_o_trained, E_residual, current_mse] = runOutputTraining(...
        config.output_trainer, X_output_input, T_train, w_o_initial_new, ...
        max_epochs_output, all_params);

    current_fit = (1 - (sum(E_residual.^2) / T_variance_sum)) * 100;
    
    % Durma Kontrolü
    mse_history(num_hidden_units + 1) = current_mse;
    if (mse_history(end-1) - current_mse) < 1e-6 && num_hidden_units > 1
        fprintf('*** İyileşme yetersiz. Döngü sonlandırılıyor. ***\n');
        break; 
    end
        
    fprintf('Gizli Birim #%d | MSE: %f | Fit: %.2f%%\n', num_hidden_units, current_mse, current_fit);
end

%% 3. VE 4. ADIM: PERFORMANS ANALİZLERİ (Optimize Edildi)

% --- A) EĞİTİM PERFORMANSI ---
% X_output_input zaten son halini (tüm nöronlarla) almış durumda. Tekrar hesaplamaya gerek yok.
plotPerformanceSimple(T_train, Y_pred_stage1, X_output_input, w_o_trained, ...
    'EĞİTİM Verisi Performansı');

% --- B) DOĞRULAMA (VALIDATION) PERFORMANSI ---
% DİKKAT: Burada optimize ettiğimiz fonksiyonu kullanıyoruz.
% X_val_bias ve T_val'i zaten en başta hesaplamıştık. Tekrar hesaplamıyoruz.
evaluateModelOptimized(X_val_bias, T_val, w_o_stage1_trained, w_o_trained, W_hidden, g, ...
    'DOĞRULAMA Verisi (One-Step Prediction)');

%% 5. ADIM: SİMÜLASYON MODU (Free Run)
% Simülasyon recursive olduğu için (kendi çıktısını beslediği için)
% burada matris kullanamayız, döngü içinde hesaplamalıyız.
% Ancak temizlenmiş u_val ve y_val verilerini kullanıyoruz.
fprintf('\n--- Simülasyon (Free Run) Modu ---\n');
[y_simulation, fit_simulation] = simulateCCNNModel(z2f.u, z2f.y, w_o_trained, W_hidden, g);

% Simülasyon Grafiği
figure('Name', 'Simulation Result', 'Color', 'w');
plot(z2f.y, 'k', 'LineWidth', 1.5); hold on;
plot(y_simulation, 'r--', 'LineWidth', 1.2);
title(['Simülasyon (Free Run) - Fit: %' num2str(fit_simulation, '%.2f')]);
legend('Gerçek', 'Simülasyon'); grid on;


%% --- OPTİMİZE EDİLMİŞ YARDIMCI FONKSİYONLAR ---

function [X_bias, T] = prepareRegressors(u, y)
    % prepareRegressors: Verilen u ve y vektörlerinden Lag=2 yapısına göre
    % regresör matrisi oluşturur.
    % Çıktı: Bias eklenmiş X matrisi ve Hedef T vektörü.
    
    L = length(u);
    % Girişler: u(k-1), u(k-2), y(k-1), y(k-2)
    X = [u(2:L-1), u(1:L-2), y(2:L-1), y(1:L-2)];
    % Hedef: y(k) (3. adımdan itibaren)
    T = y(3:L);
    % Bias Ekleme
    X_bias = [ones(size(X, 1), 1), X];
end

function evaluateModelOptimized(X_val, T_val, w_stage1, w_final, W_hidden, g, plot_title)
    % Bu fonksiyon artık veri işleme YAPMAZ. Sadece hesaplar ve çizer.
    
    fprintf('\n--- [%s] Değerlendiriliyor ---\n', plot_title);
    num_hidden = length(W_hidden);
    
    % 1. Aşama 1 (Gizli Katmansız) Tahmin
    Y_stage1 = X_val * w_stage1;
    fit_stage1 = (1 - (sum((T_val - Y_stage1).^2) / sum((T_val - mean(T_val)).^2))) * 100;
    
    % 2. Aşama 2 (Tam Model) Tahmin
    % Validation verisi için de nöron çıktılarını hesaplamamız lazım
    X_curr = X_val;
    X_cand = X_val;
    
    for k = 1:num_hidden
        % Bu nöronun (W_hidden{k}) validation verisine tepkisini ölç
        V_h = g(X_cand * W_hidden{k});
        % Matrislere ekle
        X_curr = [X_curr, V_h];
        X_cand = [X_cand, V_h];
    end
    
    Y_final = X_curr * w_final;
    fit_final = (1 - (sum((T_val - Y_final).^2) / sum((T_val - mean(T_val)).^2))) * 100;
    
    fprintf('Sonuç -> Başlangıç Fit: %.2f%% | Final Fit: %.2f%%\n', fit_stage1, fit_final);
    
    % Grafik
    figure('Name', plot_title, 'Color', 'w');
    plot(T_val, 'k', 'LineWidth', 1.5); hold on;
    plot(Y_stage1, 'r:', 'LineWidth', 1);
    plot(Y_final, 'b-', 'LineWidth', 1.2);
    legend('Gerçek', 'Başlangıç (Linear)', 'Final (CCNN)');
    title([plot_title ' (Fit: ' num2str(fit_final, '%.2f') '%)']);
    grid on;
end

function plotPerformanceSimple(T, Y_stage1, X_final_input, w_final, title_txt)
    % Eğitim performansını çizmek için basit fonksiyon
    Y_final = X_final_input * w_final;
    fit = (1 - (sum((T - Y_final).^2) / sum((T - mean(T)).^2))) * 100;
    
    figure('Name', title_txt, 'Color', 'w');
    plot(T, 'k'); hold on;
    plot(Y_stage1, 'r:');
    plot(Y_final, 'b');
    title([title_txt ' (Fit: ' num2str(fit, '%.2f') '%)']);
    legend('Gerçek', 'Başlangıç', 'Final'); grid on;
end

function [y_sim, fit_sim] = simulateCCNNModel(u_val, y_real, w_o, W_hidden, g_func)
    % Simülasyon mecburen döngü ile yapılmak zorundadır (Recursive)
    N = length(u_val);
    y_sim = zeros(N, 1);
    y_sim(1:2) = y_real(1:2); % Başlangıç şartı
    
    num_hidden = length(W_hidden);
    
    for k = 3:N
        % Regresörler (Lag=2)
        curr_in = [1, u_val(k-1), u_val(k-2), y_sim(k-1), y_sim(k-2)];
        
        % Nöronlar
        for h = 1:num_hidden
            v = g_func(curr_in * W_hidden{h});
            curr_in = [curr_in, v];
        end
        y_sim(k) = curr_in * w_o;
    end
    y_sim = max(0, y_sim);
    fit_sim = (1 - (norm(y_real - y_sim) / norm(y_real - mean(y_real)))) * 100;
end