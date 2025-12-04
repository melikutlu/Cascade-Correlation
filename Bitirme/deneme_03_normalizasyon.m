% Kaskad Korelasyon (CCNN) - NORMALİZE EDİLMİŞ FULL VERSİYON
clear; clc; close all; rng(0);
%% 1. VERİ SETİNİ YÜKLEME, FİLTRELEME VE MATRİS HAZIRLIĞI
load twotankdata;
z_full = iddata(y, u, 0.2, 'Name', 'Two-tank system');
% --- A) EĞİTİM VERİSİ HAZIRLIĞI ---
z1 = z_full(1:1500);
z1f = idfilt(z1, 3, 0.066902); % Filtreleme
z1f = z1f(20:end);             % Warm-up atma
% --- B) DOĞRULAMA (VALIDATION) VERİSİ HAZIRLIĞI ---
z2 = z_full(1501:3000);
z2f = idfilt(z2, 3, 0.066902); % Aynı filtre
z2f = z2f(20:end);             % Warm-up atma
%% 2. NORMALİZASYON YÖNTEMİ SEÇİMİ VE UYGULAMA
% Seçenekler: 'ZScore', 'MinMax', 'None'
config.norm_method = 'ZScore'; 
fprintf('Normalizasyon Yöntemi: %s\n', config.norm_method);
% --- [YENİ] Modüler Fonksiyon Çağrısı ---
% Bu fonksiyon eğitim verisinden öğrenir, validasyon verisine uygular.
[u_train_norm, y_train_norm, u_val_norm, y_val_norm, norm_stats] = ...
    normalizeData(config.norm_method, z1f.u, z1f.y, z2f.u, z2f.y);
% Regresör Matrislerini Hazırla
[X_train_bias, T_train] = prepareRegressors(u_train_norm, y_train_norm);
[X_val_bias, T_val]     = prepareRegressors(u_val_norm, y_val_norm);
[N_train, num_inputs] = size(X_train_bias);
num_outputs = 1;
disp('Veri hazırlığı tamamlandı.');
% --- ÇIKIŞ AĞIRLIKLARI İÇİN EĞİTİM YÖNTEMİ SEÇİMİ ---
% Seçenekler:
%   'Quickprop_DL'  -> trainOutputLayer_Quickprop_With_dlgrad.m (Gradyan descenti Matlab'ın kendi fonksiyonu ile kullanır.)
%   'GD_Autograd'   -> trainOutputLayer_GD_Autograd.m (Gradyan descenti Matlab'ın kendi fonksiyonu ile kullanır.)
%   'GD_Fullbatch'  -> trainOutputLayer_GD_fullbatch.m (Gradyan descenti kendi yazdığımız kod ile kullanır.)
%   'GD_MiniBatch'  -> trainOutputLayer_GD.m (Gradyan descenti kendi yazdığımız kod ile kullanır.)
%   'Quickprop_Org' -> trainOutputLayer.m (Quickprop)
%% 3. HİPERPARAMETRELER
config.output_trainer = 'GD_Autograd'; 
eta_output = 0.001;
mu = 0.75;
max_epochs_output = 300;
min_mse_change = 1e-9;
epsilon = 1e-8;
eta_candidate = 0.005; % Normalize veride biraz artırılabilir
max_epochs_candidate = 100;
g = @(a) tanh(a);
g_prime = @(v) 1 - v.^2;
target_mse = 0.00005;
max_hidden_units = 100; 
num_hidden_units = 0; 
eta_output_gd = 0.002; 
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
    
    
    if (mse_history(end-1) - current_mse) < 1e-2 && num_hidden_units > 1
        fprintf('*** İyileşme yetersiz. Döngü sonlandırılıyor. ***\n');
        break; 
    end
        
    fprintf('Gizli Birim #%d | MSE: %f | Fit: %.2f%%\n', num_hidden_units, current_mse, current_fit);
end
%% 3. VE 4. ADIM: PERFORMANS ANALİZLERİ
% Not: Buradaki grafikler normalize değerler üzerinden olacaktır.
% --- A) EĞİTİM PERFORMANSI ---
plotPerformanceSimple(T_train, Y_pred_stage1, X_output_input, w_o_trained, ...
    'EĞİTİM Verisi Performansı (Normalize)');
% --- B) DOĞRULAMA (VALIDATION) PERFORMANSI ---
evaluateModelOptimized(X_val_bias, T_val, w_o_stage1_trained, w_o_trained, W_hidden, g, ...
    'DOĞRULAMA Verisi (One-Step Prediction - Normalize)');
%% 5. ADIM: SİMÜLASYON MODU (Free Run) ve GERİ NORMALİZASYON
fprintf('\n--- Simülasyon (Free Run) Modu ---\n');
% Simülasyona NORMALIZE verileri gönderiyoruz
[y_simulation_norm, ~] = simulateCCNNModel(u_val_norm, y_val_norm, w_o_trained, W_hidden, g);
% --- [YENİ] GERİ NORMALİZASYON (DENORMALIZATION) ---
% Normalize sonuçları gerçek birimlere (cm/volt vs) çeviriyoruz.
% Formül: y_real = y_norm * std + mean
y_simulation_real = denormalizeData(y_simulation_norm, config.norm_method, norm_stats.y);
y_val_real = z2f.y; % Gerçek veri
% Fit değerini gerçek veriler üzerinden hesaplıyoruz
fit_simulation_real = (1 - (norm(y_val_real - y_simulation_real) / norm(y_val_real - mean(y_val_real)))) * 100;
% Simülasyon Grafiği (Gerçek Birimler)
figure('Name', 'Simulation Result (Real World Units)', 'Color', 'w');
plot(y_val_real, 'k', 'LineWidth', 1.5); hold on;
plot(y_simulation_real, 'r--', 'LineWidth', 1.2);
title(['Simülasyon (Gerçek Birimler) - Fit: ' num2str(fit_simulation_real, '%.2f') '%']);
legend('Gerçek Veri', 'Simülasyon'); 
ylabel('Output (Real)'); grid on;
%% --- YARDIMCI FONKSİYONLAR ---
function [X_bias, T] = prepareRegressors(u, y)
    L = length(u);
    X = [u(2:L-1), u(1:L-2), y(2:L-1), y(1:L-2)];
    T = y(3:L);
    X_bias = [ones(size(X, 1), 1), X];
end
function evaluateModelOptimized(X_val, T_val, w_stage1, w_final, W_hidden, g, plot_title)
    fprintf('\n--- [%s] Değerlendiriliyor ---\n', plot_title);
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
    
    fprintf('Sonuç -> Başlangıç Fit: %.2f%% | Final Fit: %.2f%%\n', fit_stage1, fit_final);
    
    figure('Name', plot_title, 'Color', 'w');
    plot(T_val, 'k', 'LineWidth', 1.5); hold on;
    plot(Y_stage1, 'r--', 'DisplayName', sprintf('Tahmin (Gizli Katman Yok - Fit: %.2f%%)', fit_stage1));
    plot(Y_final, 'b-', 'DisplayName', sprintf('Tahmin (%d Gizli Katmanlı - Fit: %.2f%%)', num_hidden, fit_final));
    legend('show'); title([plot_title ' (Fit: ' num2str(fit_final, '%.2f') '%)']); grid on;
end
function plotPerformanceSimple(T, Y_stage1, X_final_input, w_final, title_txt)
    Y_final = X_final_input * w_final;
    fit_final = (1 - (sum((T - Y_final).^2) / sum((T - mean(T)).^2))) * 100;
    fit_stage1 = (1 - (sum((T - Y_stage1).^2) / sum((T - mean(T)).^2))) * 100;
    num_hidden_calc = size(X_final_input, 2) - 5; 
    
    figure('Name', title_txt, 'Color', 'w');
    plot(T, 'k'); hold on;
    plot(Y_stage1, 'r--', 'DisplayName', sprintf('Tahmin (Gizli Katman Yok - Fit: %.2f%%)', fit_stage1));
    plot(Y_final, 'b-', 'DisplayName', sprintf('Tahmin (%d Gizli Katmanlı - Fit: %.2f%%)', num_hidden_calc, fit_final));
    title([title_txt ' (Fit: ' num2str(fit_final, '%.2f') '%)']); legend('show'); grid on;
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
    % Normalize veride "max(0, ...)" kullanılmaz çünkü negatif değerler olabilir.
    fit_sim = (1 - (norm(y_real - y_sim) / norm(y_real - mean(y_real)))) * 100;
end

%% 6. MLP (MULTI-LAYER PERCEPTRON) İLE KIYASLAMA
fprintf('\n--- MLP (Klasik YSA) Eğitimi Başlıyor ---\n');

% 1. Veriyi Hazırla (MATLAB Toolbox sütun bazlı çalışır, Transpoze alıyoruz)
% X_train_bias'ın ilk sütunu (1'ler) bias'tır. Toolbox kendi biasını ekler,
% bu yüzden ilk sütunu siliyoruz: X(:, 2:end)
input_train = X_train_bias(:, 2:end)'; 
target_train = T_train';

% 2. MLP Ağını Oluştur
hidden_layer_size = 10; % CCNN kaç nöron bulduysa ona yakın seçebilirsin
net_mlp = fitnet(hidden_layer_size, 'trainlm'); % Levenberg-Marquardt (Hızlıdır)

% Eğitim ayarlarını sessize al (konsolu doldurmasın)
net_mlp.trainParam.showWindow = false; 
net_mlp.trainParam.epochs = 100;

% 3. Eğit
net_mlp = train(net_mlp, input_train, target_train);

% 4. MLP ile Simülasyon (Free Run - Recursive)
% CCNN simülasyonu ile aynı mantıkta, kendi ürettiğini girdiye vererek test ediyoruz.
fprintf('MLP Simülasyonu (Free Run) yapılıyor...\n');
[y_sim_mlp_norm, fit_mlp_norm] = simulateMLPRecursive(u_val_norm, y_val_norm, net_mlp);

% 5. Geri Normalizasyon (Gerçek Birimler)
y_sim_mlp_real = denormalizeData(y_sim_mlp_norm, config.norm_method, norm_stats.y);

% 6. Gerçek Fit Değeri Hesapla
fit_mlp_real = (1 - (norm(y_val_real - y_sim_mlp_real) / norm(y_val_real - mean(y_val_real)))) * 100;

fprintf('MLP Sonucu -> Normalize Fit: %.2f%% | Gerçek Fit: %.2f%%\n', fit_mlp_norm, fit_mlp_real);

%% 7. KIYASLAMA GRAFİĞİ (CCNN vs MLP)
figure('Name', 'CCNN vs MLP Kıyaslaması', 'Color', 'w');
plot(y_val_real, 'k', 'LineWidth', 1.5, 'DisplayName', 'Gerçek Veri'); hold on;
plot(y_simulation_real, 'r--', 'LineWidth', 1.2, 'DisplayName', ['CCNN (Fit: ' num2str(fit_simulation_real,'%.2f') '%)']);
plot(y_sim_mlp_real, 'b-.', 'LineWidth', 1.2, 'DisplayName', ['MLP (Fit: ' num2str(fit_mlp_real,'%.2f') '%)']);
title('Model Kıyaslaması: CCNN vs MLP');
ylabel('Su Seviyesi'); xlabel('Zaman');
legend('show'); grid on;

% --- MLP İÇİN YARDIMCI SİMÜLASYON FONKSİYONU ---
function [y_sim, fit] = simulateMLPRecursive(u_val, y_real, net)
    N = length(u_val);
    y_sim = zeros(N, 1);
    y_sim(1:2) = y_real(1:2); % Başlangıç şartı
    
    for k = 3:N
        % Giriş vektörü: [u(k-1); u(k-2); y_sim(k-1); y_sim(k-2)]
        % DİKKAT: Bias yok, çünkü 'net' nesnesi kendi biasını ekler.
        input_vec = [u_val(k-1); u_val(k-2); y_sim(k-1); y_sim(k-2)];
        
        % Tahmin (Tek adım)
        y_sim(k) = net(input_vec);
    end
    
    fit = (1 - (norm(y_real - y_sim) / norm(y_real - mean(y_real)))) * 100;
end

%% 9. MLP ONE-STEP PREDICTION EĞİTİMİ VE SONUÇLARI
fprintf('\n--- MLP One-Step Prediction Analizi ---\n');

% 1. Eğitim Verisini Hazırla (Transpoze alıyoruz: D x N formatı)
% Input: u(k-1), u(k-2), y(k-1), y(k-2) -> Hepsi GERÇEK veri
input_train = X_train_bias(:, 2:end)'; 
target_train = T_train';

% 2. Validasyon Verisini Hazırla
input_val = X_val_bias(:, 2:end)';
target_val = T_val'; % Bu y_val_norm'un aynısıdır (Lag sebebiyle kesilmiş)

% 3. MLP Ağını Oluştur ve Eğit
hidden_size = 1;
net_mlp = fitnet(hidden_size, 'trainlm');
net_mlp.trainParam.showWindow = false; % Pencere açılmasın
net_mlp.trainParam.epochs = 50;

% EĞİTİM (Burada One-Step mantığı kullanılır)
net_mlp = train(net_mlp, input_train, target_train);

% 4. SONUÇLARI GÖRME (One-Step Prediction)
% Recursive döngü YOK. Direkt matris çarpımı gibi tahmin alıyoruz.
y_pred_mlp_train_norm = net_mlp(input_train)';
y_pred_mlp_val_norm   = net_mlp(input_val)';

% 5. Geri Normalizasyon
y_pred_mlp_val_real = denormalizeData(y_pred_mlp_val_norm, config.norm_method, norm_stats.y);

% Hedef veriyi de hizalayalım (Lag=2 kesmesi)
offset = 2;
y_val_real_aligned = z2f.y(offset+1:end);

% 6. Fit Hesapla (One-Step)
fit_mlp_onestep = (1 - (norm(y_val_real_aligned - y_pred_mlp_val_real) / norm(y_val_real_aligned - mean(y_val_real_aligned)))) * 100;

fprintf('MLP One-Step Prediction Fit: %% %.4f (Çok yüksek çıkmalı)\n', fit_mlp_onestep);

% 7. GRAFİK
figure('Name', 'MLP One-Step Prediction Result', 'Color', 'w');
plot(y_val_real_aligned, 'k', 'LineWidth', 1.5); hold on;
plot(y_pred_mlp_val_real, 'b-.', 'LineWidth', 1.2);
title(['MLP One-Step Prediction (Fit: ' num2str(fit_mlp_onestep, '%.2f') '%)']);
legend('Gerçek Veri', 'MLP Tahmini');
xlabel('Zaman'); ylabel('Su Seviyesi');
grid on;