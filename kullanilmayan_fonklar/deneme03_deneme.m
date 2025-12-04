%% KASKAD KORELASYON AĞI (CCNN) - Z-SCORE NORMALİZASYONLU TAM KOD
% Yazar: Gemini (Thought Partner)
% Tarih: 2025
% Açıklama: Two-tank sistemi için normalize edilmiş veri ile CCNN eğitimi.

clear; clc; close all; rng(0);

%% 1. VERİ YÜKLEME VE ÖN İŞLEME
load twotankdata; % System Identification Toolbox verisi
z_full = iddata(y, u, 0.2, 'Name', 'Two-tank system');

% --- Veri Bölümleme ---
% Eğitim: İlk 1500 adım
z1 = z_full(1:1500);
z1f = idfilt(z1, 3, 0.066902); % Gürültü filtresi
z1f = z1f(20:end);             % Başlangıç geçiş etkisini (Warm-up) at

% Doğrulama (Validation): Sonraki 1500 adım
z2 = z_full(1501:3000);
z2f = idfilt(z2, 3, 0.066902);
z2f = z2f(20:end);

%% 2. NORMALİZASYON (STANDARTLAŞTIRMA)
fprintf('Normalizasyon işlemi başlatılıyor...\n');

% A) İstatistikleri SADECE EĞİTİM verisinden hesapla (Veri sızmasını önlemek için)
mu_u = mean(z1f.u);  std_u = std(z1f.u);
mu_y = mean(z1f.y);  std_y = std(z1f.y);

% B) Eğitim verisini normalize et: (x - mu) / std
u_train_norm = (z1f.u - mu_u) / std_u;
y_train_norm = (z1f.y - mu_y) / std_y;

% C) Doğrulama verisini normalize et (Eğitim parametrelerini kullanarak!)
u_val_norm = (z2f.u - mu_u) / std_u;
y_val_norm = (z2f.y - mu_y) / std_y;

fprintf('Mean U: %.4f, Std U: %.4f | Mean Y: %.4f, Std Y: %.4f\n', mu_u, std_u, mu_y, std_y);

% D) Regresör Matrislerini Hazırla (Lag = 2)
[X_train, T_train] = prepareRegressors(u_train_norm, y_train_norm);
[X_val, T_val]     = prepareRegressors(u_val_norm, y_val_norm);

[N_train, num_inputs] = size(X_train);
num_outputs = 1;

%% 3. HİPERPARAMETRELER
% Çıktı katmanı eğitimi için (Gradient Descent + Momentum)
params.eta_output = 0.005;      % Öğrenme katsayısı
params.mu = 0.8;                % Momentum katsayısı
params.max_epochs_output = 500; % Maksimum epoch

% Aday birim eğitimi için
params.eta_candidate = 0.01;
params.max_epochs_candidate = 150;

% Genel ayarlar
target_mse = 1e-4;         % Hedeflenen hata (Normalize veride küçük olmalı)
max_hidden_units = 10;     % Maksimum eklenecek nöron sayısı
min_improvement = 1e-6;    % Minimum MSE iyileşmesi

% Aktivasyon fonksiyonları
g = @(x) tanh(x);          % Aktivasyon
g_prime = @(x) 1 - x.^2;   % Türev

%% AŞAMA 1: BAŞLANGIÇ AĞI EĞİTİMİ (GİZLİ KATMAN YOK)
fprintf('\n--- AŞAMA 1: Lineer Model Eğitimi ---\n');

w_o = randn(num_inputs, num_outputs) * 0.01; % Ağırlık başlatma
X_curr = X_train; % Mevcut giriş matrisi

% Çıktı katmanını eğit
[w_o, E_res, mse_curr] = trainOutputLayer(X_curr, T_train, w_o, params);

mse_history = [mse_curr];
fit_stage1 = (1 - (sum(E_res.^2) / sum((T_train - mean(T_train)).^2))) * 100;
fprintf('Başlangıç MSE: %.6f | Fit: %.2f%%\n', mse_curr, fit_stage1);

%% AŞAMA 2: GİZLİ BİRİM EKLEME DÖNGÜSÜ
W_hidden = {};     % Gizli nöron ağırlıklarını saklayacak hücre dizisi
hidden_count = 0;
X_cand = X_train;  % Aday nöronların kullanacağı giriş matrisi

fprintf('\n--- AŞAMA 2: Nöron Ekleme Döngüsü ---\n');

while mse_curr > target_mse && hidden_count < max_hidden_units
    hidden_count = hidden_count + 1;
    
    % 1. Aday Nöron Eğitimi (Korelasyon Maksimizasyonu)
    % Aday nöron, hatayla (E_res) en çok korele olacak şekilde eğitilir.
    [w_new, v_new] = trainCandidate(X_cand, E_res, params, g, g_prime);
    
    W_hidden{hidden_count} = w_new;
    
    % 2. Yeni nöronun çıktısını ana matrislere ekle
    X_curr = [X_curr, v_new]; % Çıktı katmanı için giriş
    X_cand = [X_cand, v_new]; % Gelecek adaylar için giriş
    
    % 3. Çıktı Katmanını Yeniden Eğit
    % Yeni ağırlık ekle (sıfıra yakın rastgele)
    w_o_new = [w_o; randn(1, num_outputs) * 0.01];
    
    [w_o, E_res, mse_new] = trainOutputLayer(X_curr, T_train, w_o_new, params);
    
    % İyileşme kontrolü
    improvement = mse_curr - mse_new;
    mse_curr = mse_new;
    mse_history(end+1) = mse_curr;
    
    fit_curr = (1 - (sum(E_res.^2) / sum((T_train - mean(T_train)).^2))) * 100;
    fprintf('Nöron #%d Eklendi | MSE: %.6f | Fit: %.2f%%\n', hidden_count, mse_curr, fit_curr);
    
    if improvement < min_improvement
        fprintf('*** İyileşme durdu (%g). Döngü sonlandırılıyor. ***\n', improvement);
        break;
    end
end

%% 4. SİMÜLASYON VE GERİ NORMALİZASYON (SONUÇLAR)
fprintf('\n--- Simülasyon (Free Run) Başlatılıyor ---\n');

% Simülasyon fonksiyonuna NORMALIZE verileri gönderiyoruz
[y_sim_norm, fit_sim_norm] = simulateCCNN(u_val_norm, y_val_norm, w_o, W_hidden, g);

% --- GERİ NORMALİZASYON (DENORMALIZATION) ---
% Formül: x_real = x_norm * std + mean
y_sim_real = (y_sim_norm * std_y) + mu_y;
y_val_real = z2f.y; % Orijinal gerçek veri

% Fit değerini gerçek veriler üzerinden hesapla
mse_real = mean((y_val_real - y_sim_real).^2);
fit_real = (1 - (norm(y_val_real - y_sim_real) / norm(y_val_real - mean(y_val_real)))) * 100;

fprintf('Simülasyon Tamamlandı.\n');
fprintf('Normalize Fit: %.2f%% | Gerçek Fit: %.2f%%\n', fit_sim_norm, fit_real);

%% 5. GRAFİKLER
figure('Name', 'CCNN Sonuçları', 'Color', 'w', 'Position', [100 100 1000 400]);

% A) MSE Geçmişi
subplot(1, 2, 1);
plot(0:hidden_count, mse_history, '-bo', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
title('Eğitim Süreci (MSE)');
xlabel('Eklenen Nöron Sayısı'); ylabel('MSE (Normalize)');
grid on;

% B) Simülasyon Sonucu (Gerçek Birimler)
subplot(1, 2, 2);
plot(y_val_real, 'k', 'LineWidth', 1.5); hold on;
plot(y_sim_real, 'r--', 'LineWidth', 1.2);
title(['Simülasyon Sonucu (Fit: %' num2str(fit_real, '%.2f') ')']);
legend('Gerçek Veri', 'Model Tahmini');
xlabel('Zaman Adımı'); ylabel('Su Seviyesi (Orijinal Birim)');
grid on;


%% --- YARDIMCI FONKSİYONLAR ---

% 1. Regresör Hazırlama (u(k-1), u(k-2), y(k-1), y(k-2))
function [X_bias, T] = prepareRegressors(u, y)
    L = length(u);
    % Girişler: u(k-1), u(k-2), y(k-1), y(k-2)
    X = [u(2:L-1), u(1:L-2), y(2:L-1), y(1:L-2)];
    % Hedef: y(k) (3. adımdan itibaren)
    T = y(3:L);
    % Bias Ekleme (sütun olarak 1'ler)
    X_bias = [ones(size(X, 1), 1), X];
end

% 2. Çıktı Katmanı Eğitimi (Gradient Descent + Momentum)
function [w, E, mse] = trainOutputLayer(X, T, w_init, params)
    w = w_init;
    delta_w_prev = zeros(size(w));
    N = size(X, 1);
    
    for epoch = 1:params.max_epochs_output
        Y_pred = X * w;
        E = T - Y_pred;
        
        % Gradient: dE/dw = -X' * E  (Toplu/Batch güncelleme)
        grad = - (X' * E) / N;
        
        % Momentum kuralı ile ağırlık güncelleme
        delta_w = -params.eta_output * grad + params.mu * delta_w_prev;
        w = w + delta_w;
        delta_w_prev = delta_w;
    end
    
    Y_final = X * w;
    E = T - Y_final;
    mse = mean(E.^2);
end

% 3. Aday Nöron Eğitimi (Korelasyon Maksimizasyonu - Basitleştirilmiş Gradient Ascent)
function [w_cand, v_best] = trainCandidate(X, E_res, params, g, g_prime)
    % Amaç: Adayın çıktısı (v) ile mevcut ağın hatası (E_res) arasındaki 
    % kovaryansı maksimize etmek.
    
    [N, n_in] = size(X);
    w_cand = randn(n_in, 1) * 0.1; % Rastgele başlatma
    
    % Hatanın ortalamadan arındırılmış hali (Correlation için)
    E_bar = E_res - mean(E_res);
    
    for epoch = 1:params.max_epochs_candidate
        % İleri besleme
        net = X * w_cand;
        v = g(net);
        
        % Adayın ortalamadan arındırılmış hali
        v_bar = v - mean(v);
        
        % Korelasyon (S) = sum( v_bar * E_bar )
        % Negatifini minimize etmek yerine, türevini ekliyoruz (Ascent).
        % S'in w'ya göre türevi: X' * (sign(Corr) * (E_bar) .* g_prime(net))
        % Burada tek çıkış olduğu için sign(Corr) ihmal edilebilir veya sabit kabul edilir.
        
        % Gradient hesaplama (Basit Kovaryans Gradyanı)
        d_corr = (E_bar .* g_prime(net)); 
        grad = (X' * d_corr) / N; 
        
        % Ağırlık Güncelleme (Gradient Ascent - yukarı tırmanma)
        w_cand = w_cand + params.eta_candidate * grad; 
    end
    
    % Son çıktıyı döndür
    v_best = g(X * w_cand);
end

% 4. Simülasyon (Recursive / Free Run)
function [y_sim, fit] = simulateCCNN(u, y_real, w_o, W_hidden, g_func)
    N = length(u);
    y_sim = zeros(N, 1);
    
    % İlk iki adımı gerçek veriden al (başlangıç şartı)
    y_sim(1:2) = y_real(1:2);
    
    num_hidden = length(W_hidden);
    
    for k = 3:N
        % Regresör vektörü oluştur: [Bias, u(k-1), u(k-2), y_sim(k-1), y_sim(k-2)]
        % DİKKAT: y_real yerine önceki tahmin edilen y_sim kullanılıyor (Feedback)
        x_in = [1, u(k-1), u(k-2), y_sim(k-1), y_sim(k-2)];
        
        curr_in_vec = x_in; % Bu vektör nöron eklendikçe büyüyecek
        
        % Gizli katmanlardan geçir
        for h = 1:num_hidden
            w_h = W_hidden{h};
            % Nöron girişi
            net_h = curr_in_vec * w_h;
            out_h = g_func(net_h);
            
            % Bir sonraki katman/çıkış için girişe ekle
            curr_in_vec = [curr_in_vec, out_h];
        end
        
        % Çıktı katmanı hesapla
        y_sim(k) = curr_in_vec * w_o;
    end
    
    fit = (1 - (norm(y_real - y_sim) / norm(y_real - mean(y_real)))) * 100;
end