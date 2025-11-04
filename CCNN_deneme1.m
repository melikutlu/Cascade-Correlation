% Kaskad Korelasyon (CCNN) - Orijinal Mimari + Yüzdelik Performans
% Amaç: Quickprop ve Gradient Ascent kullanarak 1 gizli birim ekleme.
%       Performansı her aşamada yüzdelik olarak gösterme.
%
clear;
clc;
close all;
rng default;

%% 1. VERİ SETİNİ YÜKLEME VE HAZIRLAMA
% (Bu bölüm önceki kodla aynıdır)

load twotankdata;

z1f_full = iddata(y, u, 0.2, 'Name', 'Two-tank system');
z1 = z1f_full(1:1500);
z1f = idfilt(z1, 3, 0.066902);
z1f = z1f(20:end);
u_data = z1f.u;
t_data = z1f.y;
X_regressors = [u_data(1:end-1), t_data(1:end-1)];
T_targets = t_data(2:end);
X_RegressorsWithBias = [ones(size(X_regressors, 1), 1), X_regressors];
[N, num_inputs] = size(X_RegressorsWithBias);
num_outputs = 1;
disp('Veri seti yüklendi ve hazırlandı.');

%% 2. HİPERPARAMETRELER VE YARDIMCI FONKSİYONLAR
eta_output = 0.0001;
mu = 1.75;
max_epochs_output = 300;
min_mse_change = 1e-7;
epsilon = 1e-8;
eta_candidate = 0.000005;
max_epochs_candidate = 200;
g = @(a) tanh(a);
g_prime = @(v) 1 - v.^2;

%% AŞAMA 1: BAŞLANGIÇ AĞI EĞİTİMİ (Quickprop ile)
fprintf('Aşama 1: Başlangıç ağı (w_o) Quickprop ile eğitiliyor...\n');
w_o = randn(num_inputs, num_outputs)*0.01;
%% 
prev_dw_o = zeros(size(w_o)); % Delta w_{t-1}
prev_grad_o = zeros(size(w_o)); % Gradient {t-1}
prev_mse = Inf;

for epoch = 1:max_epochs_output
    Y_pred_stage1 = X_RegressorsWithBias * w_o;
    E = T_targets - Y_pred_stage1;
    mse = 0.5 * mean(E.^2);
    grad_o = X_RegressorsWithBias' * (-E) / N;
    
    if epoch > 1
        sign_change = (grad_o .* prev_grad_o) < 0;
        step_ratio = grad_o ./ (prev_grad_o - grad_o + epsilon);
        dw_o = step_ratio .* prev_dw_o;
        dw_o(sign_change) = -eta_output * grad_o(sign_change);
        max_step = mu * abs(prev_dw_o);
        dw_o = max(-max_step, min(max_step, dw_o));
    else
        dw_o = -eta_output * grad_o;
    end
    
    w_o = w_o + dw_o;
    prev_dw_o = dw_o;
    prev_grad_o = grad_o;
    
    if abs(prev_mse - mse) < min_mse_change
        fprintf('Epoch %d: MSE değişimi durdu. (MSE: %f)\n', epoch, mse);
        break;
    end
    prev_mse = mse;
end

Y_pred_stage1 = X_RegressorsWithBias * w_o;
E_residual = T_targets - Y_pred_stage1;
mse_stage1 = mean(E_residual.^2);

% --- AŞAMA 1 FİT YÜZDESİ HESAPLAMA ---
% Fit % = (1 - (Hata Kareleri Toplamı / Hedef Varyansı)) * 100
fit_percentage_train_stage1 = (1 - (sum(E_residual.^2) / ...
                                 sum((T_targets - mean(T_targets)).^2))) * 100;
% --- BİTİŞ ---

fprintf('Aşama 1 (Gizli Katmansız) MSE: %f\n', mse_stage1);
fprintf('Aşama 1 (Gizli Katmansız) EĞİTİM Fit Yüzdesi: %.2f%%\n', fit_percentage_train_stage1); %% <-- YENİ

%% AŞAMA 2.a: ADAY BİRİM EĞİTİMİ (Gradient Ascent ile)
fprintf('Aşama 2.a: Aday birim (w_c) Gradient Ascent ile eğitiliyor...\n');
w_c = rand(num_inputs, 1);
best_S = -Inf;

for epoch = 1:max_epochs_candidate
    a_c = X_RegressorsWithBias * w_c;
    v_c = g(a_c);
    S = sum(v_c .* E_residual);
    sigma = sign(S);
    grad_S = X_RegressorsWithBias' * (sigma * E_residual .* g_prime(v_c));
    w_c = w_c + eta_candidate * grad_S;
    
    if abs(S) > best_S
        best_S = abs(S);
    elseif epoch > max_epochs_candidate / 2
        fprintf('Epoch %d: Korelasyon iyileşmesi durdu. (S_best: %f)\n', epoch, best_S);
        break;
    end
end
fprintf('Aday birim eğitimi tamamlandı.\n');

%% AŞAMA 2.b: BİRİMİ KURMA VE ÇIKIŞ AĞIRLIKLARINI YENİDEN EĞİTME
V_h1 = g(X_RegressorsWithBias * w_c);
X_final = [X_RegressorsWithBias, V_h1];
[~, num_inputs_final] = size(X_final);

fprintf('Aşama 2.b: Yeni çıkış ağırlıkları (w_o_final) Quickprop ile yeniden eğitiliyor...\n');
w_o_final = [w_o; randn(1, num_outputs) * 0.01];
prev_dw_o = zeros(size(w_o_final));
prev_grad_o = zeros(size(w_o_final));
prev_mse = Inf;

for epoch = 1:max_epochs_output
    Y_pred_final = X_final * w_o_final;
    E = T_targets - Y_pred_final;
    mse = 0.5 * mean(E.^2);
    grad_o = X_final' * (-E) / N;
    
    if epoch > 1
        sign_change = (grad_o .* prev_grad_o) < 0;
        step_ratio = grad_o ./ (prev_grad_o - grad_o + epsilon);
        dw_o = step_ratio .* prev_dw_o;
        dw_o(sign_change) = -eta_output * grad_o(sign_change);
        max_step = mu * abs(prev_dw_o);
        dw_o = max(-max_step, min(max_step, dw_o));
    else
        dw_o = -eta_output * grad_o;
    end
    
    w_o_final = w_o_final + dw_o;
    prev_dw_o = dw_o;
    prev_grad_o = grad_o;
    
    if abs(prev_mse - mse) < min_mse_change
        fprintf('Epoch %d: MSE değişimi durdu. (MSE: %f)\n', epoch, mse);
        break;
    end
    prev_mse = mse;
end

Y_pred_final = X_final * w_o_final;
mse_final = mean((T_targets - Y_pred_final).^2);

% --- AŞAMA 2 FİT YÜZDESİ HESAPLAMA ---
fit_percentage_train_final = (1 - (sum((T_targets - Y_pred_final).^2) / ...
                               sum((T_targets - mean(T_targets)).^2))) * 100;
% --- BİTİŞ ---

fprintf('Aşama 2 (1 Gizli Katmanlı) Final MSE: %f\n', mse_final);
fprintf('Aşama 2 (1 Gizli Katmanlı) Final EĞİTİM Fit Yüzdesi: %.2f%%\n', fit_percentage_train_final);

%% 3. EĞİTİM SONUÇLARINI GÖRSELLEŞTİRME
figure;
hold on;
plot(T_targets, 'k', 'LineWidth', 1.5, 'DisplayName', 'Gerçek Veri (Hedef)');
%% <-- GÜNCELLENDİ (Aşağıdaki 2 satır) -->
plot(Y_pred_stage1, 'r--', 'DisplayName', sprintf('Tahmin (Gizli Katman Yok - Fit: %.2f%%)', fit_percentage_train_stage1));
plot(Y_pred_final, 'b-', 'DisplayName', sprintf('Tahmin (1 Gizli Katmanlı - Fit: %.2f%%)', fit_percentage_train_final));
hold off;
legend('show', 'Location', 'best');
title('CCNN EĞİTİM Performansı Karşılaştırması'); %% <-- GÜNCELLENDİ
xlabel('Örnek (Zaman adımı)');
ylabel('Su Seviyesi (y)');
grid on;

fprintf('\nEğitim Setindeki İyileşme: Fit yüzdesi %.2f%% değerinden %.2f%% değerine yükseldi.\n', ...
    fit_percentage_train_stage1, fit_percentage_train_final); %% <-- YENİ


%% 4. ADIM: DOĞRULAMA VERİSİ (VALIDATION) İLE PERFORMANS TESTİ
fprintf('\n--- Doğrulama (Validation) Aşaması Başlatıldı ---\n');

% 1. Validation verisini al ve filtrele
z2 = z1f_full(1501:3000);
z2f = idfilt(z2, 3, 0.066902);
z2f = z2f(20:end);

% 2. Validation için regresörleri ve hedefleri oluştur
u_val = z2f.u;
t_val = z2f.y;
X_val_regressors = [u_val(1:end-1), t_val(1:end-1)];
T_val_targets = t_val(2:end);
X_val_bias = [ones(size(X_val_regressors, 1), 1), X_val_regressors];

% --- AŞAMA 1 (Gizli Katmansız) DOĞRULAMA PERFORMANSI --- %% <-- YENİ BÖLÜM
Y_pred_val_stage1 = X_val_bias * w_o; % (Eğitimli w_o kullanılır)
mse_val_stage1 = mean((T_val_targets - Y_pred_val_stage1).^2);
fit_percentage_val_stage1 = (1 - (sum((T_val_targets - Y_pred_val_stage1).^2) / ...
                             sum((T_val_targets - mean(T_val_targets)).^2))) * 100;
fprintf('Doğrulama (Gizli Katmansız) MSE: %f\n', mse_val_stage1);
fprintf('Doğrulama (Gizli Katmansız) FİT YÜZDESİ: %.2f%%\n', fit_percentage_val_stage1);
% --- BİTİŞ ---

% --- AŞAMA 2 (1 Gizli Katmanlı) DOĞRULAMA PERFORMANSI ---
V_h1_val = g(X_val_bias * w_c); % (Eğitimli w_c kullanılır)
X_val_final = [X_val_bias, V_h1_val];
Y_pred_val_final = X_val_final * w_o_final; % (Eğitimli w_o_final kullanılır)

mse_val_final = mean((T_val_targets - Y_pred_val_final).^2);
fit_percentage_val_final = (1 - (sum((T_val_targets - Y_pred_val_final).^2) / ...
                             sum((T_val_targets - mean(T_val_targets)).^2))) * 100;
fprintf('Doğrulama (1 Gizli Katmanlı) MSE: %f\n', mse_val_final);
fprintf('Doğrulama (1 Gizli Katmanlı) FİT YÜZDESİ: %.2f%%\n', fit_percentage_val_final);

%% 5. ADIM: DOĞRULAMA (VALIDATION) GRAFİĞİ
figure;
hold on;
plot(T_val_targets, 'k', 'LineWidth', 1.5, 'DisplayName', 'Gerçek Veri (Validation)');
%% <-- GÜNCELLENDİ (Aşağıdaki 2 satır) -->
plot(Y_pred_val_stage1, 'r--', 'DisplayName', sprintf('Tahmin (Gizli Katman Yok - Fit: %.2f%%)', fit_percentage_val_stage1));
plot(Y_pred_val_final, 'b-', 'DisplayName', sprintf('Tahmin (1 Gizli Katmanlı - Fit: %.2f%%)', fit_percentage_val_final));
hold off;
legend('show', 'Location', 'best');
title('CCNN DOĞRULAMA Performansı Karşılaştırması'); %% <-- GİNCELLE
xlabel('Örnek (Zaman adımı)');
ylabel('Su Seviyesi (y)');
grid on;

fprintf('\nDoğrulama Setindeki İyileşme: Fit yüzdesi %.2f%% değerinden %.2f%% değerine yükseldi.\n', ...
    fit_percentage_val_stage1, fit_percentage_val_final); %% <-- YENİ