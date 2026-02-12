function [fit_final, fit_stage1] = evaluateModel_1(z1f_full, data_indices, ...
                                    w_o_stage1_trained, w_o_trained, W_hidden, g, plot_title)
% evaluateModel: Eğitilmiş CCNN modelini yeni bir veri seti (Validasyon/Test)
% üzerinde test eder ve sonuçları görselleştirir. (2. Derece Yapı için güncellendi)

fprintf('\n--- Model Değerlendirme Başlatıldı: [%s] ---\n', plot_title);
num_hidden_units = length(W_hidden); % Gizli birim sayısını ağırlıklardan al

% 1. Veriyi al ve filtrele
z_data = z1f_full(data_indices);
z_data_f = idfilt(z_data, 3, 0.066902); % Filtre parametreleri sabit

% Bu sefer, k-2'ye ihtiyacımız olduğu için ilk 2 satırı atlıyoruz.
% (Warm-up ve ilk 2 veri noktası)
u_data = z_data_f.u;
t_data = z_data_f.y;
L_val = length(u_data); % Veri uzunluğu

% --- DÜZELTME 1: 2 GECİKMELİ REGRESÖR OLUŞTURMA (Hata Çözümü) ---
% Girişler: u(k-1), u(k-2), y(k-1), y(k-2)
% Veri 3. indisten başlamalı (L_val)

% Regressörler (4 Sütun):
u_val_k_eksi_1 = u_data(2:L_val-1); 
u_val_k_eksi_2 = u_data(1:L_val-2); 
y_val_k_eksi_1 = t_data(2:L_val-1); 
y_val_k_eksi_2 = t_data(1:L_val-2); 

X_data_regressors = [u_val_k_eksi_1, u_val_k_eksi_2, ...
                     y_val_k_eksi_1, y_val_k_eksi_2];
T_data_targets = t_data(3:L_val); % Hedefler y(k) (3'ten L'ye kadar)

X_data_bias = [ones(size(X_data_regressors, 1), 1), X_data_regressors];
% Artık X_data_bias matrisi [N x 5] boyutundadır (Bias + 4 Regressör).

% --- AŞAMA 1 (Gizli Katmansız) PERFORMANS ---
% W_o_stage1_trained artık 5 satırlı olduğu için çarpım çalışacaktır.
Y_pred_data_stage1 = X_data_bias * w_o_stage1_trained; 
mse_data_stage1 = mean((T_data_targets - Y_pred_data_stage1).^2);
fit_stage1 = (1 - (sum((T_data_targets - Y_pred_data_stage1).^2) / ...
                   sum((T_data_targets - mean(T_data_targets)).^2))) * 100;
fprintf('Performans (Gizli Katmansız) - MSE: %f, FİT: %.2f%%\n', ...
        mse_data_stage1, fit_stage1);

% --- AŞAMA 2 (TÜM GİZLİ KATMANLARLA) PERFORMANS ---
% Ağı, veri üzerinde yeniden kur
X_data_output_input = X_data_bias; % Başlangıçta 5 sütun (Bias + 4 Regressör)
X_data_candidate_input = X_data_bias;

for k = 1:num_hidden_units
    w_c_k = W_hidden{k}; 
    
    % Bu birimin veri üzerindeki çıkışını hesapla
    % Not: w_c_k'nın satır sayısı, X_data_candidate_input'ın sütun sayısına uyacaktır.
    V_h_k_data = g(X_data_candidate_input * w_c_k);
    
    % Bu çıkışı, bir sonraki adımlar için girdi matrislerine ekle
    X_data_output_input = [X_data_output_input, V_h_k_data];
    X_data_candidate_input = [X_data_candidate_input, V_h_k_data];
end

% Final çıkış ağırlıklarını ('w_o_trained') kullanarak son tahmini yap
Y_pred_data_final = X_data_output_input * w_o_trained;
mse_data_final = mean((T_data_targets - Y_pred_data_final).^2);
fit_final = (1 - (sum((T_data_targets - Y_pred_data_final).^2) / ...
                 sum((T_data_targets - mean(T_data_targets)).^2))) * 100;
fprintf('Performans (%d Gizli Katmanlı) - MSE: %f, FİT: %.2f%%\n', ...
        num_hidden_units, mse_data_final, fit_final);

% --- GÖRSELLEŞTİRME ---
figure;
hold on;
plot(T_data_targets, 'k', 'LineWidth', 1.5, 'DisplayName', 'Gerçek Veri');
plot(Y_pred_data_stage1, 'r--', 'DisplayName', sprintf('Tahmin (Gizli Katman Yok - Fit: %.2f%%)', fit_stage1));
plot(Y_pred_data_final, 'b-', 'DisplayName', sprintf('Tahmin (%d Gizli Katmanlı - Fit: %.2f%%)', num_hidden_units, fit_final));
hold off;
legend('show', 'Location', 'best');
title(plot_title); 
xlabel('Örnek (Zaman adımı)');
ylabel('Su Seviyesi (y)');
grid on;
fprintf('İyileşme: Fit yüzdesi %.2f%% değerinden %.2f%% değerine yükseldi.\n', ...
    fit_stage1, fit_final);
end