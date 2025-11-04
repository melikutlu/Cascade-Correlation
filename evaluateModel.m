function [fit_final, fit_stage1] = evaluateModel(z1f_full, data_indices, ...
                                    w_o_stage1_trained, w_o_trained, W_hidden, g, plot_title)
% evaluateModel: Eğitilmiş CCNN modelini yeni bir veri seti (Validasyon/Test)
% üzerinde test eder ve sonuçları görselleştirir. (Eski AŞAMA 4 ve 5)
%
% GİRDİLER:
%   z1f_full:            Tüm 'iddata' nesnesi
%   data_indices:        Bu test için kullanılacak veri aralığı (örn: 1501:3000)
%   w_o_stage1_trained:  Aşama 1'in (gizli katmansız) eğitilmiş ağırlıkları
%   w_o_trained:         Aşama 2'nin (son) eğitilmiş çıkış ağırlıkları
%   W_hidden:            Tüm gizli katman ağırlıklarını içeren Cell Array
%   g:                   Aktivasyon fonksiyonu (örn: @tanh)
%   plot_title:          Grafik için başlık (örn: 'Doğrulama Performansı')

fprintf('\n--- Model Değerlendirme Başlatıldı: [%s] ---\n', plot_title);

num_hidden_units = length(W_hidden); % Gizli birim sayısını ağırlıklardan al

% 1. Veriyi al ve filtrele
z_data = z1f_full(data_indices);
z_data_f = idfilt(z_data, 3, 0.066902); % Filtre parametreleri sabit
z_data_f = z_data_f(20:end);             % Warm-up periyodunu atla

% 2. Regresörleri ve hedefleri oluştur
u_data = z_data_f.u;
t_data = z_data_f.y;
X_data_regressors = [u_data(1:end-1), t_data(1:end-1)];
T_data_targets = t_data(2:end);
X_data_bias = [ones(size(X_data_regressors, 1), 1), X_data_regressors];

% --- AŞAMA 1 (Gizli Katmansız) PERFORMANS ---
Y_pred_data_stage1 = X_data_bias * w_o_stage1_trained; 
mse_data_stage1 = mean((T_data_targets - Y_pred_data_stage1).^2);
fit_stage1 = (1 - (sum((T_data_targets - Y_pred_data_stage1).^2) / ...
                   sum((T_data_targets - mean(T_data_targets)).^2))) * 100;
fprintf('Performans (Gizli Katmansız) - MSE: %f, FİT: %.2f%%\n', ...
        mse_data_stage1, fit_stage1);

% --- AŞAMA 2 (TÜM GİZLİ KATMANLARLA) PERFORMANS ---
% Ağı, veri üzerinde yeniden kur
X_data_output_input = X_data_bias; % Başlangıçta sadece orijinal girişler
X_data_candidate_input = X_data_bias;

for k = 1:num_hidden_units
    w_c_k = W_hidden{k}; % (Düzeltme Hata 4)
    
    % Bu birimin veri üzerindeki çıkışını hesapla
    V_h_k_data = g(X_data_candidate_input * w_c_k);
    
    % Bu çıkışı, bir sonraki adımlar için girdi matrislerine ekle
    X_data_output_input = [X_data_output_input, V_h_k_data];
    X_data_candidate_input = [X_data_candidate_input, V_h_k_data];
end

% Final çıkış ağırlıklarını ('w_o_trained') kullanarak son tahmini yap (Düzeltme Hata 5)
Y_pred_data_final = X_data_output_input * w_o_trained;
mse_data_final = mean((T_data_targets - Y_pred_data_final).^2);
fit_final = (1 - (sum((T_data_targets - Y_pred_data_final).^2) / ...
                 sum((T_data_targets - mean(T_data_targets)).^2))) * 100;
fprintf('Performans (%d Gizli Katmanlı) - MSE: %f, FİT: %.2f%%\n', ...
        num_hidden_units, mse_data_final, fit_final);

% --- GÖRSELLEŞTİRME (Eski AŞAMA 5) ---
figure;
hold on;
plot(T_data_targets, 'k', 'LineWidth', 1.5, 'DisplayName', 'Gerçek Veri');
plot(Y_pred_data_stage1, 'r--', 'DisplayName', sprintf('Tahmin (Gizli Katman Yok - Fit: %.2f%%)', fit_stage1));
plot(Y_pred_data_final, 'b-', 'DisplayName', sprintf('Tahmin (%d Gizli Katmanlı - Fit: %.2f%%)', num_hidden_units, fit_final));
hold off;
legend('show', 'Location', 'best');
title(plot_title); % Başlığı dinamik olarak al
xlabel('Örnek (Zaman adımı)');
ylabel('Su Seviyesi (y)');
grid on;

fprintf('İyileşme: Fit yüzdesi %.2f%% değerinden %.2f%% değerine yükseldi.\n', ...
    fit_stage1, fit_final);
end