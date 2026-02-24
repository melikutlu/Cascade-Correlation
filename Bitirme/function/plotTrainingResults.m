function [Y_pred_final, fit_percentage_train_final] = plotTrainingResults(T_targets, Y_pred_stage1, fit_percentage_train_stage1, X_output_input, w_o_trained, num_hidden_units)
% plotTrainingResults: Eğitim verisi üzerindeki son tahminleri hesaplar ve
% eğitim performans grafiğini çizer. (Eski AŞAMA 3)
%
% GİRDİLER:
%   T_targets:                   (N x 1) Gerçek eğitim hedefleri
%   Y_pred_stage1:               (N x 1) Gizli katmansız modelin tahminleri
%   fit_percentage_train_stage1: (scalar) Gizli katmansız modelin fit yüzdesi
%   X_output_input:              (N x F) SON EĞİTİLMİŞ modelin giriş matrisi
%   w_o_trained:                 (F x 1) SON EĞİTİLMİŞ modelin çıkış ağırlıkları
%   num_hidden_units:            (scalar) Eklenen toplam gizli birim sayısı
%
% ÇIKTILAR:
%   Y_pred_final:                (N x 1) SON EĞİTİLMİŞ modelin tahminleri
%   fit_percentage_train_final:  (scalar) SON EĞİTİLMİŞ modelin fit yüzdesi

fprintf('\n--- AŞAMA 3: Eğitim Sonuçları Hazırlanıyor ve Görselleştiriliyor ---\n');

% Döngü bittikten sonraki SON TAHMİNİ ve FİT yüzdesini hesapla
Y_pred_final = X_output_input * w_o_trained;
E_final_train = T_targets - Y_pred_final;
fit_percentage_train_final = (1 - (sum(E_final_train.^2) / ...
                                    sum((T_targets - mean(T_targets)).^2))) * 100;

% --- Grafik Çizimi ---
figure;
hold on;
plot(T_targets, 'k', 'LineWidth', 1.5, 'DisplayName', 'Gerçek Veri (Hedef)');
plot(Y_pred_stage1, 'r--', 'DisplayName', sprintf('Tahmin (Gizli Katman Yok - Fit: %.2f%%)', fit_percentage_train_stage1));
plot(Y_pred_final, 'b-', 'DisplayName', sprintf('Tahmin (%d Gizli Katmanlı - Fit: %.2f%%)', num_hidden_units, fit_percentage_train_final));
hold off;
legend('show', 'Location', 'best');
title('CCNN EĞİTİM Performansı Karşılaştırması');
xlabel('Örnek (Zaman adımı)');
ylabel('Su Seviyesi (y)');
grid on;

fprintf('Eğitim Setindeki İyileşme: Fit yüzdesi %.2f%% değerinden %.2f%% değerine yükseldi.\n', ...
    fit_percentage_train_stage1, fit_percentage_train_final);

end