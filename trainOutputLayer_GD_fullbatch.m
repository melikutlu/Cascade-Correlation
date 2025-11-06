function [w_o_stage1_trained, E_residual, current_mse] = trainOutputLayer_GD_fullbatch(X, T, w_initial, ...
                                                      max_epochs, eta_output)
% trainOutputLayer_GD: Çıktı katmanını 'Batch Gradient Descent' (TÜM VERİ SETİ) ile eğitir.
%[w_o_stage1_trained, E_residual, current_mse]
% NOT: Bu versiyon, Mini-Batch (orijinal kod) ile karşılaştırma yapabilmeniz 
% için 'batch_size' parametresini almasına rağmen, bu parametreyi KULLANMAZ.

N = size(X, 1); % Toplam örnek sayısı
if N == 0
    error('Giriş matrisi (X) boş olamaz.');
end

% --- Başlangıç Değerlerini Ayarla ---
w_o_stage1_trained = w_initial;
eta = eta_output; % Öğrenme oranını ata

fprintf('Çıktı katmanı eğitimi (Full-Batch Gradient Descent) başlıyor...\n');
% 'batch_size'ın kullanılmadığına dair bir not:
if N > 0
   % fprintf('(Not: batch_size parametresi [Full-Batch modunda] kullanılmayacaktır.)\n');
end

% --- Eğitim Döngüsü (Batch Gradient Descent) ---
% Bu döngü, ağırlık güncelleme döngüsüdür.
% Her epoch'ta ağırlıklar SADECE BİR KEZ güncellenir.
for epoch = 1:max_epochs
    

    % 1. İleri Yayılım (Forward Pass) - TÜM VERİ SETİ ('X')
    Y_pred = X * w_o_stage1_trained;
    
    % 2. Hata (Error) - TÜM VERİ SETİ ('T')
    E = T - Y_pred;
    
    % 3. Gradyan (Gradient) - TÜM VERİ SETİ ('N')
    % grad_o = X_batch' * (-E_batch) / N_batch; --- YERİNE:
    grad_o = X' * (-E) / N;
    
    % 4. Ağırlık Güncelleme (Gradient "Descent")
    % Bu güncelleme, epoch başına SADECE BİR KEZ yapılır.
    w_o_stage1_trained = w_o_stage1_trained - eta * grad_o;
    
    % ------------------------------------------
    
end % Epoch döngüsünün sonu


% --- FİNAL ÇIKTILARINI AYARLA ---
% Bu kısım orijinal kodla aynı, çünkü zaten TÜM veri seti üzerinde
% son durumu hesaplıyordu.
Y_pred_final = X * w_o_stage1_trained;
E_residual = T - Y_pred_final;
current_mse = 0.5 * mean(E_residual(:).^2);

fprintf('Çıktı katmanı (Full-Batch GD) eğitimi tamamlandı. Son MSE: %f\n', current_mse);

end % Fonksiyonun sonu