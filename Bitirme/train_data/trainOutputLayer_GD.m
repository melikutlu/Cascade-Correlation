function [w_trained, E_final, final_mse] = trainOutputLayer_GD(X, T, w_initial, ...
                                                      max_epochs, eta_output, batch_size)
% trainOutputLayer_GD: Çıktı katmanını 'Mini-Batch Gradient Descent' ile eğitir.
% 'params' struct KULLANMAZ.
%
% ÇIKTILAR:
%   w_trained = (F x K) Eğitilmiş ağırlık matrisi
%   E_final   = (N x K) Eğitim sonrası son HATA matrisi (E_residual için)
%   final_mse = Eğitim sonrası son Ortalama Kare Hata (current_mse için)
%
% GİRDİLER (DİKKAT!):
%   eta_output: Gradient Descent için öğrenme oranı (Quickprop'tan farklıdır!)
%   batch_size: Mini-batch boyutu (örn: 32 veya 64)

N = size(X, 1); % Toplam örnek sayısı
if N == 0
    error('Giriş matrisi (X) boş olamaz.');
end

% --- Başlangıç Değerlerini Ayarla ---
w_trained = w_initial;
eta = eta_output; % Öğrenme oranını ata

fprintf('Çıktı katmanı eğitimi (Gradient Descent) başlıyor...\n');

% --- Eğitim Döngüsü (Mini-Batch Gradient Descent) ---
for epoch = 1:max_epochs
    
    % Her epoch'un başında veriyi karıştır (Stokastik doğası için önemli)
    indices = randperm(N);
    X_shuffled = X(indices, :);
    T_shuffled = T(indices, :);
    
    % Mini-batch'ler halinde döngü
    for i = 1:batch_size:N
        % Batch'in bitiş indeksini belirle (son batch'in taşmasını engelle)
        end_idx = min(i + batch_size - 1, N);
        
        % Gerekli veri aralığını al
        batch_indices = i:end_idx;
        X_batch = X_shuffled(batch_indices, :);
        T_batch = T_shuffled(batch_indices, :);
        
        N_batch = size(X_batch, 1); % Bu batch'teki örnek sayısı
        
        % --- Standart Gradient Descent Adımları ---
        
        % 1. İleri Yayılım (Forward Pass)
        Y_pred_batch = X_batch * w_trained;
        
        % 2. Hata (Error)
        E_batch = T_batch - Y_pred_batch;
        
        % 3. Gradyan (Gradient) - MSE'nin türevi
        % Not: Quickprop'taki (-E) ile aynı, ancak formül daha standart
        grad_o = X_batch' * (-E_batch) / N_batch;
        
        % 4. Ağırlık Güncelleme (Gradient "Descent" = İniş, bu yüzden EKSİ)
        w_trained = w_trained - eta * grad_o;
        
        % ------------------------------------------
    end
end

% --- FİNAL ÇIKTILARINI AYARLA ---
% Döngü bittikten sonra, son MSE ve Hatayı TÜM veri seti üzerinde hesapla
Y_pred_final = X * w_trained;
E_final = T - Y_pred_final;
final_mse = 0.5 * mean(E_final(:).^2);

fprintf('Çıktı katmanı (GD) eğitimi tamamlandı. Son MSE: %f\n', final_mse);

end % Fonksiyonun sonu