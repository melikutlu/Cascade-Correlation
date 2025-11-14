function [w_o_stage1_trained, E_final, current_mse] = trainOutputLayer_GD_Autograd(X, T, w_initial, ...
                                                      max_epochs, eta_output, batch_size)
% trainOutputLayer_Adam: Çıktı katmanını 'Mini-Batch' ve 'ADAM' optimizatörü ile eğitir.
% 'Adam', (Quickprop gibi) 'momentum' ve 'uyarlanabilir öğrenme oranını' birleştirir.
N = size(X, 1);
if N == 0
    error('Giriş matrisi (X) boş olamaz.');
end
% --- Başlangıç Değerlerini Ayarla ---
eta = eta_output; % Adam için temel öğrenme oranı
% <<< DEĞİŞİKLİK 1: Ağırlıkları 'dlarray' yap >>>
w_o_stage1_trained = dlarray(w_initial); 
% <<< DEĞİŞİKLİK 2: Adam optimizatörünün 'durum' (state) değişkenlerini başlat >>>
% Bunlar, Quickprop'taki 'prev_dw_o' ve 'prev_grad_o'nun modern karşılığıdır.
avg_grad = []; % Ortalama gradyan (Momentum / İvme)
avg_sq_grad = []; % Ortalama kare gradyan (Uyarlanabilir Adım)
fprintf('Çıktı katmanı eğitimi (Mini-Batch Adam Autograd) başlıyor...\n');
for epoch = 1:max_epochs
    
    indices = randperm(N);
    X_shuffled = X(indices, :);
    T_shuffled = T(indices, :);
    
    % <<< PERFORMANS DÜZELTMESİ (Hız İçin) >>>
    % dlarray'leri DÖNGÜNÜN DIŞINDA, epoch başına BİR KEZ oluştur.
    X_dl = dlarray(X_shuffled);
    T_dl = dlarray(T_shuffled);
    
    for i = 1:batch_size:N
        end_idx = min(i + batch_size - 1, N);
        batch_indices = i:end_idx;
        
        % <<< PERFORMANS DÜZELTMESİ (Hız İçin) >>>
        % Yeni dlarray oluşturmak yerine, mevcut olanı 'dilimle' (çok hızlı).
        X_batch_dl = X_dl(batch_indices, :);
        T_batch_dl = T_dl(batch_indices, :);
        
        % 1. Gradyanı 'dlfeval' ile OTOMATİK hesapla (Bu aynı)
        grad_o_batch = dlfeval(@modelGradient, w_o_stage1_trained, X_batch_dl, T_batch_dl);
        
        % <<< HATA DÜZELTMESİ (Yakınsama İçin) >>>
        % 2. Ağırlık Güncelleme (Adam Optimizatörü)
        % 'w = w - eta * grad' YERİNE, 'adamupdate' kullan.
        [w_o_stage1_trained, avg_grad, avg_sq_grad] = adamupdate(w_o_stage1_trained, grad_o_batch, ...
                                    avg_grad, avg_sq_grad, epoch, eta);
    end
end
% --- FİNAL ÇIKTILARINI AYARLA ---
w_final_double = extractdata(w_o_stage1_trained);
Y_pred_final = X * w_final_double; 

% DÜZELTME (Değişken adları imza ile eşleşmeli):
E_final = T - Y_pred_final;
final_mse = 0.5 * mean(E_final(:).^2);

fprintf('Çıktı katmanı (Mini-Batch Autograd) tamamlandı. Son MSE: %f\n', final_mse);
end % Fonksiyonun sonu
% --- YARDIMCI FONKSİYON (Değişmedi) ---
function [gradients] = modelGradient(w, X_batch, T_batch)
    Y_pred_batch = X_batch * w;
    E_batch = T_batch - Y_pred_batch;
    loss = 0.5 * mean(E_batch(:).^2); 
    gradients = dlgradient(loss, w);
end