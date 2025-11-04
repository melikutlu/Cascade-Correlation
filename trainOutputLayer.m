function [w_trained, E_final, final_mse] = trainOutputLayer(X, T, w_initial, ...
                                                      max_epochs_output, eta_output, mu, epsilon)
% trainOutputLayer: Çıktı katmanını eğitir.
% 'params' struct KULLANMAZ.
%
% ÇIKTILAR:
%   w_trained = (F x K) Eğitilmiş ağırlık matrisi
%   E_final   = (N x K) Eğitim sonrası son HATA matrisi (E_residual için)
%   final_mse = Eğitim sonrası son Ortalama Kare Hata (current_mse için)

% --- Parametreleri Yükle ---
N = size(X, 1);
if N == 0
    error('Giriş matrisi (X) boş olamaz.');
end

% Argüman olarak gelen parametreleri döngüde kullanılacak
% değişkenlere ata.
eta = eta_output;
max_epochs = max_epochs_output;
% 'mu' ve 'epsilon' isimleri zaten döngüdekiyle aynı olduğu
% için yeniden atamaya gerek yok.

% --- Başlangıç Değerlerini Ayarla ---
w_trained = w_initial;
prev_dw_o = zeros(size(w_trained));
prev_grad_o = zeros(size(w_trained));
prev_mse = inf;

% --- Eğitim Döngüsü (Sizin Quickprop kodunuz) ---
for epoch = 1:max_epochs
    Y_pred = X * w_trained; 
    E = T - Y_pred;
    
    mse = 0.5 * mean(E(:).^2);
    grad_o = X' * (-E) / N;
    
    if epoch > 1
        sign_change = (grad_o .* prev_grad_o) < 0;
        step_ratio = grad_o ./ (prev_grad_o - grad_o + epsilon);
        dw_o = step_ratio .* prev_dw_o;
        dw_o(sign_change) = -eta * grad_o(sign_change);
        
        max_step = mu * abs(prev_dw_o); 
        dw_o = max(-max_step, min(max_step, dw_o));
    else
        dw_o = -eta * grad_o;
    end
    
    w_trained = w_trained + dw_o; 
    prev_dw_o = dw_o;
    prev_grad_o = grad_o;
    prev_mse = mse;
end

% --- FİNAL ÇIKTILARINI AYARLA ---
% Döngüden sonra en son hata ve MSE'yi hesapla
Y_pred_final = X * w_trained;
E_final = T - Y_pred_final;
final_mse = 0.5 * mean(E_final(:).^2);

fprintf('Çıktı katmanı eğitimi tamamlandı. Son MSE: %f\n', final_mse);

end % Fonksiyonun sonu