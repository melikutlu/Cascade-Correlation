function [w_o_stage1_trained, E_residual, current_mse] = trainOutputLayer_Quickprop_With_dlgrad(X, T, w_initial, ...
                                                      max_epochs_output, eta_output, mu, epsilon)
% Quickprop algoritmasını korur, SADECE gradyan hesabını 'dlgradient' ile yapar.
% BU, ÖĞRETİCİ OLMASINA RAĞMEN TAVSİYE EDİLEN BİR YÖNTEM DEĞİLDİR.

N = size(X, 1);
if N == 0; error('Giriş matrisi (X) boş olamaz.'); end

eta = eta_output;
max_epochs = max_epochs_output;

% --- Başlangıç Değerlerini Ayarla ---
% <<< DEĞİŞİKLİK 1: Ağırlığı 'dlarray' yap >>>
w_o_stage1_trained = dlarray(w_initial); 

% Quickprop'un "durum" (state) değişkenleri hala manuel olarak tutuluyor
prev_dw_o = zeros(size(w_o_stage1_trained));
prev_grad_o = zeros(size(w_o_stage1_trained));

% <<< DEĞİŞİKLİK 2: Veriyi de 'dlarray' yap >>>
% (Sadece bir kez, çünkü bu Full-Batch)
X_dl = dlarray(X);
T_dl = dlarray(T);

fprintf('Çıktı katmanı eğitimi (Quickprop + dlgrad) başlıyor...\n');

for epoch = 1:max_epochs
    
    % <<< DEĞİŞİKLİK 3: Gradyanı 'dlfeval' ile OTOMATİK hesapla >>>
    grad_o = dlfeval(@modelGradient, w_o_stage1_trained, X_dl, T_dl);
    
    % --- Manuel gradyan hesabı SATIRLARI KALDIRILDI ---
    % Y_pred = X * w_trained; 
    % E = T - Y_pred;
    % mse = 0.5 * mean(E(:).^2);
    % grad_o = X' * (-E) / N;
    % ---------------------------------------------------
    
    % --- Quickprop'un GERİ KALAN TÜM MANTIĞI AYNEN UYGULANIR ---
    % 'grad_o' artık bir dlarray olduğu için, normal matris işlemleri
    % için 'extractdata' kullanmamız gerekebilir (veya dlarray'de kalır)
    
    if epoch > 1
        % Bu işlemlerin hepsi 'dlarray' üzerinde de çalışır
        sign_change = (grad_o .* prev_grad_o) < 0;
        step_ratio = grad_o ./ (prev_grad_o - grad_o + epsilon);
        dw_o = step_ratio .* prev_dw_o;
        dw_o(sign_change) = -eta * grad_o(sign_change);
        
        max_step = mu * abs(prev_dw_o); 
        dw_o = max(-max_step, min(max_step, dw_o));
    else
        dw_o = -eta * grad_o;
    end
    
    w_o_stage1_trained = w_o_stage1_trained + dw_o; 
    prev_dw_o = dw_o;
    prev_grad_o = grad_o;
end

% --- FİNAL ÇIKTILARINI AYARLA ---
w_final_double = extractdata(w_o_stage1_trained); % Sonucu 'double' yap
Y_pred_final = X * w_final_double;
E_residual = T - Y_pred_final;
current_mse = 0.5 * mean(E_residual(:).^2);
fprintf('Çıktı katmanı (Quickprop + dlgrad) tamamlandı. Son MSE: %f\n', current_mse);
end

% --- Yardımcı Fonksiyon (Sadece gradyanı hesaplar) ---
function grad_o = modelGradient(w, X, T)
    N = size(X, 1);
    Y_pred = X * w; 
    E = T - Y_pred;
    
    % Elle türev formülü: grad_o = X' * (-E) / N;
    % Autograd ile: Önce loss'u (kaybı) hesapla
    loss = 0.5 * mean(E(:).^2);
    
    % Sonra loss'un w'ye göre gradyanını bul
    grad_o = dlgradient(loss, w);
end