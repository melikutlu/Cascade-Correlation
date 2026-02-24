function [w_c_trained, v_c_final] = trainCandidateUnit_DL(X_candidate_input, E_residual, ...
                                                    max_epochs_candidate, eta_candidate, g, g_prime)
% trainCandidateUnit: Bir aday gizli birimi (w_c) eğitir.
% 'params' struct KULLANMAZ.
%
% Bu fonksiyon, Kaskad Korelasyon'un 2.a Aşamasını (Aday Eğitimi) uygular.
%
% GİRDİLER:
%   X_candidate_input    = (N x F_c) Aday birimin giriş matrisi.
%   E_residual           = (N x K) Çıktı katmanından gelen mevcut hata.
%   max_epochs_candidate = Aday eğitimi için maksimum epoch.
%   eta_candidate        = Aday eğitimi için öğrenme oranı.
%   g                    = Aktivasyon fonksiyonu (örn: @tanh)
%   g_prime              = Aktivasyon fonksiyonu türevi (örn: @(v) 1-v.^2)
%
% ÇIKTILAR:
%   w_c_trained       = (F_c x 1) Eğitilmiş aday birim ağırlıkları.
%   v_c_final         = (N x 1) Eğitilmiş birimin son aktivasyon (çıktı) vektörü.

% --- Parametreleri Yükle ---
eta = eta_candidate;
max_epochs = max_epochs_candidate;

% --- Ağırlıkları Başlat ---
[N, num_inputs] = size(X_candidate_input);
w_c = dlarray(randn(num_inputs, 1) * 0.01); % Aday ağırlıkları (dlarray)

best_S = -Inf; % En iyi korelasyon skorunu izle

fprintf('Aşama 2.a: Aday birim (w_c) eğitiliyor (Max Epoch: %d)...\n', max_epochs);

% --- Eğitim Döngüsü (Gradient Ascent) ---
for epoch = 1:max_epochs
    % İleri yayılım (Aday birim)
    a_c = X_candidate_input * w_c; % Net aktivasyon
    v_c = g(a_c);                  % Çıktı (örn: tanh(a_c))
    
    % Korelasyonu (S) hesapla
    S = sum(v_c .* E_residual); 
    sigma = sign(S);
    
    % Gradyanı hesapla (S'yi maksimize etmek için) -> dlfeval içinde olmalı
    grad_S = dlfeval(@grad_S_fun, w_c, X_candidate_input, E_residual, g, g_prime);
    
    % Ağırlıkları güncelle (Gradient Ascent - Yükseliş)
    w_c = w_c + eta * grad_S;
    
    % Erken durdurma (Korelasyon iyileşmiyorsa)
    if abs(S) > best_S
        best_S = abs(S);
    elseif epoch > max_epochs / 2 % Yarı yoldan sonra kontrol et
        % Opsiyonel: Daha esnek bir durdurma kriteri eklenebilir
        % fprintf('Epoch %d: Korelasyon iyileşmesi durdu.\n', epoch);
        % break;
    end
end

% --- Çıktıları Ayarla ---
w_c_trained = extractdata(w_c); % Çıktıyı normal double'a çevir

% Fonksiyonun son çıktısını (v_c) hesapla.
a_c_final = X_candidate_input * w_c_trained;
v_c_final = g(a_c_final);

fprintf('Aday birim eğitimi tamamlandı. (Best S: %f)\n', best_S);

end % Fonksiyonun sonu

% ------------------------------
% Korelasyon Gradyanı Hesaplama
% ------------------------------
function grad_S = grad_S_fun(w_c, X_candidate_input, E_residual, g, g_prime)
    % Net aktivasyon
    a_c = X_candidate_input * w_c;   
    v_c = g(a_c);                     
    S = sum(v_c .* E_residual);       % Korelasyon
    sigma = sign(S);
    
    % Gradyanı hesapla (S'yi maksimize etmek için)
    grad_S = X_candidate_input' * (sigma * E_residual .* g_prime(v_c));
end
