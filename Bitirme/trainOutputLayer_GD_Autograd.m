function [w_o_trained, E_final, current_mse, Y_pred] = trainOutputLayer_GD_Autograd( ...
    X, T, w_initial, max_epochs, eta, batch_size)
% trainOutputLayer_Adam_Autograd
% Amaç: Çıkış katmanını dlarray + ADAM optimizasyonu ile eğitmek.
%
% Girdiler:
%   X          : [N x D] giriş matrisi
%   T          : [N x K] hedef matrisi
%   w_initial  : [D x K] başlangıç ağırlıkları
%   max_epochs : maksimum epoch sayısı
%   eta        : öğrenme katsayısı (learning rate)
%   batch_size : mini-batch boyutu
%
% Çıktılar:
%   w_o_trained : eğitilmiş ağırlıklar [D x K]
%   E_final     : son hata matrisi
%   current_mse : son MSE
%   Y_pred      : son tahminler

    % -------------------------
    % 1) Boyut kontrolü
    % -------------------------
    [N, num_inputs] = size(X);
    [N_T, num_outputs] = size(T);
    
    if N ~= N_T
        error('X ve T satır sayıları uyuşmuyor.');
    end
    if ~isequal(size(w_initial), [num_inputs, num_outputs])
        error('w_initial boyutu hatalı.');
    end

    % -------------------------
    % 2) Batch boyutu ayarı
    % -------------------------
    if nargin < 6 || isempty(batch_size)
        batch_size = N;
    end
    batch_size = min(batch_size, N);

    % -------------------------
    % 3) Ağırlıkları dlarray yap
    % -------------------------
    w = dlarray(w_initial);
    
    % -------------------------
    % 4) ADAM Parametreleri Başlatma
    % -------------------------
    % Adam, gradyanların hareketli ortalamalarını tutar.
    averageGrad = [];       % 1. Moment (Gradient Moving Average)
    averageSqGrad = [];     % 2. Moment (Squared Gradient Moving Average)
    iteration = 0;          % Global iterasyon sayacı
    
    % Standart Adam hiperparametreleri (İstersen fonksiyona parametre olarak ekleyebilirsin)
    gradDecay = 0.9;        % Beta1
    sqGradDecay = 0.999;    % Beta2
    epsilon = 1e-8;         % Sıfıra bölünmeyi önlemek için

    % -------------------------
    % 5) Eğitim döngüsü
    % -------------------------
    for epoch = 1:max_epochs
        % ---- 5.1) Veriyi karıştır ----
        idx = randperm(N);
        X_shuffled = X(idx, :);
        T_shuffled = T(idx, :);

        % ---- 5.2) Mini-batch döngüsü ----
        for start_idx = 1:batch_size:N
            iteration = iteration + 1; % Her update bir iterasyondur
            
            end_idx = min(start_idx + batch_size - 1, N);
            batch_idx = start_idx:end_idx;
            
            % dlarray'e çevir
            X_batch = dlarray(X_shuffled(batch_idx, :));
            T_batch = dlarray(T_shuffled(batch_idx, :));

            % ---- 5.3) Loss ve gradyan hesapla ----
            % (loss_and_grad fonksiyonu aşağıda aynı kalacak)
            [~, grad_w] = dlfeval(@loss_and_grad, w, X_batch, T_batch);

            % ---- 5.4) ADAM GÜNCELLEMESİ (Değişen Kısım) ----
            % Klasik GD yerine MATLAB'ın optimize adamupdate fonksiyonu:
            [w, averageGrad, averageSqGrad] = adamupdate(w, grad_w, ...
                averageGrad, averageSqGrad, iteration, ...
                eta, gradDecay, sqGradDecay, epsilon);
        end
        
        % İsteğe bağlı: İlerlemeyi görmek için her 50 epochta bir yazdır
        % if mod(epoch, 50) == 0
        %    [l_val, ~] = dlfeval(@loss_and_grad, w, dlarray(X), dlarray(T));
        %    fprintf('Epoch %d - MSE: %.6f\n', epoch, extractdata(l_val));
        % end
    end

    % -------------------------
    % 6) Sonuçları hesapla
    % -------------------------
    w_o_trained = extractdata(w);     % dlarray -> double
    Y_pred = X * w_o_trained;
    E_final = T - Y_pred;
    current_mse = 0.5 * mean(E_final.^2, 'all');
end

% =============================
% Yardımcı fonksiyonlar (Aynı Kalıyor)
% =============================
% function [loss, grad_w] = loss_and_grad(w, X_batch, T_batch)
%     Y = X_batch * w;                    
%     E = T_batch - Y;                    
%     loss = 0.5 * mean(E.^2, 'all');     
%     grad_w = dlgradient(loss, w);       
% end

function [loss, grad_w] = loss_and_grad(w, X_batch, T_batch)
    Y = X_batch * w;                    
    E = T_batch - Y;                    
    % MSE yerine MAE: Farkların mutlak değerinin ortalaması
    loss = mean(abs(E), 'all');     
    grad_w = dlgradient(loss, w);       
end