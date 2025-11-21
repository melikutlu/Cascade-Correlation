function [w_o_trained, E_final, current_mse, Y_pred] = trainOutputLayer_GD_Autograd( ...
    X, T, w_initial, max_epochs, eta, batch_size)
% trainOutputLayer_GD_Autograd
% Amaç: Çıkış katmanını dlarray + klasik mini-batch gradient descent ile eğitmek.
%
% Girdiler:
%   X          : [N x D] giriş matrisi
%   T          : [N x K] hedef matrisi (sende K=1)
%   w_initial  : [D x K] başlangıç ağırlıkları
%   max_epochs : maksimum epoch sayısı
%   eta        : öğrenme katsayısı (learning rate)
%   batch_size : mini-batch boyutu
%
% Çıktılar:
%   w_o_trained : eğitilmiş ağırlıklar [D x K]
%   E_final     : son hata matrisi (T - Y_pred)
%   current_mse : son MSE (0.5 * mean(E.^2))
%   Y_pred      : tam eğitim seti için son tahminler

    % -------------------------
    % 1) Boyut kontrolü
    % -------------------------
    [N, num_inputs] = size(X);
    [N_T, num_outputs] = size(T);

    if N ~= N_T
        error('X ve T satır sayıları uyuşmuyor (X: %d, T: %d).', N, N_T);
    end

    if ~isequal(size(w_initial), [num_inputs, num_outputs])
        error('w_initial boyutu [D x K] = [%d x %d] olmalı.', ...
              num_inputs, num_outputs);
    end

    % -------------------------
    % 2) Batch boyutu ayarı
    % -------------------------
    if nargin < 6 || isempty(batch_size)
        batch_size = N;  % full batch training
    end
    batch_size = min(batch_size, N);

    % -------------------------
    % 3) Ağırlıkları dlarray yap
    % -------------------------
    w = dlarray(w_initial);   % ister CPU, ister GPU, otomatik karar verir

    % İsteğe bağlı: GPU kullanmak istersen
    % if canUseGPU
    %     w = gpuArray(w);
    % end

    % -------------------------
    % 4) Eğitim döngüsü
    % -------------------------
    for epoch = 1:max_epochs

        % ---- 4.1) Veriyi karıştır ----
        idx = randperm(N);
        X_shuffled = X(idx, :);
        T_shuffled = T(idx, :);

        % İsteğe bağlı: full-epoch MSE'yi takip etmek için
        epoch_loss = 0;
        num_batches = 0;

        % ---- 4.2) Mini-batch döngüsü ----
        for start_idx = 1:batch_size:N
            end_idx = min(start_idx + batch_size - 1, N);
            batch_idx = start_idx:end_idx;

            X_batch = dlarray(X_shuffled(batch_idx, :));
            T_batch = dlarray(T_shuffled(batch_idx, :));

            % GPU istersen:
            % if canUseGPU
            %     X_batch = gpuArray(X_batch);
            %     T_batch = gpuArray(T_batch);
            % end

            % ---- 4.3) Loss ve gradyan hesapla ----
            [loss, grad_w] = dlfeval(@loss_and_grad, w, X_batch, T_batch);

            % ---- 4.4) Klasik GRADIENT DESCENT adımı ----
            % DİKKAT: Gradient descent → w = w - eta * grad
            w = w - eta * grad_w;

            % Epoch loss takibi (isteğe bağlı)
            epoch_loss = epoch_loss + double(loss);
            num_batches = num_batches + 1;
        end

        % Bu epoch'un ortalama loss'u (sadece çıktı için, durdurma vs. kullanabilirsin)
        avg_epoch_loss = epoch_loss / num_batches;

        % İstersen ekrana yaz:
        % fprintf('Epoch %d / %d - Ortalama MSE: %.6f\n', epoch, max_epochs, avg_epoch_loss);

        % Basit bir erken durdurma istersen (opsiyonel):
        % if avg_epoch_loss < 1e-6
        %     fprintf('Erken durdurma: loss eşiğin altına indi.\n');
        %     break;
        % end

    end

    % -------------------------
    % 5) Sonuçları hesapla
    % -------------------------
    w_o_trained = extractdata(w);     % dlarray → normal double

    Y_pred = X * w_o_trained;        % [N x K]
    E_final = T - Y_pred;            % [N x K]

    % Çoklu çıktı olsa bile mean(E.^2,'all')
    current_mse = 0.5 * mean(E_final.^2, 'all');

end

% =============================
% Yardımcı fonksiyonlar
% =============================

function [loss, grad_w] = loss_and_grad(w, X_batch, T_batch)
    % X_batch: [M x D], w: [D x K], T_batch: [M x K]
    Y = X_batch * w;                    % Lineer çıkış
    E = T_batch - Y;                    % Hata
    loss = 0.5 * mean(E.^2, 'all');     % MSE (0.5 çarpanı klasik)
    grad_w = dlgradient(loss, w);       % d(loss)/d(w)
end
