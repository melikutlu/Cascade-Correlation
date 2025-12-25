function [w_o_trained, E_final, current_mse, Y_pred] = trainOutputLayer_GD_Autograd_1( ...
    X, T, w_initial, max_epochs, eta, ~) % batch_size argümanı artık kullanılmıyor (~)
% trainOutputLayer_GD_Autograd (FULL BATCH - NO SHUFFLE)
% Amaç: Çıkış katmanını dlarray + ADAM optimizasyonu ile TÜM VERİ SETİNİ kullanarak eğitmek.
% Not: Karıştırma (shuffle) kapalıdır.

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
    % 2) Veriyi Hazırla (Full Batch için tek seferde çevir)
    % -------------------------
    % Döngü içinde sürekli çevirmek yerine başta bir kez çevirmek hız kazandırır.
    X_dl = dlarray(X);
    T_dl = dlarray(T);
    w = dlarray(w_initial);
    
    % -------------------------
    % 3) ADAM Parametreleri Başlatma
    % -------------------------
    averageGrad = [];       
    averageSqGrad = [];     
    iteration = 0;          
    
    gradDecay = 0.9;        
    sqGradDecay = 0.999;    
    epsilon = 1e-8;         
    
    % -------------------------
    % 4) Eğitim Döngüsü (Full Batch)
    % -------------------------
    for epoch = 1:max_epochs
        
        % Full-Batch olduğu için her epoch sadece 1 iterasyondur.
        iteration = iteration + 1; 
        
        % ---- 4.1) Loss ve gradyan hesapla (TÜM VERİ İLE) ----
        [~, grad_w] = dlfeval(@loss_and_grad, w, X_dl, T_dl);
        
        % ---- 4.2) ADAM GÜNCELLEMESİ ----
        [w, averageGrad, averageSqGrad] = adamupdate(w, grad_w, ...
            averageGrad, averageSqGrad, iteration, ...
            eta, gradDecay, sqGradDecay, epsilon);
            
    end
    
    % -------------------------
    % 5) Sonuçları hesapla
    % -------------------------
    w_o_trained = extractdata(w);     % dlarray -> double
    
    % Son tahminleri hesapla (X zaten double olarak girişte vardı)
    Y_pred = X * w_o_trained;
    E_final = T - Y_pred;
    current_mse = 0.5 * mean(E_final.^2, 'all');
end

% =============================
% Yardımcı fonksiyonlar
% =============================
function [loss, grad_w] = loss_and_grad(w, X_in, T_in)
    Y = X_in * w;                    
    E = T_in - Y;                    
    loss = 0.5 * mean(E.^2, 'all');     
    grad_w = dlgradient(loss, w);       
end