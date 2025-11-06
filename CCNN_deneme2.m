% Kaskad Korelasyon (CCNN) - DÜZELTİLMİŞ KOD
% Amaç: Ağı, hedef hataya ulaşana kadar otomatik olarak büyütme.
clear;
clc;
close all;
rng(0); % Tekrarlanabilir sonuçlar için

%% 1. VERİ SETİNİ YÜKLEME VE HAZIRLAMA
load twotankdata;
z1f_full = iddata(y, u, 0.2, 'Name', 'Two-tank system');
z1 = z1f_full(1:1500);
z1f = idfilt(z1, 3, 0.066902);
z1f = z1f(20:end);
u_data = z1f.u;
t_data = z1f.y;
X_regressors = [u_data(1:end-1), t_data(1:end-1)];
T_targets = t_data(2:end);
X_RegressorsWithBias = [ones(size(X_regressors, 1), 1), X_regressors];
[N, num_inputs] = size(X_RegressorsWithBias);
num_outputs = 1;
disp('Veri seti yüklendi ve hazırlandı.');

%% 2. HİPERPARAMETRELER VE YARDIMCI FONKSİYONLAR
% Quickprop (Çıkış Katmanı) Parametreleri
eta_output = 0.0001;
mu = 1.75;
max_epochs_output = 50;
min_mse_change = 1e-7;
epsilon = 1e-8;
% Gradient Ascent (Aday Katman) Parametreleri
eta_candidate = 0.000005;
max_epochs_candidate = 100;
% Aktivasyon Fonksiyonları
g = @(a) tanh(a);
g_prime = @(v) 1 - v.^2;
% --- DİNAMİK BÜYÜME PARAMETRELERİ ---
target_mse = 0.001; % Hedeflenen MSE
max_hidden_units = 100; 
num_hidden_units = 0; 

%%%%%GRADİAN PARAMETRE
eta_output_gd = 0.005; % **** DENEME YAPILMALI ****
%max_epochs_output = 100; % GD genellikle daha fazla epoch ister
batch_size = 32;

mse_history = [];
%% AŞAMA 1: BAŞLANGIÇ AĞI EĞİTİMİ (Quickprop ile)
fprintf('Aşama 1: Başlangıç ağı (w_o) Quickprop ile eğitiliyor...\n');

w_o_initial = randn(num_inputs, num_outputs)*0.01; % Ham başlangıç

% --- GİRİŞ MATRİSLERİNİ BAŞLATMA ---
X_output_input = X_RegressorsWithBias; % Çıkış katmanının (w_o) gördüğü
X_candidate_input = X_RegressorsWithBias; % Aday birimlerin (w_c) gördüğü

% --- DÜZELTME (Hata 1) ---
% trainOutputLayer fonksiyonu zaten eğitimi yapar.
% Dönen değerler, Aşama 1'in nihai sonuçlarıdır.
%%%%%%%%%QUICKPROP TRAINING%%%%%%%%%%%%%%%%%
% [w_o_stage1_trained, E_residual, current_mse] = trainOutputLayer(...
%     X_output_input, ...
%     T_targets, ...
%     w_o_initial, ...
%     max_epochs_output, ... 
%     eta_output, ...        
%     mu, ...                
%     epsilon);
%%%%%GRADİAN TRAINING%%%%%%%%%%%%%%%%%%%
[w_o_stage1_trained, E_residual, current_mse] = trainOutputLayer_GD(...
    X_output_input, T_targets, w_o_initial, ...
    max_epochs_output, eta_output_gd, batch_size);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%GRADIAN FULL BATCH TEST%%%%%%%%%%%%%%%%%%%%%%55
% [w_o_stage1_trained, E_residual, current_mse] = trainOutputLayer_GD_fullbatch(X_output_input, T_targets, w_o_initial, ...
%                                                       max_epochs_output, eta_output_gd)
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_variance_sum = sum((T_targets - mean(T_targets)).^2);

% DÜZELTME: Tahmin ve hata, EĞİTİLMİŞ w_o_stage1_trained ile hesaplanır
Y_pred_stage1 = X_output_input * w_o_stage1_trained;
% E_residual ve current_mse zaten fonksiyondan geldi.
% Sadece FİT yüzdesini hesaplamamız gerekiyor:
fit_percentage_train_stage1 = (1 - (sum(E_residual.^2) / T_variance_sum)) * 100;
                                    
fprintf('Aşama 1 (Gizli Katmansız) MSE: %f\n', current_mse);
fprintf('Aşama 1 (Gizli Katmansız) EĞİTİM Fit Yüzdesi: %.2f%%\n', fit_percentage_train_stage1);
mse_history(1) = current_mse; % İlk (0 gizli birim) MSE'sini kaydet
%% AŞAMA 2: DİNAMİK BİRİM EKLEME DÖNGÜSÜ
W_hidden = {}; % Dondurulmuş aday birim ağırlıklarını saklamak için

% DÜZELTME (Hata 5): w_o_trained, döngüde güncellenecek olan son ağırlıktır
% Başlangıç değeri, Aşama 1'de eğittiğimiz değerdir.
w_o_trained = w_o_stage1_trained;

fprintf('\n--- GİZLİ BİRİM EKLEME DÖNGÜSÜ BAŞLATILDI ---\n');

while current_mse > target_mse && num_hidden_units < max_hidden_units
    num_hidden_units = num_hidden_units + 1;
    fprintf('\n--- Gizli Birim #%d Ekleniyor ---\n', num_hidden_units);
    
    % --- AŞAMA 2.a: ADAY BİRİM EĞİTİMİ ---
    [w_new_hidden, v_new_hidden] = ...
        trainCandidateUnit(X_candidate_input, E_residual, ...
                           max_epochs_candidate, eta_candidate, g, g_prime);
    
    % Ağırlıkları ileride kullanmak için sakla (Hata 4 için kontrol)
    W_hidden{num_hidden_units} = w_new_hidden;
    
    % --- AŞAMA 2.b: ÇIKTI KATMANINI YENİDEN EĞİTME ---
    fprintf('Aşama 2.b: Çıktı katmanı (w_o) yeniden eğitiliyor...\n');
    
    % GİRDİ MATRİSLERİNİ GÜNCELLE:
    X_output_input = [X_output_input, v_new_hidden];
    X_candidate_input = [X_candidate_input, v_new_hidden];
    
    % Strateji: Ağırlıkları sıfırdan eğit (Sizin "daha basit" dediğiniz yöntem)
    [~, num_output_inputs_new] = size(X_output_input);
    %w_o_initial_new = randn(num_output_inputs_new, num_outputs) * 0.01;
    w_o_initial_new = [w_o_trained;  % ESKİ, EĞİTİLMİŞ AĞIRLIKLARI KORU
                       randn(1, num_outputs) * 0.01];
    
    % --- DÜZELTME (Hata 2) ---
    % trainOutputLayer'ı GÜNCELLENMİŞ GİRDİLERLE ve
    % YENİ BAŞLANGIÇ AĞIRLIĞI (w_o_initial_new) ile yeniden eğit
    %%%%%%%%%QUICKDROP TRAINING%%%%%%%%%%%%%%%%%
    % [w_o_trained, E_residual, current_mse] = trainOutputLayer(...
    %     X_output_input, ...    % Boyut [N, 4 + num_hidden_units]
    %     T_targets, ...
    %     w_o_initial_new, ... % Boyut [4 + num_hidden_units, 1] <-- DOĞRU!
    %     max_epochs_output, ... 
    %     eta_output, ...        
    %     mu, ...                
    %     epsilon);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%GRADİAN TRAINING%%%%%%%%%%%%%%

[w_o_trained, E_residual, current_mse] = trainOutputLayer_GD(...
    X_output_input, T_targets, w_o_initial_new, ...
    max_epochs_output, eta_output_gd, batch_size);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    current_fit = (1 - (sum(E_residual.^2) / T_variance_sum)) * 100;
    % --- YENİ KOD BİTİŞ ---
        
    % Yazdırma (fprintf) satırını güncelle
    fprintf('Gizli Birim #%d eklendi. Yeni MSE: %f | YENİ FİT: %.2f%%\n', ...
            num_hidden_units, current_mse, current_fit);

    mse_history(num_hidden_units + 1) = current_mse;
end
fprintf('--- Gizli Birim Ekleme Döngüsü Tamamlandı. Toplam %d birim eklendi ---\n', num_hidden_units);

%% 3. EĞİTİM SONUÇLARINI GÖRSELLEŞTİRME

% (Eski AŞAMA 3'ün yerine geçer)
[Y_pred_final, fit_percentage_train_final] = plotTrainingResults(...
    T_targets, ...
    Y_pred_stage1, ...
    fit_percentage_train_stage1, ...
    X_output_input, ...
    w_o_trained, ...
    num_hidden_units);

%% 4. ADIM: DOĞRULAMA (VALIDATION) İLE PERFORMANS TESTİ
% (Eski AŞAMA 4 ve 5'in yerine geçer)
[fit_val, fit_val_stage1] = evaluateModel(...
    z1f_full, ...
    1501:3000, ... % Doğrulama verisi indisleri
    w_o_stage1_trained, ...
    w_o_trained, ...
    W_hidden, ...
    g, ...
    'CCNN DOĞRULAMA Performansı');

%% 6. ADIM: KAYIP (LOSS) GELİŞİM GRAFİĞİ
plotLossHistory(mse_history, target_mse);