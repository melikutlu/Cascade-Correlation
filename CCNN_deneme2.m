% Kaskad Korelasyon (CCNN) - Geliştirilmiş Kodu
clear;
clc;
close all;
rng(0); % Tekrarlanabilir sonuçlar için

% --- ÇIKIŞ AĞIRLIKLARI İÇİN EĞİTİM YÖNTEMİ SEÇİMİ ---
% Seçenekler:
%   'Quickprop_DL'  -> trainOutputLayer_Quickprop_With_dlgrad.m (Gradyan descenti Matlab'ın kendi fonksiyonu ile kullanır.)
%   'GD_Autograd'   -> trainOutputLayer_GD_Autograd.m (Gradyan descenti Matlab'ın kendi fonksiyonu ile kullanır.)
%   'GD_Fullbatch'  -> trainOutputLayer_GD_fullbatch.m (Gradyan descenti kendi yazdığımız kod ile kullanır.)
%   'GD_MiniBatch'  -> trainOutputLayer_GD.m (Gradyan descenti kendi yazdığımız kod ile kullanır.)
%   'Quickprop_Org' -> trainOutputLayer.m (Quickprop)

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

% --- ÇIKIŞ AĞIRLIKLARI İÇİN EĞİTİM YÖNTEMİ SEÇİMİ ---
% Kullanıcıya farklı seçenekler sunuluyor.
config.output_trainer = 'GD_Fullbatch';  % 'Quickprop_Org', 'GD_Autograd', 'GD_Fullbatch', 'GD_MiniBatch', 'Quickprop_DL'

fprintf('*** Seçilen Çıkış Eğitim Yöntemi: %s ***\n', config.output_trainer);

eta_output = 0.001;
eta_output_gd = 0.005;
mu = 1.75;
max_epochs_output = 100;
min_mse_change = 1e-7;
epsilon = 1e-8;
% Gradient Ascent (Aday Katman) Parametreleri
eta_candidate = 0.000005;
max_epochs_candidate = 100;
% Aktivasyon Fonksiyonları
g = @(a) tanh(a);
g_prime = @(v) 1 - v.^2;
% --- DİNAMİK BÜYÜME PARAMETRELERİ ---
target_mse = 0.0001; % Hedeflenen MSE
max_hidden_units = 100; 
num_hidden_units = 0; 

%%%%%GRADİAN PARAMETRE

batch_size = 32;

mse_history = [];

%% AŞAMA 1: BAŞLANGIÇ AĞI EĞİTİMİ (Seçilen Yönteme Göre)
fprintf('Aşama 1: Başlangıç ağı (w_o) "%s" ile eğitiliyor...\n', config.output_trainer);

w_o_initial = randn(num_inputs, num_outputs)*0.01; % Ham başlangıç

% --- GİRİŞ MATRİSLERİNİ BAŞLATMA ---
X_output_input = X_RegressorsWithBias; % Çıkış katmanının (w_o) gördüğü
X_candidate_input = X_RegressorsWithBias; % Aday birimlerin (w_c) gördüğü


batch_size = 32;
all_params.mu = mu;
all_params.epsilon = epsilon;
all_params.eta_output_gd = eta_output_gd;
all_params.eta_output = eta_output;
all_params.batch_size = batch_size;

[w_o_stage1_trained, E_residual, current_mse] = runOutputTraining(...
    config.output_trainer, ...
    X_output_input, ...
    T_targets, ...
    w_o_initial, ...
    max_epochs_output, ...
    all_params); % Tüm hiperparametreler

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

w_o_trained = w_o_stage1_trained;

fprintf('\n--- GİZLİ BİRİM EKLEME DÖNGÜSÜ BAŞLATILDI ---\n');

while current_mse > target_mse && num_hidden_units < max_hidden_units
    num_hidden_units = num_hidden_units + 1;
    fprintf('\n--- Gizli Birim #%d Ekleniyor ---\n', num_hidden_units);
    
    % --- AŞAMA 2.a: ADAY BİRİM EĞİTİMİ ---
    [w_new_hidden, v_new_hidden] = trainCandidateUnit(X_candidate_input, E_residual, ...
                           max_epochs_candidate, eta_candidate, g, g_prime);
    
    % Ağırlıkları ileride kullanmak için sakla
    W_hidden{num_hidden_units} = w_new_hidden;
    
    % --- AŞAMA 2.b: ÇIKTI KATMANINI YENİDEN EĞİTME ---
    fprintf('Aşama 2.b: Çıktı katmanı (w_o) yeniden eğitiliyor...\n');
    
    % GİRDİ MATRİSLERİNİ GÜNCELLE:
    X_output_input = [X_output_input, v_new_hidden];
    X_candidate_input = [X_candidate_input, v_new_hidden];
    
    w_o_initial_new = [w_o_trained;  % ESKİ, EĞİTİLMİŞ AĞIRLIKLARI KORU
                       randn(1, num_outputs) * 0.01];
    
    % Çıkış katmanını GÜNCELLENMİŞ GİRDİLERLE ve seçilen metotla yeniden eğit
    [w_o_trained, E_residual, current_mse] = runOutputTraining(...
        config.output_trainer, ...
        X_output_input, ...
        T_targets, ...
        w_o_initial_new, ... % Yeni başlangıç ağırlığı
        max_epochs_output, ...
        all_params); % Tüm hiperparametreler

    current_fit = (1 - (sum(E_residual.^2) / T_variance_sum)) * 100;
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
[fit_val, fit_val_stage1] = evaluateModel(...
    z1f_full, ...
    1501:3000, ...
    w_o_stage1_trained, ...
    w_o_trained, ...
    W_hidden, ...
    g, ...
    'CCNN DOĞRULAMA Performansı');

%% 6. ADIM: KAYIP (LOSS) GELİŞİM GRAFİĞİ
plotLossHistory(mse_history, target_mse);
