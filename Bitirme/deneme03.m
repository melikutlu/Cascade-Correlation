% Kaskad Korelasyon (CCNN)
clear;
clc;
close all;
rng(0);

%% 1. VERİ SETİNİ YÜKLEME VE HAZIRLAMA%% 1. VERİ SETİNİ YÜKLEME VE HAZIRLAMA
load twotankdata;
%Eğitim verisi
z_full = iddata(y, u, 0.2, 'Name', 'Two-tank system');
z1 = z_full(1:1500);
z1f = idfilt(z1, 3, 0.066902);
z1f = z1f(20:end);
u_data = z1f.u;
t_data = z1f.y;

%Doğrulama verisi
z2 = z_full(1501:3000);
z2f = idfilt(z2, 3, 0.066902); % Aynı filtre parametreleri
z2f = z2f(20:end);
u_val = z2f.u;
y_val = z2f.y;



% -- DÜZELTME BURADA --
L = length(u_data); % Uzunluğu baştan alıyoruz, 'end' karışıklığını önlüyoruz.

% Girişler: u(k-1), u(k-2), y(k-1), y(k-2)
% Hedef: y(k)
% Veri mecburen 3. adımdan başlayacak (çünkü k-2'ye ihtiyacımız var)

% X Regresörleri: [u(k-1), u(k-2), y(k-1), y(k-2)]
% u(k-1) -> 2'den L-1'e kadar
% u(k-2) -> 1'den L-2'ye kadar
X_regressors = [u_data(2:L-1), u_data(1:L-2), t_data(2:L-1), t_data(1:L-2)];

% Hedefler: y(k) -> 3'ten L'ye kadar
T_targets = t_data(3:L);

X_RegressorsWithBias = [ones(size(X_regressors, 1), 1), X_regressors];
[N, num_inputs] = size(X_RegressorsWithBias);
num_outputs = 1;

disp('Veri seti yüklendi. Yapı: 2. Dereceden (Lag=2)');

%% 2. HİPERPARAMETRELER VE YARDIMCI FONKSİYONLAR

% --- ÇIKIŞ AĞIRLIKLARI İÇİN EĞİTİM YÖNTEMİ SEÇİMİ ---
% Seçenekler:
%   'Quickprop_DL'  -> trainOutputLayer_Quickprop_With_dlgrad.m (Gradyan descenti Matlab'ın kendi fonksiyonu ile kullanır.)
%   'GD_Autograd'   -> trainOutputLayer_GD_Autograd.m (Gradyan descenti Matlab'ın kendi fonksiyonu ile kullanır.)
%   'GD_Fullbatch'  -> trainOutputLayer_GD_fullbatch.m (Gradyan descenti kendi yazdığımız kod ile kullanır.)
%   'GD_MiniBatch'  -> trainOutputLayer_GD.m (Gradyan descenti kendi yazdığımız kod ile kullanır.)
%   'Quickprop_Org' -> trainOutputLayer.m (Quickprop)

config.output_trainer = 'GD_Autograd'; % <-- DENEME YAPMAK İÇİN SADECE BURAYI DEĞİŞTİRİN

fprintf('*** Seçilen Çıkış Eğitim Yöntemi: %s ***\n', config.output_trainer);

eta_output = 0.001;
mu = 1.75;
max_epochs_output = 300;
min_mse_change = 1e-7;
epsilon = 1e-8;
% Gradient Ascent (Aday Katman) Parametreleri
eta_candidate = 0.00005;
max_epochs_candidate = 100;
% Aktivasyon Fonksiyonları
g = @(a) tanh(a);
g_prime = @(v) 1 - v.^2;
% --- DİNAMİK BÜYÜME PARAMETRELERİ ---
target_mse = 0.00005; % Hedeflenen MSE
max_hidden_units = 100; 
num_hidden_units = 0; 

%%%%%GRADİAN PARAMETRE
eta_output_gd = 0.005; 
batch_size = 32;

mse_history = [];
%% AŞAMA 1: BAŞLANGIÇ AĞI EĞİTİMİ (Quickprop ile)
fprintf('Aşama 1: Başlangıç ağı (w_o) Quickprop ile eğitiliyor...\n');

w_o_initial = randn(num_inputs, num_outputs)*0.01; % Ham başlangıç

% --- GİRİŞ MATRİSLERİNİ BAŞLATMA ---
X_output_input = X_RegressorsWithBias; % Çıkış katmanının (w_o) gördüğü
X_candidate_input = X_RegressorsWithBias; % Aday birimlerin (w_c) gördüğü

fprintf('Aşama 1: Başlangıç ağı (w_o) "%s" ile eğitiliyor...\n', config.output_trainer);
% Aşama 1:
% Tüm parametreleri tek bir yapıya toplayın
all_params.eta_output = eta_output;
all_params.mu = mu;
all_params.epsilon = epsilon;
all_params.eta_output_gd = eta_output_gd;
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
    % Çıkış katmanını GÜNCELLENMİŞ GİRDİLERLE ve seçilen metotla yeniden eğit
    
% Aşama 2.b:
[w_o_trained, E_residual, current_mse] = runOutputTraining(...
    config.output_trainer, ...
    X_output_input, ...
    T_targets, ...
    w_o_initial_new, ... % Yeni başlangıç ağırlığı
    max_epochs_output, ...
    all_params); % Tüm hiperparametreler

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    current_fit = (1 - (sum(E_residual.^2) / T_variance_sum)) * 100;
    
    % --- DÜZELTME BAŞLANGIÇ ---
    if num_hidden_units > 0 % Sadece en az 1 birim eklendiyse (yani döngü 2. kez çalışıyorsa) bu kontrolü yap
        
        % current_mse'yi mse_history'ye kaydetmeden önce iyileşmeyi hesapla
        mse_improvement = mse_history(end) - current_mse;
        min_mse_improvement = 1e-6; % Durma eşiği

        % Kalan MSE'yi tarihe kaydet (artık doğru sırada)
        mse_history(num_hidden_units + 1) = current_mse;
        
        % Durma Kriteri Kontrolü
        if mse_improvement < min_mse_improvement
            fprintf('Gizli Birim #%d eklendi. YENİ FİT: %.2f%%\n', num_hidden_units, current_fit);
            fprintf('*** MSE iyileşmesi durma eşiği (%.2e) altında kaldı. Döngü sonlandırılıyor. ***\n', min_mse_improvement);
            break; % While döngüsünden çık
        end
    else
        % İlk (0. birim) MSE'sini buraya kaydediyoruz
        % NOT: İlk MSE zaten döngü öncesinde kaydedildiği için bu blok gereksiz olabilir. 
        % Ancak yapıyı korumak için, yine de son MSE'yi döngü sonunda kaydetmeliyiz.
        mse_history(num_hidden_units + 1) = current_mse;
    end
    % --- DÜZELTME BİTİŞ ---
        
    % Yazdırma (fprintf) satırını güncelle
    fprintf('Gizli Birim #%d eklendi. Yeni MSE: %f | YENİ FİT: %.2f%%\n', ...
            num_hidden_units, current_mse, current_fit);
    
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
[fit_val, fit_val_stage1] = evaluateModel_1(...
    z_full, ...
    1501:3000, ... % Doğrulama verisi indisleri
    w_o_stage1_trained, ...
    w_o_trained, ...
    W_hidden, ...
    g, ...
    'CCNN DOĞRULAMA Performansı');

%% 6. ADIM: KAYIP (LOSS) GELİŞİM GRAFİĞİ
plotLossHistory(mse_history, target_mse);
%% 7. ADIM: SİMÜLASYON MODU (Recursive Prediction / Free Run)
fprintf('\n--- Simülasyon (Free Run) Modu Başlatılıyor ---\n');

% --- SİMÜLASYON FONKSİYONUNU ÇAĞIR ---
[y_simulation, fit_simulation] = simulateCCNNModel(...
    u_val, ...
    y_val, ...
    w_o_trained, ...
    W_hidden, ...
    g);

fprintf('Simülasyon (Infinite Horizon) Fit Yüzdesi: %.2f%%\n', fit_simulation);

% --- KARŞILAŞTIRMALI GRAFİK ÇİZ ---
figure('Name', 'CCNN: Prediction vs Simulation', 'Color', 'w');
time_axis = 1:length(y_val);

% 1. Gerçek Veri
plot(time_axis, y_val, 'k', 'LineWidth', 1.5, 'DisplayName', 'Gerçek Veri'); hold on;

% 2. One-Step Prediction (Eski validation sonucu - Eğer varsa buraya eklenir)
% (Burada görsel karmaşayı önlemek için sadece simülasyonu çiziyoruz, 
% ama isterseniz fit_val hesapladığınız Y_pred'i de çizebilirsiniz)

% 3. Simülasyon (Modelin kendi ürettiği feedback)
plot(time_axis, y_simulation, 'r--', 'LineWidth', 1.2, 'DisplayName', 'Simülasyon (Free Run)');

title(['Validasyon Verisi Üzerinde Simülasyon Başarısı (Fit: %' num2str(fit_simulation, '%.2f') ')']);
xlabel('Zaman Adımları');
ylabel('Çıkış (Seviye)');
legend('Location', 'Best');
grid on;



function [y_sim, fit_sim] = simulateCCNNModel(u_val, y_real_val, w_o, W_hidden, g_func)
    % 2. DERECEDEN SİMÜLASYON (u(k-1), u(k-2), y(k-1), y(k-2))
    
    N = length(u_val);
    y_sim = zeros(N, 1);
    
    % Başlangıç koşulları: İlk 2 adımı gerçek veriden alıyoruz
    % Çünkü k=2 iken k-2 (yani 0. an) elimizde yok.
    y_sim(1) = y_real_val(1); 
    y_sim(2) = y_real_val(2);
    
    num_hidden = length(W_hidden);
    
    % Döngü 3'ten başlıyor (k-2'ye erişmek için)
    for k = 3:N
        % 1. Regresörleri Oluştur
        % Yapı: [Bias, u(k-1), u(k-2), y_sim(k-1), y_sim(k-2)]
        
        u_prev1 = u_val(k-1);
        u_prev2 = u_val(k-2);
        
        y_prev1 = y_sim(k-1); % Kendi tahminimiz (Feedback)
        y_prev2 = y_sim(k-2); % Kendi tahminimiz (Feedback)
        
        current_input = [1, u_prev1, u_prev2, y_prev1, y_prev2];
        
        % 2. Kaskad (Gizli) Katmanlar
        for h = 1:num_hidden
            w_h = W_hidden{h};
            net_h = current_input * w_h;
            v_h = g_func(net_h);
            current_input = [current_input, v_h];
        end
        
        % 3. Çıkış
        y_sim(k) = current_input * w_o;
    end
    % Simülasyon çıktısı fiziksel olarak sıfırın altına düşemez.Çünkü
    % yükseklik negatif olamaz.
    %y_sim = max(0, y_sim); 
    % Fit Hesabı
    % NRMSE Fit formülü
    fit_sim = (1 - (norm(y_real_val - y_sim) / norm(y_real_val - mean(y_real_val)))) * 100;
end