%% Two-Tank Veri Seti için NLARX Modeli
clear all; close all; clc;

%% 1. Two-Tank Veri Setini Yükleme ve Hazırlama
% MATLAB'ın hazır two-tank veri setini yükle
load twotankdata.mat

% Veriyi incele
whos  % Değişkenleri görüntüle

% Genellikle twotankdata.mat içinde:
% u: giriş sinyali (pompa gerilimi)
% y: çıkış sinyali (tank seviyesi)
% Ts: örnekleme zamanı

% Eğer değişken isimleri farklıysa düzenle
% Örnek: u, y, Ts değişkenlerini kontrol et
if exist('u', 'var') && exist('y', 'var')
    disp('u ve y değişkenleri bulundu');
elseif exist('input', 'var') && exist('output', 'var')
    u = input;
    y = output;
    disp('input/output değişkenleri u/y olarak yeniden adlandırıldı');
end

% Örnekleme zamanını kontrol et
if ~exist('Ts', 'var')
    Ts = 1;  % Varsayılan örnekleme zamanı
    disp('Ts tanımlı değil, varsayılan Ts=1 kullanılıyor');
end

% iddata objesi oluştur
data = iddata(y, u, Ts);

% Veri seti bilgileri
fprintf('\n=== VERI SETI BILGILERI ===\n');
fprintf('Toplam veri sayisi: %d\n', length(y));
fprintf('Ornekleme zamani: %f saniye\n', Ts);
fprintf('Giris sinyali araligi: [%f, %f]\n', min(u), max(u));
fprintf('Cikis sinyali araligi: [%f, %f]\n', min(y), max(y));

%% 2. Veriyi Eğitim ve Validasyon Olarak Böl
% Verinin %70'i eğitim, %30'u validasyon
n = length(y);
n_train = round(0.7 * n);
n_valid = n - n_train;

train_data = data(1:n_train);
valid_data = data(n_train+1:end);

fprintf('\n=== VERI BOLUNMESI ===\n');
fprintf('Egitim verisi: %d ornek (%d%%)\n', n_train, round(100*n_train/n));
fprintf('Validasyon verisi: %d ornek (%d%%)\n', n_valid, round(100*n_valid/n));

%% 3. Verileri Görselleştir
figure('Position', [100, 100, 1200, 800]);

% Alt grafik 1: Tüm veri seti
subplot(3,1,1);
plot(data.u, 'b-', 'LineWidth', 1); hold on;
plot(data.y, 'r-', 'LineWidth', 1);
xlabel('Örnek Indeksi');
ylabel('Genlik');
title('Two-Tank Sistemi - Tüm Veri Seti');
legend('Giris (Pompa Gerilimi)', 'Cikis (Tank Seviyesi)');
grid on;

% Alt grafik 2: Eğitim ve validasyon ayrımı (giriş)
subplot(3,1,2);
plot(train_data.u, 'g-', 'LineWidth', 1.5); hold on;
plot(n_train+1:n, valid_data.u, 'm-', 'LineWidth', 1.5);
xlabel('Örnek Indeksi');
ylabel('Giris (Pompa Gerilimi)');
title('Giris Sinyali - Egitim ve Validasyon');
legend('Egitim Verisi', 'Validasyon Verisi');
grid on;
xline(n_train, 'k--', 'LineWidth', 2);

% Alt grafik 3: Eğitim ve validasyon ayrımı (çıkış)
subplot(3,1,3);
plot(train_data.y, 'g-', 'LineWidth', 1.5); hold on;
plot(n_train+1:n, valid_data.y, 'm-', 'LineWidth', 1.5);
xlabel('Örnek Indeksi');
ylabel('Cikis (Tank Seviyesi)');
title('Cikis Sinyali - Egitim ve Validasyon');
legend('Egitim Verisi', 'Validasyon Verisi');
grid on;
xline(n_train, 'k--', 'LineWidth', 2);

%% 4. NLARX Modeli Oluşturma
% Model derecelerini belirle
na = 2;  % Output derecesi (geçmiş çıkışlar)
nb = 2;  % Input derecesi (geçmiş girişler)
nk = 1;  % Gecikme

fprintf('\n=== NLARX MODEL PARAMETRELERI ===\n');
fprintf('na (output derecesi): %d\n', na);
fprintf('nb (input derecesi): %d\n', nb);
fprintf('nk (gecikme): %d\n', nk);

% Nonlinear fonksiyon seçenekleri:
% 'wavenet' - Wavelet ağ
% 'sigmoidnet' - Sigmoid ağ
% 'treepartition' - Karar ağacı
% 'neuralnet' - Sinir ağı

% Wavelet Network ile NLARX
fprintf('\nWavelet Network modeli egitiliyor...\n');
nlarx_wave = nlarx(train_data, [na nb nk], 'wavenet');

% Sigmoid Network ile NLARX
fprintf('Sigmoid Network modeli egitiliyor...\n');
nlarx_sig = nlarx(train_data, [na nb nk], 'sigmoidnet');

% Neural Network ile NLARX
fprintf('Neural Network modeli egitiliyor...\n');
nlarx_nn = nlarx(train_data, [na nb nk], 'neuralnet');

%% 5. Modelleri Doğrulama (fitpercent ile)
fprintf('\n=== MODEL DOGRULAMA SONUCLARI ===\n');

% Wavelet Network
yp_wave = predict(nlarx_wave, valid_data);
fit_wave = fitpercent(valid_data.y, yp_wave.y);
fprintf('Wavelet Network Fit: %.2f%%\n', fit_wave);

% Sigmoid Network
yp_sig = predict(nlarx_sig, valid_data);
fit_sig = fitpercent(valid_data.y, yp_sig.y);
fprintf('Sigmoid Network Fit: %.2f%%\n', fit_sig);

% Neural Network
yp_nn = predict(nlarx_nn, valid_data);
fit_nn = fitpercent(valid_data.y, yp_nn.y);
fprintf('Neural Network Fit: %.2f%%\n', fit_nn);

% En iyi modeli bul
[best_fit, best_idx] = max([fit_wave, fit_sig, fit_nn]);
modeller = {'Wavelet', 'Sigmoid', 'Neural'};
fprintf('\nEn iyi model: %s Network (Fit = %.2f%%)\n', modeller{best_idx}, best_fit);

%% 6. Görsel Karşılaştırma
figure('Position', [100, 100, 1400, 900]);

% Alt grafik 1: Wavelet Network
subplot(3,1,1);
plot(valid_data.y, 'b-', 'LineWidth', 1.5); hold on;
plot(yp_wave.y, 'r--', 'LineWidth', 1.5);
xlabel('Örnek Indeksi');
ylabel('Tank Seviyesi');
title(['Wavelet Network - Fit: ', num2str(fit_wave, '%.2f'), '%']);
legend('Gerçek Çıkış', 'Tahmin');
grid on;

% Alt grafik 2: Sigmoid Network
subplot(3,1,2);
plot(valid_data.y, 'b-', 'LineWidth', 1.5); hold on;
plot(yp_sig.y, 'r--', 'LineWidth', 1.5);
xlabel('Örnek Indeksi');
ylabel('Tank Seviyesi');
title(['Sigmoid Network - Fit: ', num2str(fit_sig, '%.2f'), '%']);
legend('Gerçek Çıkış', 'Tahmin');
grid on;

% Alt grafik 3: Neural Network
subplot(3,1,3);
plot(valid_data.y, 'b-', 'LineWidth', 1.5); hold on;
plot(yp_nn.y, 'r--', 'LineWidth', 1.5);
xlabel('Örnek Indeksi');
ylabel('Tank Seviyesi');
title(['Neural Network - Fit: ', num2str(fit_nn, '%.2f'), '%']);
legend('Gerçek Çıkış', 'Tahmin');
grid on;

%% 7. Detaylı Analiz (En iyi model için)
best_model = eval(['nlarx_' lower(modeller{best_idx})]);

figure('Position', [100, 100, 1200, 1000]);

% Alt grafik 1: Tahmin vs Gerçek
subplot(3,2,[1,2]);
plot(valid_data.y, 'b-', 'LineWidth', 2); hold on;
yp_best = predict(best_model, valid_data);
plot(yp_best.y, 'r--', 'LineWidth', 2);
xlabel('Örnek Indeksi');
ylabel('Tank Seviyesi');
title([modeller{best_idx}, ' Network Model - Fit: ', num2str(best_fit, '%.2f'), '%']);
legend('Gerçek', 'Tahmin');
grid on;

% Alt grafik 2: Hata
subplot(3,2,3);
hata = valid_data.y - yp_best.y;
plot(hata, 'k-', 'LineWidth', 1);
xlabel('Örnek Indeksi');
ylabel('Hata');
title('Tahmin Hatasi');
grid on;
yline(0, 'r--');

% Alt grafik 3: Hata histogramı
subplot(3,2,4);
histogram(hata, 30, 'FaceColor', 'b', 'EdgeColor', 'k');
xlabel('Hata');
ylabel('Frekans');
title('Hata DAgilimi');

% Alt grafik 4: Korelasyon
subplot(3,2,5);
scatter(valid_data.y, yp_best.y, 20, 'filled');
hold on;
min_val = min([valid_data.y; yp_best.y]);
max_val = max([valid_data.y; yp_best.y]);
plot([min_val max_val], [min_val max_val], 'r--', 'LineWidth', 2);
xlabel('Gerçek Değer');
ylabel('Tahmin Değeri');
title('Gerçek vs Tahmin Korelasyonu');
grid on;
axis equal;

% Alt grafik 5: Residual analizi
subplot(3,2,6);
resid(best_model, valid_data);

%% 8. Performans Metrikleri
fprintf('\n=== PERFORMANS METRIKLERI ===\n');
fprintf('En iyi model: %s Network\n', modeller{best_idx});

% MSE
mse = mean((valid_data.y - yp_best.y).^2);
fprintf('MSE (Mean Squared Error): %.6f\n', mse);

% RMSE
rmse = sqrt(mse);
fprintf('RMSE (Root Mean Squared Error): %.6f\n', rmse);

% MAE
mae = mean(abs(valid_data.y - yp_best.y));
fprintf('MAE (Mean Absolute Error): %.6f\n', mae);

% R²
SS_res = sum((valid_data.y - yp_best.y).^2);
SS_tot = sum((valid_data.y - mean(valid_data.y)).^2);
R2 = 1 - SS_res/SS_tot;
fprintf('R² (Determinasyon Katsayisi): %.4f\n', R2);

% MAPE (y değerleri sıfıra yakın değilse)
if min(abs(valid_data.y)) > 0.01
    mape = mean(abs((valid_data.y - yp_best.y)./valid_data.y)) * 100;
    fprintf('MAPE (Mean Absolute Percentage Error): %.2f%%\n', mape);
end

%% 9. Model Geçerlilik Testleri
fprintf('\n=== MODEL GECERLILIK TESTLERI ===\n');

% 1. Otokorelasyon testi (residualler)
[e, r] = resid(best_model, valid_data);
conf = 1.96/sqrt(length(e.y));  % %95 güven aralığı

if max(abs(r.y(2:end))) < conf
    fprintf('✓ Residual otokorelasyon testi GECTI\n');
else
    fprintf('✗ Residual otokorelasyon testi KALDI\n');
end

% 2. Bağımsızlık testi (giriş-residual korelasyonu)
[xe, xr] = resid(best_model, valid_data, 'corr');
if max(abs(xr.y)) < conf
    fprintf('✓ Giriş-residual bağımsızlık testi GECTI\n');
else
    fprintf('✗ Giriş-residual bağımsızlık testi KALDI\n');
end

%% 10. Simülasyon Modunda Doğrulama
% 1-adım ileri tahmin yerine serbest simülasyon
ys_sim = sim(best_model, valid_data.u);
fit_sim = fitpercent(valid_data.y, ys_sim.y);

figure('Position', [100, 100, 800, 400]);
plot(valid_data.y, 'b-', 'LineWidth', 2); hold on;
plot(ys_sim, 'r--', 'LineWidth', 2);
xlabel('Örnek Indeksi');
ylabel('Tank Seviyesi');
title(['Serbest Simülasyon Modu - Fit: ', num2str(fit_sim, '%.2f'), '%']);
legend('Gerçek', 'Simülasyon');
grid on;

fprintf('\n=== SIMULASYON MODU PERFORMANSI ===\n');
fprintf('Serbest simulasyon fit: %.2f%%\n', fit_sim);
fprintf('1-adim tahmin fit: %.2f%%\n', best_fit);

%% yardımcı fonksiyon
function fit = fitpercent(y_true, y_pred)
    % fitpercent - İki sinyal arasındaki uyum yüzdesini hesaplar
    % Bu fonksiyon MATLAB'ın compare fonksiyonundaki fit metriğini kullanır
    
    y_true = y_true(:);
    y_pred = y_pred(:);
    
    % NRMSE (Normalized Root Mean Square Error) bazlı fit
    fit = 100 * (1 - norm(y_true - y_pred) / norm(y_true - mean(y_true)));
    
    % Eğer fit değeri çok düşük veya negatif çıkarsa
    if fit < 0
        fit = max(0, fit);  % Negatif değerleri 0'a çek
    end
end