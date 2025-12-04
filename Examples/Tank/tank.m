%% 1. Veri Hazırlık ve Filtreleme
clc; clear; close all;

load twotankdata
z_all = iddata(y, u, 0.2, 'Name', 'Two-tank system');

% Eğitim ve Doğrulama verilerini ayır
z1 = z_all(1:1500);      % Eğitim
z2 = z_all(1501:3000);   % Doğrulama

% Filtreleme (Gürültü azaltma)
z1f = idfilt(z1, 3, 0.066902);
z2f = idfilt(z2, 3, 0.066902);
z1f = z1f(20:end);
z2f = z2f(20:end);

%% 2. Model Kurulumu ve Eğitim
f = idNeuralNetwork("cascade-correlation", "sigmoid", 0, 0, ...
    'MaxNumActLayers', 50, 'SizeSelection', 'off');

opt = nlarxOptions;
opt.SearchOptions.MaxIterations = 0; 
opt.NormalizationOptions.NormalizationMethod = 'norm';
%opt.CrossValidationOptions.HoldoutFraction = 0.1;

orders = [2 2 0]; 

fprintf('Model eğitiliyor, lütfen bekleyiniz...\n');
sys1 = nlarx(z1f, orders, f, opt);

%% 3. DETAYLI RAPORLAMA (DÜZELTİLMİŞ HALİ)

% A) Başarı Oranını (Fit %) Hesapla
[y_sim, fit_yuzdesi, ~] = compare(z2f, sys1);

% B) Katman Sayısını Modelin İçinden Çek (DÜZELTİLEN KISIM)
% 'NonlinearFcn' yerine 'Nonlinearity' kullanılmalı.
% Cascade-Correlation ağı 'Deep Learning Network' yapısında saklanır.

try
    % Ağ yapısına erişim
    network_obj = sys1.Nonlinearity.Network; 
    tum_katmanlar = network_obj.Layers;
    
    % Giriş ve Çıkış katmanlarını (InputLayer ve RegressionOutputLayer) düşüyoruz
    % Aradaki her şey Cascade-Correlation tarafından eklenen nöronlardır.
    gizli_katman_sayisi = numel(tum_katmanlar) - 2; 
    
    % Bazen (matlab sürümüne göre) aktivasyon katmanları ayrı sayılabilir.
    % En garantisi, isminde 'Hidden' geçenleri veya aktivasyon dışı katmanları saymaktır
    % ama basitçe toplam sayıdan giriş/çıkış düşmek genelde doğru sonuç verir.
    
catch
    % Eğer toolbox sürüm farkından dolayı ağa erişemezse hata vermesin
    gizli_katman_sayisi = NaN; 
    fprintf('Uyarı: Ağ katman sayısına bu sürümde doğrudan erişilemedi.\n');
end

% C) Konsola Yazdır
fprintf('\n==============================================\n');
fprintf('           EĞİTİM SONUÇ RAPORU                \n');
fprintf('==============================================\n');
fprintf('Model Tipi              : Cascade-Correlation NARX\n');
fprintf('Kullanılan Veri Sayısı  : %d örnek\n', size(z1f.y, 1));

if ~isnan(gizli_katman_sayisi)
    fprintf('Toplam Katman (Layers)  : %d \n', numel(tum_katmanlar));
    fprintf('Tahmini Gizli Nöron     : %d adet\n', gizli_katman_sayisi);
else
    fprintf('Gizli Katman Bilgisi    : Otomatik belirlendi (Erişilemedi)\n');
end

fprintf('----------------------------------------------\n');
fprintf('DOĞRULAMA BAŞARISI (Fit): %%%.2f \n', fit_yuzdesi);
fprintf('==============================================\n\n');
%% 4. Grafik Çizimi (Manuel Plot)
t = z2f.SamplingInstants;
gercek_y = z2f.y;
model_sim_y = y_sim.y;

figure('Name', 'Model Performans Grafiği', 'Color', 'white');
plot(t, gercek_y, 'b', 'LineWidth', 1.5); hold on;
plot(t, model_sim_y, 'r--', 'LineWidth', 1.5);
hold off;

% Grafiğin başlığına da başarı oranını ekleyelim
title(['Doğrulama Sonucu - Başarı: %' num2str(fit_yuzdesi, '%.2f')]);
xlabel('Zaman (sn)'); ylabel('Çıkış (Output)');
legend('Gerçek Veri', 'Model Tahmini', 'Location', 'best');
grid on;