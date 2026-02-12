clear; clc; close all;
%% 1. SENİN HAZIRLADIĞIN KISIM (Veri Yükleme ve Filtreleme)
load twotankdata;
z_full = iddata(y, u, 0.2, 'Name', 'Two-tank system');
% --- A) EĞİTİM VERİSİ HAZIRLIĞI ---
z1 = z_full(1:1500);
z1f = idfilt(z1, 3, 0.066902); 
z1f = z1f(20:end);             
% --- B) DOĞRULAMA (VALIDATION) VERİSİ HAZIRLIĞI ---
z2 = z_full(1501:3000);
z2f = idfilt(z2, 3, 0.066902); 
z2f = z2f(20:end);             
%% 2. TRANSFER FONKSİYONU İÇİN KRİTİK ADIM: DETREND
% Transfer fonksiyonları sapmalar (deviation) üzerinden çalışır.
% Bu yüzden verinin ortalamasını 0'a çekmeliyiz (DC bileşeni at).
z1f_d = detrend(z1f); % Eğitim verisinin ortalamasını çıkar
z2f_d = detrend(z2f); % Doğrulama verisinin ortalamasını çıkar
%% 3. MODELİN KESTİRİLMESİ (ESTIMATION)
% Fiziksel Bilgi: 2 Tank var -> Genellikle 2 Kutup (Poles) gerekir.
fprintf('Sistem Modelleniyor (tfest)...\n');
best_fit = -inf;
best_sys = [];
best_order = 0;

% Modelin en iyi uyumunu bulmak için döngü başlat
np=5;
for order = 1:np
    % DÜZELTME 1: Burada 'np' yerine 'order' kullanılmalı.
    % Yoksa np=5 olduğu için 1. dereceden model denerken 4 tane sıfır koymaya çalışır ve hata verir.
    nz = max(0, order - 1);
    
    sys_temp = tfest(z1f_d, order, nz);
    
    % Not: Karşılaştırmayı genelde test verisi (z2f_d) ile yapmak daha sağlıklıdır 
    % ama senin kodunu değiştirmemek için z1f_d bıraktım.
    [~, fit_val, ~] = compare(z1f_d, sys_temp);
    
    % DÜZELTME 2: Ekrana yazarken sabit 'np' değil, değişken 'order' yazılmalı.
    fprintf('Derece: %d -> Fit: %%%.2f\n', order, fit_val);
    
    if fit_val > best_fit
        best_fit = fit_val;
        best_sys = sys_temp;
        % DÜZELTME 3: En iyi dereceyi kaydederken o anki 'order'ı kaydetmeliyiz.
        best_order = order; 
    end
end
% En iyi transfer fonksiyonunu bulduktan sonra, modelin başarı oranını yazdır
fprintf('En iyi uyum: %.2f\n', best_fit);
fprintf('En iyi model derecesi: %d\n', best_order);
% En iyi transfer fonksiyonunu kullanarak modelin başarı oranını yazdır
fprintf('En iyi transfer fonksiyonu:\n');
disp(best_sys);
%% 4. DOĞRULAMA (VALIDATION)
% Modeli, hiç görmediği z2f verisi ile karşılaştırıyoruz.
figure('Name', 'Transfer Fonksiyonu Başarısı', 'Color', 'w');
compare(z2f_d, best_sys); % Grafik çizer ve Fit oranını gösterir
title('Doğrulama Verisi ile Karşılaştırma');
grid on;