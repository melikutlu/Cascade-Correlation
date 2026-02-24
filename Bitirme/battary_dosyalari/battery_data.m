% battery_prepare_ccnn.m
% Bu script batarya verisini CCNN formatına dönüştürür

clear; clc; close all;

fprintf('=== BATARYA VERİSİ CCNN FORMATINA DÖNÜŞTÜRÜLÜYOR ===\n\n');

%% 1. ORİJİNAL VERİLERİ YÜKLE
fprintf('1. Orijinal veriler yükleniyor...\n');

try
    % FTP75 - EĞİTİM VERİSİ
    data_train = load('FTP75.mat');
    fprintf('   FTP75.mat yüklendi\n');
    
    % Alan isimlerini kontrol et
    train_fields = fieldnames(data_train);
    fprintf('   Alanlar: %s\n', strjoin(train_fields, ', '));
    
    % LA92 - DOĞRULAMA VERİSİ
    data_val = load('LA92.mat');
    fprintf('   LA92.mat yüklendi\n');
    
    val_fields = fieldnames(data_val);
    fprintf('   Alanlar: %s\n', strjoin(val_fields, ', '));
    
catch err
    fprintf('HATA: Dosya yüklenemedi: %s\n', err.message);
    fprintf('Dosyaların mevcut dizinde olduğundan emin ol:\n');
    fprintf('   - FTP75.mat\n');
    fprintf('   - LA92.mat\n');
    return;
end

%% 2. DEĞİŞKEN İSİMLERİNİ BELİRLE (Esnek)
fprintf('\n2. Değişken isimleri belirleniyor...\n');

% Eğitim verisi için
if isfield(data_train, 'curr') && isfield(data_train, 'soc') && isfield(data_train, 'temp')
    % İsim 1: curr, soc, temp
    u1_train = data_train.curr;
    u2_train = data_train.soc;
    y_train = data_train.temp;
    fprintf('   Eğitim: curr, soc, temp kullanılıyor\n');
    
elseif isfield(data_train, 'u_curr') && isfield(data_train, 'u_soc') && isfield(data_train, 'y_temp')
    % İsim 2: u_curr, u_soc, y_temp
    u1_train = data_train.u_curr;
    u2_train = data_train.u_soc;
    y_train = data_train.y_temp;
    fprintf('   Eğitim: u_curr, u_soc, y_temp kullanılıyor\n');
    
elseif isfield(data_train, 'datae') && isfield(data_train.datae, 'curr')
    % İç içe struct
    u1_train = data_train.datae.curr;
    u2_train = data_train.datae.soc;
    y_train = data_train.datae.temp;
    fprintf('   Eğitim: datae struct''ından alınıyor\n');
    
else
    % İlk 3 sütunu kullan
    all_data = struct2array(data_train);
    if size(all_data, 2) >= 3
        u1_train = all_data(:, 1);
        u2_train = all_data(:, 2);
        y_train = all_data(:, 3);
        fprintf('   Eğitim: İlk 3 sütun kullanılıyor\n');
    else
        error('Eğitim verisinde yeterli sütun yok');
    end
end

% Doğrulama verisi için
if isfield(data_val, 'curr') && isfield(data_val, 'soc') && isfield(data_val, 'temp')
    u1_val = data_val.curr;
    u2_val = data_val.soc;
    y_val = data_val.temp;
    fprintf('   Doğrulama: curr, soc, temp kullanılıyor\n');
    
elseif isfield(data_val, 'u_curr') && isfield(data_val, 'u_soc') && isfield(data_val, 'y_temp')
    u1_val = data_val.u_curr;
    u2_val = data_val.u_soc;
    y_val = data_val.y_temp;
    fprintf('   Doğrulama: u_curr, u_soc, y_temp kullanılıyor\n');
    
elseif isfield(data_val, 'datav') && isfield(data_val.datav, 'curr')
    u1_val = data_val.datav.curr;
    u2_val = data_val.datav.soc;
    y_val = data_val.datav.temp;
    fprintf('   Doğrulama: datav struct''ından alınıyor\n');
    
else
    all_data = struct2array(data_val);
    if size(all_data, 2) >= 3
        u1_val = all_data(:, 1);
        u2_val = all_data(:, 2);
        y_val = all_data(:, 3);
        fprintf('   Doğrulama: İlk 3 sütun kullanılıyor\n');
    else
        error('Doğrulama verisinde yeterli sütun yok');
    end
end

%% 3. VERİYİ BİRLEŞTİR (CCNN FORMATI)
fprintf('\n3. CCNN formatına dönüştürülüyor...\n');

% CCNN formatı: u = [giriş1, giriş2], y = çıkış
u_train = [u1_train, u2_train];
u_val = [u1_val, u2_val];

% Boyutları kontrol et
fprintf('   Eğitim: u = %dx%d, y = %dx%d\n', ...
    size(u_train, 1), size(u_train, 2), size(y_train, 1), size(y_train, 2));
fprintf('   Doğrulama: u = %dx%d, y = %dx%d\n', ...
    size(u_val, 1), size(u_val, 2), size(y_val, 1), size(y_val, 2));

% Örnek sayısını sınırla (isteğe bağlı)
max_samples = 5000;  % Daha hızlı eğitim için
if size(u_train, 1) > max_samples
    fprintf('   NOT: Eğitim verisi %d örneğe indiriliyor\n', max_samples);
    u_train = u_train(1:max_samples, :);
    y_train = y_train(1:max_samples, :);
end

if size(u_val, 1) > max_samples
    fprintf('   NOT: Doğrulama verisi %d örneğe indiriliyor\n', max_samples);
    u_val = u_val(1:max_samples, :);
    y_val = y_val(1:max_samples, :);
end

%% 4. VERİYİ GÖRSELLEŞTİR (İsteğe bağlı)
fprintf('\n4. Veri görselleştiriliyor...\n');

figure('Name', 'Batarya Verisi - Eğitim (FTP75)', 'Position', [100 100 1200 800]);

% Giriş 1: Akım
subplot(3,2,1);
plot(u_train(:,1), 'b-', 'LineWidth', 1);
title('Eğitim - Akım (curr)');
xlabel('Örnek'); ylabel('Akım');
grid on;

% Giriş 2: SOC
subplot(3,2,2);
plot(u_train(:,2), 'g-', 'LineWidth', 1);
title('Eğitim - SOC');
xlabel('Örnek'); ylabel('SOC (%)');
grid on;

% Çıkış: Sıcaklık
subplot(3,2,3);
plot(y_train, 'r-', 'LineWidth', 1);
title('Eğitim - Sıcaklık (temp)');
xlabel('Örnek'); ylabel('Sıcaklık');
grid on;

% Doğrulama verisi
subplot(3,2,4);
plot(u_val(:,1), 'b-', 'LineWidth', 1);
title('Doğrulama - Akım (curr)');
xlabel('Örnek'); ylabel('Akım');
grid on;

subplot(3,2,5);
plot(u_val(:,2), 'g-', 'LineWidth', 1);
title('Doğrulama - SOC');
xlabel('Örnek'); ylabel('SOC (%)');
grid on;

subplot(3,2,6);
plot(y_val, 'r-', 'LineWidth', 1);
title('Doğrulama - Sıcaklık (temp)');
xlabel('Örnek'); ylabel('Sıcaklık');
grid on;

%% 5. İSTATİSTİKLERİ HESAPLA
fprintf('\n5. Veri istatistikleri:\n');
fprintf('   EĞİTİM (FTP75):\n');
fprintf('     Akım:    min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
    min(u_train(:,1)), max(u_train(:,1)), mean(u_train(:,1)), std(u_train(:,1)));
fprintf('     SOC:     min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
    min(u_train(:,2)), max(u_train(:,2)), mean(u_train(:,2)), std(u_train(:,2)));
fprintf('     Sıcaklık: min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
    min(y_train), max(y_train), mean(y_train), std(y_train));

fprintf('\n   DOĞRULAMA (LA92):\n');
fprintf('     Akım:    min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
    min(u_val(:,1)), max(u_val(:,1)), mean(u_val(:,1)), std(u_val(:,1)));
fprintf('     SOC:     min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
    min(u_val(:,2)), max(u_val(:,2)), mean(u_val(:,2)), std(u_val(:,2)));
fprintf('     Sıcaklık: min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
    min(y_val), max(y_val), mean(y_val), std(y_val));

%% 6. CCNN İÇİN KAYDET
fprintf('\n6. CCNN formatında kaydediliyor...\n');

% Tüm veriyi bir struct'ta topla
battery_data_ccnn = struct();
battery_data_ccnn.u_train = u_train;
battery_data_ccnn.y_train = y_train;
battery_data_ccnn.u_val = u_val;
battery_data_ccnn.y_val = y_val;
battery_data_ccnn.description = 'Batarya sıcaklık tahmini - FTP75 (train) ve LA92 (val)';
battery_data_ccnn.created_date = datestr(now);
battery_data_ccnn.original_files = {'FTP75.mat', 'LA92.mat'};

% Kaydet
save('battery_data_ccnn.mat', '-struct', 'battery_data_ccnn');
fprintf('   battery_data_ccnn.mat kaydedildi\n');

% Alternatif: Ayrı değişkenler olarak da kaydedebiliriz
save('battery_for_ccnn.mat', 'u_train', 'y_train', 'u_val', 'y_val');
fprintf('   battery_for_ccnn.mat kaydedildi\n');

%% 7. KULLANIM İÇİN KONFİG ÖRNEĞİ
fprintf('\n7. CCNN KONFİG ÖRNEĞİ:\n');
fprintf('========================================\n');
fprintf('%% CCNN config ayarları:\n');
fprintf('config.data.source = ''mat'';\n');
fprintf('config.data.filepath = ''battery_for_ccnn.mat'';\n');
fprintf('config.data.input_columns = [1, 2];  %% curr ve soc\n');
fprintf('config.data.output_columns = 3;      %% temp\n');
fprintf('config.model.num_inputs = 2;\n');
fprintf('config.model.num_outputs = 1;\n');
fprintf('config.regressors.na = 5;  %% 5 sıcaklık gecikmesi\n');
fprintf('config.regressors.nb = 3;  %% 3 giriş gecikmesi\n');
fprintf('config.regressors.nk = 1;  %% 1 adım gecikme\n');
fprintf('========================================\n');

fprintf('\n✅ BATARYA VERİSİ BAŞARIYLA HAZIRLANDI!\n');
fprintf('   Şimdi CCNN kodunu çalıştırabilirsin.\n');