%% CCNN - RECURSIVE N-STEP (Trajectory Loss)
% MODEL: N-step
% CANDIDATE: N-step
% (Fully consistent CCNN for system identification)
% 
% FEATURE: Kullanıcı u_lags ve y_lags parametrelerini girer,
%          regresörler otomatik oluşturulur

clear; clc; close all; rng(0);

%% =========================
%% 0) CONFIG
%% =========================
config = struct();

config.data.source = 'twotankdata';
config.data.twotank.filter_cutoff = 0.066902;
config.data.twotank.warmup_samples = 20;
config.data.twotank.sampling_time = 0.2;

config.data.train_ratio = 0.5;
config.data.val_ratio   = 0.5;

config.norm_method = 'ZScore';

config.prediction.n_steps = 480;
config.validation.n_steps = 500;

config.model.max_hidden_units   = 5;
config.model.target_mse         = 5e-5;

config.model.max_epochs_output    = 100;
config.model.eta_output           = 0.02;

config.model.max_epochs_candidate = 100;
config.model.eta_candidate        = 0.02;

config.model.activation = 'tanh';

%% =========================
%% 1) LOAD DATA
%% =========================
[Utr_raw, Ytr_raw, Uva_raw, Yva_raw] = loadDataByConfig_min(config);

%% =========================
%% 2) REGRESÖR PARAMETRELERİNİ TANIMLAYIN
%% =========================
% === BURADAN DEĞİŞTİRİNİZ ===
params.u_lags = [0, 1];      % u(k), u(k-1)
params.y_lags = [1, 2];      % y(k-1), y(k-2)
config.include_bias = false;  % Bias terimi

fprintf('\n=== REGRESÖR TANIMI ===\n');
fprintf('u gecikmeleri: %s\n', mat2str(params.u_lags));
fprintf('y gecikmeleri: %s\n', mat2str(params.y_lags));
fprintf('Toplam regresör sayısı: %d\n', length(params.u_lags) + length(params.y_lags));

% Regresör bilgilerini config'e ekle
config.regressors.u_lags = params.u_lags;
config.regressors.y_lags = params.y_lags;
config.regressors.n_regressors = length(params.u_lags) + length(params.y_lags);
config.regressors.max_lag = max([params.u_lags, params.y_lags]);

%% =========================
%% 3) NORMALIZE
%% =========================
[Utr, Ytr, Uva, Yva, norm_stats] = normalizeData_min( ...
    config.norm_method, Utr_raw, Ytr_raw, Uva_raw, Yva_raw);

%% =========================
%% 4) TRAJECTORY DATASET (DİNAMİK REGRESÖRLERLE)
%% =========================
Npred = config.prediction.n_steps;

[X0_tr, Utr_seq, Ttr_seq] = createTrajectoryDataset(Utr, Ytr, config, Npred);
[X0_va, Uva_seq, Tva_seq] = createTrajectoryDataset(Uva, Yva, config, Npred);

fprintf('\nTrajectory dataset boyutu:\n');
fprintf('X0_tr: %d x %d\n', size(X0_tr,1), size(X0_tr,2));
fprintf('Utr_seq: %d x %d\n', size(Utr_seq,1), size(Utr_seq,2));
fprintf('Ttr_seq: %d x %d\n', size(Ttr_seq,1), size(Ttr_seq,2));

%% =========================
%% 5) ACTIVATION
%% =========================
g = @(x) tanh(x);

%% =========================
%% 6) TRAIN CCNN (N-STEP MODEL + N-STEP CANDIDATE)
%% =========================
fprintf('\n=== CCNN TRAINING (N-STEP MODEL + N-STEP CANDIDATE) ===\n');

W_hidden = {};
d0 = size(X0_tr, 2);  % Regresör boyutu
w_o = randn(d0, 1) * 0.01;

% ---- Stage 1: output only ----
[w_o, current_mse] = trainOutputLayer_Trajectory( ...
    X0_tr, Utr_seq, Ttr_seq, w_o, W_hidden, g, config);

mse_hist = current_mse;
fprintf('Stage-1 Train MSE: %.6g\n', current_mse);

% ---- Stage 2: greedy growing ----
while current_mse > config.model.target_mse && numel(W_hidden) < config.model.max_hidden_units

    h = numel(W_hidden) + 1;

    % === N-STEP CANDIDATE TRAINING ===
    [w_h, cand_mse] = trainCandidateUnit_Trajectory( ...
        X0_tr, Utr_seq, Ttr_seq, W_hidden, w_o, g, config);

    fprintf('Candidate #%d | N-step MSE: %.6g\n', h, cand_mse);

    W_hidden{end+1} = w_h;

    % expand output weights
    w_o = [w_o; randn * 0.01];

    % === RETRAIN OUTPUT (N-STEP) ===
    [w_o, current_mse] = trainOutputLayer_Trajectory( ...
        X0_tr, Utr_seq, Ttr_seq, w_o, W_hidden, g, config);

    mse_hist(end+1) = current_mse;

    % ---- LIVE PLOT ----
    figure(100); clf; set(gcf,'Color','w');
    plot(0:numel(mse_hist)-1, mse_hist, '-o', 'LineWidth', 1.2);
    grid on;
    xlabel('Hidden unit count');
    ylabel('Train MSE (trajectory)');
    title(sprintf('Train MSE | u_lags=%s, y_lags=%s | Hidden=%d | MSE=%.3g', ...
        mat2str(params.u_lags), mat2str(params.y_lags), numel(W_hidden), current_mse));
    drawnow;

    fprintf('Hidden=%d | Train MSE=%.6g\n', numel(W_hidden), current_mse);

    if numel(mse_hist) > 1 && abs(mse_hist(end-1) - mse_hist(end)) < 1e-6
        fprintf('Stop: improvement too small.\n');
        break;
    end
end

%% =========================
%% 7) FULL SERIES VALIDATION (DİNAMİK REGRESÖRLERLE)
%% =========================
Yhat_tr = recursivePredictFullSeries(Utr, Ytr, W_hidden, w_o, g, config);
Yhat_va = recursivePredictFullSeries(Uva, Yva, W_hidden, w_o, g, config);

% Normalizasyonu geri al
max_y_lag = max(params.y_lags);
Yhat_tr_denorm = Yhat_tr(max_y_lag+1:end) * norm_stats.y_std + norm_stats.y_mu;
Yhat_va_denorm = Yhat_va(max_y_lag+1:end) * norm_stats.y_std + norm_stats.y_mu;

% Gerçek değerleri de aynı indeksten başlat
Ytr_raw_trimmed = Ytr_raw(max_y_lag+1:end);
Yva_raw_trimmed = Yva_raw(max_y_lag+1:end);

fit_tr = fitPercent(Ytr_raw_trimmed, Yhat_tr_denorm);
fit_va = fitPercent(Yva_raw_trimmed, Yhat_va_denorm);

fprintf('\n=== SONUÇLAR ===\n');
fprintf('Regresör: u_lags=%s, y_lags=%s\n', mat2str(params.u_lags), mat2str(params.y_lags));
fprintf('Train Fit: %.2f%% | Val Fit: %.2f%%\n', fit_tr, fit_va);

%% =========================
%% PLOTS
%% =========================
figure('Color', 'w', 'Name', 'TRAIN - Full Recursive');
plot(Ytr_raw_trimmed, 'k', 'LineWidth', 1.4); hold on;
plot(Yhat_tr_denorm, 'b--', 'LineWidth', 1.2);
grid on;
title(sprintf('TRAIN | u_lags=%s, y_lags=%s | Hidden=%d | Fit=%.2f%%', ...
    mat2str(params.u_lags), mat2str(params.y_lags), numel(W_hidden), fit_tr));
legend('True', 'CCNN', 'Location', 'best');
xlabel('Zaman (örnek)');
ylabel('Çıkış');

figure('Color', 'w', 'Name', 'VAL - Full Recursive');
plot(Yva_raw_trimmed, 'k', 'LineWidth', 1.4); hold on;
plot(Yhat_va_denorm, 'r--', 'LineWidth', 1.2);
grid on;
title(sprintf('VAL | u_lags=%s, y_lags=%s | Hidden=%d | Fit=%.2f%%', ...
    mat2str(params.u_lags), mat2str(params.y_lags), numel(W_hidden), fit_va));
legend('True', 'CCNN', 'Location', 'best');
xlabel('Zaman (örnek)');
ylabel('Çıkış');

%% =====================================================================
%% ======================= LOCAL FUNCTIONS ==============================
%% =====================================================================

function [w_h, mse] = trainCandidateUnit_Trajectory( ...
    X0, U, T, W_hidden, w_o, g, config)

    % Input boyutu: regresör boyutu + mevcut hidden unit sayısı
    n_regressors = config.regressors.n_regressors;
    d = n_regressors + numel(W_hidden);
    w_h = dlarray(randn(d, 1) * 0.01);

    X0 = dlarray(X0);
    U = dlarray(U);
    T = dlarray(T);
    w_o = dlarray(w_o);

    avgG = []; avgGSq = []; it = 0;

    for ep = 1:config.model.max_epochs_candidate
        it = it + 1;
        [L, grad] = dlfeval(@loss_candidate_traj, ...
            w_h, X0, U, T, W_hidden, w_o, g, config);
        [w_h, avgG, avgGSq] = adamupdate(w_h, grad, avgG, avgGSq, it, config.model.eta_candidate);
    end

    w_h = extractdata(w_h);

    Wtmp = [W_hidden, {w_h}];
    wtmp = [extractdata(w_o); 0.1];
    Y = forwardCCNN_recursiveTrajectory(X0, U, Wtmp, g, wtmp, config);
    mse = mean((T - Y).^2, 'all');
end

function [L, grad] = loss_candidate_traj(w_h, X0, U, T, W_hidden, w_o, g, config)
    M = size(X0, 1);
    N = size(U, 2);
    n_u_lags = length(config.regressors.u_lags);
    
    % İlk y değerini X0'dan al (son sütun)
    yprev = X0(:, end);
    Y = dlarray(zeros(M, N));

    for t = 1:N
        % Regresör vektörünü oluştur
        x = buildRegressorVector(X0(1, :), U(:, t), yprev, t, config);
        
        % Hidden layerları ekle
        for h = 1:numel(W_hidden)
            hidden_out = g(x * W_hidden{h});
            x = [x, hidden_out];
        end
        
        % Yeni candidate unit ekle
        v = g(x * w_h);
        
        % Output hesapla
        x_with_candidate = [x, v];
        w_extended = [w_o; 0.1];
        y = x_with_candidate * w_extended;
        
        Y(:, t) = y;
        yprev = y;
    end

    L = mean((T - Y).^2, 'all');
    grad = dlgradient(L, w_h);
end

function [w_o, mse] = trainOutputLayer_Trajectory( ...
    X0, U, T, w_o, W_hidden, g, config)

    w_o = dlarray(w_o);
    X0 = dlarray(X0);
    U = dlarray(U);
    T = dlarray(T);
    avgG = []; avgGSq = []; it = 0;

    for ep = 1:config.model.max_epochs_output
        it = it + 1;
        [L, grad] = dlfeval(@loss_output_traj, w_o, X0, U, T, W_hidden, g, config);
        [w_o, avgG, avgGSq] = adamupdate(w_o, grad, avgG, avgGSq, it, config.model.eta_output);
    end

    w_o = extractdata(w_o);
    Y = forwardCCNN_recursiveTrajectory(X0, U, W_hidden, g, w_o, config);
    mse = mean((T - Y).^2, 'all');
end

function [L, grad] = loss_output_traj(w, X0, U, T, W_hidden, g, config)
    Y = forwardCCNN_recursiveTrajectory(X0, U, W_hidden, g, w, config);
    L = mean((T - Y).^2, 'all');
    grad = dlgradient(L, w);
end

function Y = forwardCCNN_recursiveTrajectory(X0, U, W_hidden, g, w, config)
    M = size(X0, 1);
    N = size(U, 2);
    n_u_lags = length(config.regressors.u_lags);
    
    % İlk y değerini X0'dan al (son sütun)
    yprev = X0(:, end);
    Y = zeros(M, N, 'like', X0);

    for t = 1:N
        % Regresör vektörünü oluştur
        x = buildRegressorVector(X0(1, :), U(:, t), yprev, t, config);
        
        % Hidden layerları ekle
        for h = 1:numel(W_hidden)
            hidden_out = g(x * W_hidden{h});
            x = [x, hidden_out];
        end
        
        % Output hesapla: x (M x d) * w (d x 1) = y (M x 1)
        y = x * w;
        
        Y(:, t) = y;
        yprev = y;
    end
end

function x = buildRegressorVector(x0_row, u_current, y_prev, timestep, config)
% Dinamik regresör vektörü oluştur
% x0_row: 1 x n_regressors (ilk durumdan gelen regresörler)
% u_current: M x 1 (mevcut input)
% y_prev: M x 1 (önceki output)
% timestep: scalar (hangi zaman adımındayız)

    u_lags = config.regressors.u_lags;
    y_lags = config.regressors.y_lags;
    M = size(u_current, 1);
    
    % Regresör vektörünü başlat
    x = [];
    
    % U regresörlerini ekle
    for i = 1:length(u_lags)
        lag = u_lags(i);
        if lag == 0
            % u(k): mevcut input
            x = [x, u_current];  % M x 1
        else
            % u(k-lag): X0'dan al
            x = [x, repmat(x0_row(i), M, 1)];  % M x 1
        end
    end
    
    % Y regresörlerini ekle
    for i = 1:length(y_lags)
        lag = y_lags(i);
        if timestep == 1
            % İlk adımda X0'dan al
            idx = length(u_lags) + i;
            x = [x, repmat(x0_row(idx), M, 1)];  % M x 1
        else
            % Sonraki adımlarda önceki tahmini kullan
            x = [x, y_prev];  % M x 1
        end
    end
end

function [Utr, Ytr, Uva, Yva] = loadDataByConfig_min(config)
% Minimal data loader (no toolbox dependency)

switch lower(config.data.source)

    case 'twotankdata'
        load twotankdata.mat  % must contain u, y

        u = u(:);
        y = y(:);

        % warmup
        w = config.data.twotank.warmup_samples;
        u = u(w+1:end);
        y = y(w+1:end);

    otherwise
        error('Unknown data source');
end

N = length(u);
Ntr = floor(config.data.train_ratio * N);

Utr = u(1:Ntr);
Ytr = y(1:Ntr);

Uva = u(Ntr+1:end);
Yva = y(Ntr+1:end);
end

function [Utr, Ytr, Uva, Yva, stats] = normalizeData_min(method, Utr, Ytr, Uva, Yva)

switch lower(method)
    case 'zscore'
        stats.u_mu = mean(Utr);
        stats.u_std = std(Utr) + eps;
        stats.y_mu = mean(Ytr);
        stats.y_std = std(Ytr) + eps;

        Utr = (Utr - stats.u_mu) / stats.u_std;
        Uva = (Uva - stats.u_mu) / stats.u_std;

        Ytr = (Ytr - stats.y_mu) / stats.y_std;
        Yva = (Yva - stats.y_mu) / stats.y_std;

    otherwise
        error('Unknown normalization');
end
end

%% ==================== DİNAMİK REGRESÖR FONKSİYONLARI ====================

function [X0, Useq, Tseq] = createTrajectoryDataset(U, Y, config, N)
% Dinamik regresör tanımına göre trajectory dataset oluştur

u_lags = config.regressors.u_lags;
y_lags = config.regressors.y_lags;
n_regressors = config.regressors.n_regressors;

max_lag = max([u_lags, y_lags]);
start_idx = max_lag + 1;
Ns = length(Y) - N - max_lag;

% Hata kontrolü
if Ns <= 0
    error('Yeterli veri yok. Npred çok büyük veya max_lag çok büyük.');
end

X0 = zeros(Ns, n_regressors);
Useq = zeros(Ns, N);
Tseq = zeros(Ns, N);

for i = 1:Ns
    idx = start_idx + i - 1;
    
    % Regresör vektörünü oluştur
    reg_vec = [];
    
    % u laglarını ekle
    for lag = u_lags
        reg_vec = [reg_vec, U(idx - lag)];
    end
    
    % y laglarını ekle
    for lag = y_lags
        reg_vec = [reg_vec, Y(idx - lag)];
    end
    
    X0(i, :) = reg_vec;
    Useq(i, :) = U(idx+1:idx+N)';
    Tseq(i, :) = Y(idx+1:idx+N)';
end
end

function Yhat = recursivePredictFullSeries(U, Y, W_hidden, w_o, g, config)
% Dinamik regresör tanımına göre tam seri tahmini

u_lags = config.regressors.u_lags;
y_lags = config.regressors.y_lags;

N = length(Y);
Yhat = zeros(N, 1);
max_y_lag = max(y_lags);

% Başlangıç değerlerini gerçek değerlerle ata
for i = 1:max_y_lag
    Yhat(i) = Y(i);
end

for k = max_y_lag+1:N
    % Regresör vektörünü oluştur - SATIR VEKTÖRÜ (1 x n_regressors)
    x = [];
    
    % u laglarını ekle
    for lag = u_lags
        idx_u = k - lag;
        if idx_u >= 1 && idx_u <= length(U)
            x = [x, U(idx_u)];
        else
            x = [x, 0];
        end
    end
    
    % y laglarını ekle (tahmin edilen değerleri kullan)
    for lag = y_lags
        idx_y = k - lag;
        if idx_y >= 1 && idx_y <= length(Yhat)
            x = [x, Yhat(idx_y)];
        else
            x = [x, 0];
        end
    end
    
    x = x(:)';  % 1 x n_regressors
    
    % Hidden layer'ları ekle
    for h = 1:numel(W_hidden)
        % x: 1 x d, W_hidden{h}: d x 1 => x * W_hidden{h}: 1 x 1
        hidden_out = g(x * W_hidden{h});
        x = [x, hidden_out];
    end
    
    % Çıkışı hesapla - x: 1 x (n_regressors + n_hidden), w_o: (n_regressors + n_hidden) x 1
    Yhat(k) = x * w_o;
end
end

function fit = fitPercent(y, yhat)
% Fit yüzdesini hesapla
fit = 100 * (1 - norm(y - yhat) / norm(y - mean(y)));
end