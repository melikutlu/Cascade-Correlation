%% CCNN - RECURSIVE N-STEP (Trajectory Loss) TRAIN + VALIDATION (Sade)
clear; clc; close all; rng(0);

%% 0) CONFIG
config = struct();

% --- Data (twotankdata) ---
config.data.source = 'twotankdata';
config.data.twotank.filter_cutoff = 0.066902;
config.data.twotank.warmup_samples = 20;
config.data.twotank.sampling_time = 0.2;

config.data.train_ratio = 0.5;
config.data.val_ratio   = 0.5;

% --- Normalization ---
config.norm_method = 'ZScore'; % 'ZScore', 'MinMax', 'None'

% --- Recursive N-step ---
config.prediction.n_steps = 20;   % N
config.validation.n_steps = 1478;

% --- Regressors (SISO): model uses u(k) and y(k-1)
config.regressors.include_bias = false;

% --- CCNN hyperparams ---
config.model.max_hidden_units   = 10;
config.model.target_mse         = 5e-5;

config.model.max_epochs_output  = 1000;
config.model.eta_output_gd      = 0.002;

config.model.max_epochs_candidate = 1000;
config.model.eta_candidate        = 0.002;

config.model.activation = 'tanh'; % 'tanh', 'sigmoid', 'relu'

%% 1) LOAD DATA
fprintf('\n=== LOAD DATA ===\n');
[U_train_raw, Y_train_raw, U_val_raw, Y_val_raw] = loadDataByConfig_min(config);
fprintf('Train: %d samples | Val: %d samples\n', size(U_train_raw,1), size(U_val_raw,1));

%% 2) NORMALIZE (train stats)
fprintf('\n=== NORMALIZE (%s) ===\n', config.norm_method);
[U_train, Y_train, U_val, Y_val, norm_stats] = normalizeData_min( ...
    config.norm_method, U_train_raw, Y_train_raw, U_val_raw, Y_val_raw);

%% 3) BUILD TRAJECTORY DATASET (for recursive rollout training)
Npred = config.prediction.n_steps;
fprintf('\n=== BUILD TRAJECTORY DATASET (Recursive N=%d) ===\n', Npred);

[X0_train, Utr_seq, Ttr_seq, reg_train] = createTrajectoryDataset(U_train, Y_train, config, Npred);
[X0_val,   Uval_seq, Tval_seq, reg_val] = createTrajectoryDataset(U_val,   Y_val,   config, Npred);

fprintf('Train: X0=%dx%d | U_seq=%dx%d | T_seq=%dx%d\n', ...
    size(X0_train,1), size(X0_train,2), size(Utr_seq,1), size(Utr_seq,2), size(Ttr_seq,1), size(Ttr_seq,2));
fprintf('Val  : X0=%dx%d | U_seq=%dx%d | T_seq=%dx%d\n', ...
    size(X0_val,1), size(X0_val,2), size(Uval_seq,1), size(Uval_seq,2), size(Tval_seq,1), size(Tval_seq,2));

%% 4) ACTIVATION
switch lower(config.model.activation)
    case 'tanh'
        g = @(a) tanh(a);
        g_prime = @(v) 1 - v.^2;
    case 'sigmoid'
        g = @(a) 1./(1 + exp(-a));
        g_prime = @(v) v .* (1 - v);
    case 'relu'
        g = @(a) max(0, a);
        g_prime = @(v) double(v > 0);
    otherwise
        error('Unknown activation: %s', config.model.activation);
end

%% 5) TRAIN CCNN (Recursive N-step Trajectory Loss)
fprintf('\n=== TRAIN CCNN (Recursive N-step, Trajectory Loss) ===\n');

W_hidden = {};                         % start no hidden
X_candidate_input = X0_train;          % candidate-training inputs (step-1 features only)

d0 = size(X0_train,2);                 % base feature size: [u(k), y(k-1)] (+bias)
w_o_initial = randn(d0, 1) * 0.01;     % output weights for stage-1

% ---- Stage 1: train output only (no hidden) using trajectory loss ----
[w_o_stage1, E_residual, current_mse] = trainOutputLayer_GD_Autograd_Trajectory( ...
    X0_train, Utr_seq, Ttr_seq, w_o_initial, W_hidden, g, config, ...
    config.model.max_epochs_output, config.model.eta_output_gd);

mse_history = current_mse;
fprintf('Stage-1 Train MSE: %.6g\n', current_mse);

w_o = w_o_stage1;

% ---- Stage 2: add hidden units one by one ----
while current_mse > config.model.target_mse && numel(W_hidden) < config.model.max_hidden_units
    h = numel(W_hidden) + 1;
    
    % GÜNCELLEME: Candidate unit'i N-step trajectory loss ile eğit
    [w_h, v_h_train, current_mse_candidate] = trainCandidateUnit_Trajectory( ...
        X0_train, Utr_seq, Ttr_seq, W_hidden, g, config, ...
        config.model.max_epochs_candidate, config.model.eta_candidate);
    
    % Debug için candidate MSE'yi yazdır
    fprintf('Candidate #%d trained | Candidate MSE: %.6g\n', h, current_mse_candidate);
    
    W_hidden{h} = w_h;

    % Expand candidate input for next hidden unit
    X_candidate_input = [X_candidate_input, v_h_train];

    % Retrain output layer with updated hidden set (trajectory loss)
    w_o_init_new = [w_o; randn(1,1)*0.01];  % add one weight for new hidden feature
    [w_o, E_residual, current_mse] = trainOutputLayer_GD_Autograd_Trajectory( ...
        X0_train, Utr_seq, Ttr_seq, w_o_init_new, W_hidden, g, config, ...
        config.model.max_epochs_output, config.model.eta_output_gd);

    mse_history(end+1) = current_mse;

    % Plot training MSE vs hidden count
    figure(100); clf; set(gcf,'Color','w','Name','Train MSE vs Hidden Units');
    plot(0:(numel(mse_history)-1), mse_history, '-o', 'LineWidth', 1.2);
    grid on;
    xlabel('Hidden unit sayısı');
    ylabel('Train MSE (trajectory)');
    title(sprintf('Train MSE history | Hidden=%d | Current MSE=%.6g', numel(W_hidden), current_mse));
    drawnow;

    fprintf('Hidden #%d added | Train MSE: %.6g\n', h, current_mse);

    if numel(mse_history) >= 2
        if (mse_history(end-1) - mse_history(end)) < 1e-6
            fprintf('Stop: MSE improvement too small.\n');
            break;
        end
    end
end

fprintf('Total hidden units: %d\n', numel(W_hidden));
fprintf('Final Train MSE: %.6g\n', current_mse);

%% 6) VALIDATION - RECURSIVE PREDICTION (İlk girişten başlayarak tüm seri)
fprintf('\n=== VALIDATION - RECURSIVE PREDICTION (Full Series) ===\n');

% Başlangıç değerleri: val setinin ilk değerleri
if config.regressors.include_bias
    y_init = Y_val(1);  % y(k-1) için ilk değer
else
    y_init = Y_val(1);  % y(k-1) için ilk değer
end

% Tüm val setini tahmin et (recursive)
Yhat_val_recursive = recursivePredictFullSeries(U_val, y_init, W_hidden, w_o, g, config);

% Denormalize et
Yhat_val_real = denormalizeData_min(Yhat_val_recursive, config.norm_method, norm_stats.y);

% Fit hesapla
fit_val_full = fitPercent(Y_val_raw, Yhat_val_real);
fprintf('Full Series Validation Fit: %.2f%%\n', fit_val_full);

% ===== PLOT: VAL (Full Series) =====
figure('Color','w','Name','VAL - Full Series Recursive Prediction');
plot(Y_val_raw(:,1), 'k', 'LineWidth', 1.5); hold on;
plot(Yhat_val_real, 'r--', 'LineWidth', 1.2);
grid on;
xlabel('Time step (k)');
ylabel('y (real units)');
title(sprintf('VAL - Full Series Recursive Prediction | Hidden=%d | Fit=%.2f%%', ...
    numel(W_hidden), fit_val_full));
legend('True', 'CCNN Prediction', 'Location', 'best');

%% =========================
%% Local helper functions
%% =========================

function [U_train, Y_train, U_val, Y_val] = loadDataByConfig_min(config)
    switch lower(config.data.source)
        case 'twotankdata'
            load dryer2; % provides u, y

            z_full = iddata(y2, u2, config.data.twotank.sampling_time);
        
        % Veriyi böl
        N_total = length(z_full.y);
        train_end = floor(N_total * config.data.train_ratio);
        val_end = train_end + floor(N_total * config.data.val_ratio);
                

            % Train
            z1 = z_full(1:train_end);
            z1f = idfilt(z1, 3, config.data.twotank.filter_cutoff);
            z1f = z1f(config.data.twotank.warmup_samples:end);
            U_train = z1f.u;
            Y_train = z1f.y;

            % Val
            z2 = z_full(train_end+1:val_end);
            z2f = idfilt(z2, 3, config.data.twotank.filter_cutoff);
            z2f = z2f(config.data.twotank.warmup_samples:end);
            U_val = z2f.u;
            Y_val = z2f.y;

        otherwise
            error('This minimal script supports only twotankdata for now.');
    end
end

function [Utr_n, Ytr_n, Uva_n, Yva_n, stats] = normalizeData_min(method, Utr, Ytr, Uva, Yva)
    stats = struct();
    switch lower(method)
        case 'zscore'
            stats.u.mu = mean(Utr,1);
            stats.u.sg = std(Utr,0,1); stats.u.sg(stats.u.sg==0)=1;

            stats.y.mu = mean(Ytr,1);
            stats.y.sg = std(Ytr,0,1); stats.y.sg(stats.y.sg==0)=1;

            Utr_n = (Utr - stats.u.mu) ./ stats.u.sg;
            Ytr_n = (Ytr - stats.y.mu) ./ stats.y.sg;
            Uva_n = (Uva - stats.u.mu) ./ stats.u.sg;
            Yva_n = (Yva - stats.y.mu) ./ stats.y.sg;

        case 'minmax'
            stats.u.mn = min(Utr,[],1); stats.u.mx = max(Utr,[],1);
            du = stats.u.mx - stats.u.mn; du(du==0)=1;

            stats.y.mn = min(Ytr,[],1); stats.y.mx = max(Ytr,[],1);
            dy = stats.y.mx - stats.y.mn; dy(dy==0)=1;

            Utr_n = (Utr - stats.u.mn) ./ du;
            Ytr_n = (Ytr - stats.y.mn) ./ dy;
            Uva_n = (Uva - stats.u.mn) ./ du;
            Yva_n = (Yva - stats.y.mn) ./ dy;

        case 'none'
            Utr_n = Utr; Ytr_n = Ytr; Uva_n = Uva; Yva_n = Yva;
            stats.u = struct(); stats.y = struct();

        otherwise
            error('Unknown norm_method: %s', method);
    end
end

function X = denormalizeData_min(Xn, method, st)
    switch lower(method)
        case 'zscore'
            X = Xn .* st.sg + st.mu;
        case 'minmax'
            d = st.mx - st.mn; d(d==0)=1;
            X = Xn .* d + st.mn;
        case 'none'
            X = Xn;
        otherwise
            error('Unknown norm_method: %s', method);
    end
end

function fitp = fitPercent(ytrue, yhat)
    fitp = (1 - norm(ytrue - yhat) / norm(ytrue - mean(ytrue))) * 100;
end

function [X0_bias, U_seq, T_seq, reg] = createTrajectoryDataset(U, Y, config, Npred)
% X0 has [u(k), y(k-1)] (+bias if enabled)
% U_seq is [u(k)...u(k+Npred-1)]
% T_seq is [y(k)...y(k+Npred-1)]

    N = size(U,1);
    max_lag = 1;                 % because we use y(k-1)

    start_idx = max_lag + 1;
    end_idx   = N - Npred + 1;

    if end_idx < start_idx
        error('Npred too large for available data length.');
    end

    M = end_idx - start_idx + 1;

    U_seq = zeros(M, Npred);
    T_seq = zeros(M, Npred);

    if config.regressors.include_bias
        X0_bias = zeros(M, 3); % [1, u(k), y(k-1)]
    else
        X0_bias = zeros(M, 2); % [u(k), y(k-1)]
    end

    for i = 1:M
        k = start_idx + i - 1;

        u0    = U(k,1);
        yprev = Y(k-1,1);

        if config.regressors.include_bias
            X0_bias(i,:) = [1, u0, yprev];
        else
            X0_bias(i,:) = [u0, yprev];
        end

        U_seq(i,:) = U(k:(k+Npred-1),1).';
        T_seq(i,:) = Y(k:(k+Npred-1),1).';
    end

    reg.start_idx = start_idx;
    reg.end_idx   = end_idx;
    reg.max_lag   = max_lag;
    reg.idx_target = (start_idx + (Npred-1)) : (end_idx + (Npred-1));
end

function Yhat_seq = forwardCCNN_recursiveTrajectory(X0_bias, U_seq, W_hidden, g, w_o, config)
    M     = size(X0_bias,1);
    Npred = size(U_seq,2);
    num_hidden = numel(W_hidden);

    Yhat_seq = zeros(M, Npred);

    if config.regressors.include_bias
        y_prev = X0_bias(:,3);
    else
        y_prev = X0_bias(:,2);
    end

    for t = 1:Npred
        u_curr = U_seq(:,t);

        if config.regressors.include_bias
            x = [ones(M,1), u_curr, y_prev];
        else
            x = [u_curr, y_prev];
        end

        x_aug = x;
        for h = 1:num_hidden
            v = g(x_aug * W_hidden{h});
            x_aug = [x_aug, v];
        end

        yhat = x_aug * w_o;
        Yhat_seq(:,t) = yhat;
        y_prev = yhat;
    end
end

function [w_h, v_h, current_mse] = trainCandidateUnit_Trajectory( ...
    X0_bias, U_seq, T_seq, W_hidden, g, config, max_epochs, eta)
% GÜNCELLEME: Candidate unit'i N-step trajectory loss ile eğit
    
    [M, d0] = size(X0_bias);
    num_hidden = numel(W_hidden);
    
    % Candidate unit için rastgele ağırlık başlat
    % Boyut: temel özellikler + mevcut hidden unit sayısı
    w_h = randn(d0 + num_hidden, 1) * 0.01;
    
    % Adam optimizer parametreleri
    averageGrad = [];
    averageSqGrad = [];
    iteration = 0;
    gradDecay = 0.9;
    sqGradDecay = 0.999;
    eps_adam = 1e-8;
    
    % Convert to dlarray for autodiff
    X0_dl = dlarray(X0_bias);
    U_dl = dlarray(U_seq);
    T_dl = dlarray(T_seq);
    w_h_dl = dlarray(w_h);
    
    for epoch = 1:max_epochs
        iteration = iteration + 1;
        
        [loss, grad_w] = dlfeval(@loss_and_grad_candidate_traj, ...
            w_h_dl, X0_dl, U_dl, T_dl, W_hidden, g, config);
        
        % Adam update
        [w_h_dl, averageGrad, averageSqGrad] = adamupdate(w_h_dl, grad_w, ...
            averageGrad, averageSqGrad, iteration, ...
            eta, gradDecay, sqGradDecay, eps_adam);
    end
    
    w_h = extractdata(w_h_dl);
    
    % Candidate unit'ın çıktısını hesapla
    v_h = zeros(M, 1);
    if config.regressors.include_bias
        y_prev = X0_bias(:,3);
    else
        y_prev = X0_bias(:,2);
    end
    
    % Sadece ilk adım için candidate unit çıktısı
    u_curr = U_seq(:,1);
    if config.regressors.include_bias
        x = [ones(M,1), u_curr, y_prev];
    else
        x = [u_curr, y_prev];
    end
    
    x_aug = x;
    for h = 1:num_hidden
        v = g(x_aug * W_hidden{h});
        x_aug = [x_aug, v];
    end
    
    v_h = g(x_aug * w_h);
    
    % MSE'yi hesapla (sadece bilgi için)
    % ÖNEMLİ DÜZELTME: Candidate unit dahil edilmiş halde tahmin yap
    % Geçici olarak candidate unit'i mevcut hidden unit'lere ekle
    W_hidden_temp = W_hidden;
    W_hidden_temp{end+1} = w_h;
    
    % Çıkış ağırlıklarını güncelle: tüm ağırlıklar 0, sadece candidate unit için 1
    w_o_temp = zeros(d0 + num_hidden + 1, 1);
    w_o_temp(end) = 1;
    
    % Tahmin yap
    Yhat_seq = forwardCCNN_recursiveTrajectory(X0_bias, U_seq, W_hidden_temp, g, w_o_temp, config);
    Eseq = T_seq - Yhat_seq;
    current_mse = 0.5 * mean(Eseq.^2, 'all');
end
function [loss, grad_w] = loss_and_grad_candidate_traj(w_h, X0_dl, U_dl, T_dl, W_hidden, g, config)
% Candidate unit için trajectory loss ve gradient
    
    M = size(X0_dl,1);
    Npred = size(U_dl,2);
    num_hidden = numel(W_hidden);
    
    % Mevcut hidden unit'lerin çıktılarını hesapla
    if config.regressors.include_bias
        y_prev = X0_dl(:,3);
    else
        y_prev = X0_dl(:,2);
    end
    
    % Candidate unit eklenmiş şekilde tahmin yap
    Yhat_seq = dlarray(zeros(M, Npred));
    
    for t = 1:Npred
        u_curr = U_dl(:,t);
        
        if config.regressors.include_bias
            x = [dlarray(ones(M,1)), u_curr, y_prev];
        else
            x = [u_curr, y_prev];
        end
        
        x_aug = x;
        % Mevcut hidden unit'ler
        for h = 1:num_hidden
            v = g(x_aug * W_hidden{h});
            x_aug = [x_aug, v];
        end
        
        % Candidate unit çıktısı
        v_candidate = g(x_aug * w_h);
        x_aug_with_candidate = [x_aug, v_candidate];
        
        % Çıkış ağırlıkları: tüm ağırlıklar 0, sadece candidate unit için 1
        w_o_temp = zeros(size(x_aug_with_candidate, 2), 1);
        w_o_temp(end) = 1;
        
        yhat = x_aug_with_candidate * w_o_temp;
        Yhat_seq(:,t) = yhat;
        y_prev = yhat;
    end
    
    E = T_dl - Yhat_seq;
    loss = 0.5 * mean(E.^2, 'all');
    grad_w = dlgradient(loss, w_h);
end

function [w_o_trained, E_residual, current_mse] = trainOutputLayer_GD_Autograd_Trajectory( ...
    X0_bias, U_seq, T_seq, w_initial, W_hidden, g, config, max_epochs, eta)

    X0_dl = dlarray(X0_bias);
    U_dl  = dlarray(U_seq);
    T_dl  = dlarray(T_seq);
    w     = dlarray(w_initial);

    averageGrad = [];
    averageSqGrad = [];
    iteration = 0;

    gradDecay = 0.9;
    sqGradDecay = 0.999;
    eps_adam = 1e-8;

    for epoch = 1:max_epochs
        iteration = iteration + 1;

        [loss, grad_w] = dlfeval(@loss_and_grad_traj, w, X0_dl, U_dl, T_dl, W_hidden, g, config);

        [w, averageGrad, averageSqGrad] = adamupdate(w, grad_w, ...
            averageGrad, averageSqGrad, iteration, ...
            eta, gradDecay, sqGradDecay, eps_adam);
    end

    w_o_trained = extractdata(w);

    % residuals for candidate training
    Yhat = forwardCCNN_recursiveTrajectory(extractdata(X0_dl), extractdata(U_dl), W_hidden, g, w_o_trained, config);
    Eseq = T_seq - Yhat;

    current_mse = 0.5 * mean(Eseq.^2, 'all');
    E_residual  = mean(Eseq, 2);   % per-sample trajectory residual
end

function [loss, grad_w] = loss_and_grad_traj(w, X0_dl, U_dl, T_dl, W_hidden, g, config)
    M = size(X0_dl,1);
    Npred = size(U_dl,2);
    num_hidden = numel(W_hidden);

    if config.regressors.include_bias
        y_prev = X0_dl(:,3);
    else
        y_prev = X0_dl(:,2);
    end

    Yhat = dlarray(zeros(M, Npred));

    for t = 1:Npred
        u_curr = U_dl(:,t);

        if config.regressors.include_bias
            x = [dlarray(ones(M,1)), u_curr, y_prev];
        else
            x = [u_curr, y_prev];
        end

        x_aug = x;
        for h = 1:num_hidden
            v = g(x_aug * W_hidden{h});
            x_aug = [x_aug, v];
        end

        yhat = x_aug * w;
        Yhat(:,t) = yhat;
        y_prev = yhat;
    end

    E = T_dl - Yhat;
    loss = 0.5 * mean(E.^2, 'all');
    grad_w = dlgradient(loss, w);
end

function Yhat_full = recursivePredictFullSeries(U, y_init, W_hidden, w_o, g, config)
% GÜNCELLEME: Sadece ilk girişle başlayıp tüm seriyi recursive tahmin et
    
    N = size(U,1);
    Yhat_full = zeros(N,1);
    num_hidden = numel(W_hidden);
    
    % İlk değeri ayarla
    y_prev = y_init;
    
    for k = 1:N
        u_curr = U(k);
        
        % Regressörleri oluştur
        if config.regressors.include_bias
            x = [1, u_curr, y_prev];
        else
            x = [u_curr, y_prev];
        end
        
        % Hidden katman çıktılarını hesapla
        x_aug = x;
        for h = 1:num_hidden
            v = g(x_aug * W_hidden{h});
            x_aug = [x_aug, v];
        end
        
        % Çıkışı hesapla
        yhat = x_aug * w_o;
        Yhat_full(k) = yhat;
        
        % Bir sonraki adım için geri besleme
        y_prev = yhat;
    end
end