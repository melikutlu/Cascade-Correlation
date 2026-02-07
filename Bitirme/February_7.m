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
config.norm_method = 'ZScore'; 

% --- Recursive N-step ---
config.prediction.n_steps = 50;   % N=50
config.validation.n_steps = 500;

% --- Regressors (SISO): model uses u(k) and y(k-1)
config.regressors.include_bias = false;

% --- CCNN hyperparams ---
config.model.max_hidden_units   = 10;
config.model.target_mse         = 5e-5;

config.model.max_epochs_output  = 1500; % Output eğitimi için epoch artırıldı
config.model.eta_output_gd      = 0.005;

% Candidate ayarları
config.model.max_epochs_candidate = 1000;
config.model.eta_candidate        = 0.02; % Candidate için daha hızlı öğrenme

config.model.activation = 'tanh'; 

%% 1) LOAD DATA
fprintf('\n=== LOAD DATA ===\n');
[U_train_raw, Y_train_raw, U_val_raw, Y_val_raw] = loadDataByConfig_min(config);
fprintf('Train: %d samples | Val: %d samples\n', size(U_train_raw,1), size(U_val_raw,1));

%% 2) NORMALIZE (train stats)
fprintf('\n=== NORMALIZE (%s) ===\n', config.norm_method);
[U_train, Y_train, U_val, Y_val, norm_stats] = normalizeData_min( ...
    config.norm_method, U_train_raw, Y_train_raw, U_val_raw, Y_val_raw);

%% 3) BUILD TRAJECTORY DATASET
Npred = config.prediction.n_steps;
fprintf('\n=== BUILD TRAJECTORY DATASET (Recursive N=%d) ===\n', Npred);

[X0_train, Utr_seq, Ttr_seq, reg_train] = createTrajectoryDataset(U_train, Y_train, config, Npred);
[X0_val,   Uval_seq, Tval_seq, reg_val] = createTrajectoryDataset(U_val,   Y_val,   config, Npred);

fprintf('Train: X0=%dx%d | U_seq=%dx%d | T_seq=%dx%d\n', ...
    size(X0_train,1), size(X0_train,2), size(Utr_seq,1), size(Utr_seq,2), size(Ttr_seq,1), size(Ttr_seq,2));

%% 4) ACTIVATION
switch lower(config.model.activation)
    case 'tanh'
        g = @(a) tanh(a);
    case 'sigmoid'
        g = @(a) 1./(1 + exp(-a));
    case 'relu'
        g = @(a) max(0, a);
    otherwise
        error('Unknown activation: %s', config.model.activation);
end

%% 5) TRAIN CCNN (Recursive N-step Trajectory Loss)
fprintf('\n=== TRAIN CCNN (Recursive N-step, Trajectory Loss) ===\n');

W_hidden = {};                         
X_candidate_input = X0_train;          

d0 = size(X0_train,2);                 
w_o_initial = randn(d0, 1) * 0.01;     

% ---- Stage 1: train output only ----
[w_o_stage1, E_residual, current_mse] = trainOutputLayer_GD_Autograd_Trajectory( ...
    X0_train, Utr_seq, Ttr_seq, w_o_initial, W_hidden, g, config, ...
    config.model.max_epochs_output, config.model.eta_output_gd);

mse_history = current_mse;
fprintf('Stage-1 Train MSE: %.6g\n', current_mse);

w_o = w_o_stage1;

% ---- Stage 2: add hidden units ----
while current_mse > config.model.target_mse && numel(W_hidden) < config.model.max_hidden_units
    h = numel(W_hidden) + 1;
    
    % GÜNCELLEME 1: Candidate eğitimi (Fixed Output Scale ile)
    % Çıkış ağırlığını öğrenmeye çalışmıyoruz, onu sabit tutup sadece nöronu eğitiyoruz.
    [w_h, v_h_train, mse_cand_estimated] = trainCandidateUnit_Trajectory_FixedScale( ...
        X0_train, Utr_seq, Ttr_seq, W_hidden, w_o, g, config, ...
        config.model.max_epochs_candidate, config.model.eta_candidate);
    
    fprintf('Candidate #%d trained | Est. MSE impact: %.6g\n', h, mse_cand_estimated);
    
    W_hidden{h} = w_h;
    
    % GÜNCELLEME 2: Zero-Initialization (Sıfır Başlatma)
    % Yeni eklenen nöronun çıkış ağırlığını 0 yapıyoruz.
    % Böylece MSE artmıyor, aynı kalıyor. Sonraki adımda düşüyor.
    w_o_init_new = [w_o; 0];  
    
    % Retrain output layer
    [w_o, E_residual, current_mse] = trainOutputLayer_GD_Autograd_Trajectory( ...
        X0_train, Utr_seq, Ttr_seq, w_o_init_new, W_hidden, g, config, ...
        config.model.max_epochs_output, config.model.eta_output_gd);

    mse_history(end+1) = current_mse;

    % Plot
    figure(100); clf; set(gcf,'Color','w','Name','Train MSE vs Hidden Units');
    plot(0:(numel(mse_history)-1), mse_history, '-o', 'LineWidth', 1.2);
    grid on; xlabel('Hidden unit sayısı'); ylabel('Train MSE');
    title(sprintf('Train MSE history | Hidden=%d | Current MSE=%.6g', numel(W_hidden), current_mse));
    drawnow;

    fprintf('Hidden #%d added | Train MSE: %.6g\n', h, current_mse);

    if numel(mse_history) >= 2
        % N=50'de çok küçük düşüşler bile önemlidir, toleransı düşük tutuyoruz
        if (mse_history(end-1) - mse_history(end)) < 1e-8
            fprintf('Stop: MSE improvement too small or zero.\n');
            break;
        end
    end
end

fprintf('Total hidden units: %d\n', numel(W_hidden));
fprintf('Final Train MSE: %.6g\n', current_mse);

%% 6) VALIDATION
fprintf('\n=== TRAINING - RECURSIVE PREDICTION (Full Series) ===\n');
Yhat_train_recursive = recursivePredictFullSeries(U_train, Y_train, W_hidden, w_o, g, config);
Yhat_train_recursive = Yhat_train_recursive(2:end);  
Y_train_compare = Y_train_raw(2:end); 
Yhat_train_real = denormalizeData_min(Yhat_train_recursive, config.norm_method, norm_stats.y);
fit_train_full = fitPercent(Y_train_compare, Yhat_train_real);
fprintf('Full Series Training Fit: %.2f%%\n', fit_train_full);

% Plot Train
figure('Color','w','Name','TRAIN - Full Series');
time_train = 2:length(Y_train_raw);
plot(time_train, Y_train_compare, 'k', 'LineWidth', 1.5); hold on;
plot(time_train, Yhat_train_real, 'b--', 'LineWidth', 1.2);
grid on; legend('True', 'CCNN Prediction'); title('TRAIN Fit');

fprintf('\n=== VALIDATION - RECURSIVE PREDICTION (Full Series) ===\n');
Yhat_val_recursive = recursivePredictFullSeries(U_val, Y_val, W_hidden, w_o, g, config);
Yhat_val_recursive = Yhat_val_recursive(2:end); 
Y_val_compare = Y_val_raw(2:end); 
Yhat_val_real = denormalizeData_min(Yhat_val_recursive, config.norm_method, norm_stats.y);
fit_val_full = fitPercent(Y_val_compare, Yhat_val_real);
fprintf('Full Series Validation Fit: %.2f%%\n', fit_val_full);

% Plot Val
figure('Color','w','Name','VAL - Full Series');
time_val = 2:length(Y_val_raw);
plot(time_val, Y_val_compare, 'k', 'LineWidth', 1.5); hold on;
plot(time_val, Yhat_val_real, 'r--', 'LineWidth', 1.2);
grid on; legend('True', 'CCNN Prediction'); title('VAL Fit');

%% =========================
%% Local helper functions
%% =========================

function [U_train, Y_train, U_val, Y_val] = loadDataByConfig_min(config)
    load dryer2; 
    z_full = iddata(y2, u2, config.data.twotank.sampling_time);
    N_total = length(z_full.y);
    train_end = floor(N_total * config.data.train_ratio);
    val_end = train_end + floor(N_total * config.data.val_ratio);
    
    z1 = z_full(1:train_end);
    z1f = idfilt(z1, 3, config.data.twotank.filter_cutoff);
    z1f = z1f(config.data.twotank.warmup_samples:end);
    U_train = z1f.u; Y_train = z1f.y;

    z2 = z_full(train_end+1:val_end);
    z2f = idfilt(z2, 3, config.data.twotank.filter_cutoff);
    z2f = z2f(config.data.twotank.warmup_samples:end);
    U_val = z2f.u; Y_val = z2f.y;
end

function [Utr_n, Ytr_n, Uva_n, Yva_n, stats] = normalizeData_min(method, Utr, Ytr, Uva, Yva)
    stats = struct();
    stats.u.mu = mean(Utr,1); stats.u.sg = std(Utr,0,1); stats.u.sg(stats.u.sg==0)=1;
    stats.y.mu = mean(Ytr,1); stats.y.sg = std(Ytr,0,1); stats.y.sg(stats.y.sg==0)=1;
    Utr_n = (Utr - stats.u.mu) ./ stats.u.sg; Ytr_n = (Ytr - stats.y.mu) ./ stats.y.sg;
    Uva_n = (Uva - stats.u.mu) ./ stats.u.sg; Yva_n = (Yva - stats.y.mu) ./ stats.y.sg;
end

function X = denormalizeData_min(Xn, method, st)
    X = Xn .* st.sg + st.mu;
end

function fitp = fitPercent(ytrue, yhat)
    fitp = (1 - norm(ytrue - yhat) / norm(ytrue - mean(ytrue))) * 100;
end

function [X0_bias, U_seq, T_seq, reg] = createTrajectoryDataset(U, Y, config, Npred)
    N = size(U,1); max_lag = 1;
    start_idx = max_lag + 1; end_idx = N - Npred + 1;
    if end_idx < start_idx, error('Npred too large.'); end
    M = end_idx - start_idx + 1;
    U_seq = zeros(M, Npred); T_seq = zeros(M, Npred);
    
    if config.regressors.include_bias, X0_bias = zeros(M, 3); else, X0_bias = zeros(M, 2); end

    for i = 1:M
        k = start_idx + i - 1;
        u0 = U(k,1); yprev = Y(k-1,1);
        if config.regressors.include_bias, X0_bias(i,:) = [1, u0, yprev];
        else, X0_bias(i,:) = [u0, yprev]; end
        U_seq(i,:) = U(k:(k+Npred-1),1).';
        T_seq(i,:) = Y(k:(k+Npred-1),1).';
    end
    reg.start_idx = start_idx; reg.end_idx = end_idx;
end

function Yhat_seq = forwardCCNN_recursiveTrajectory(X0_bias, U_seq, W_hidden, g, w_o, config)
    M = size(X0_bias,1); Npred = size(U_seq,2); num_hidden = numel(W_hidden);
    Yhat_seq = zeros(M, Npred);
    if config.regressors.include_bias, y_prev = X0_bias(:,3); else, y_prev = X0_bias(:,2); end

    for t = 1:Npred
        u_curr = U_seq(:,t);
        if config.regressors.include_bias, x = [ones(M,1), u_curr, y_prev];
        else, x = [u_curr, y_prev]; end
        
        x_aug = x;
        for h = 1:num_hidden, v = g(x_aug * W_hidden{h}); x_aug = [x_aug, v]; end
        
        yhat = x_aug * w_o;
        Yhat_seq(:,t) = yhat;
        y_prev = yhat;
    end
end

function [w_h, v_h, current_mse] = trainCandidateUnit_Trajectory_FixedScale( ...
    X0_bias, U_seq, T_seq, W_hidden, w_o_current, g, config, max_epochs, eta)
    
    [M, d0] = size(X0_bias);
    num_hidden = numel(W_hidden);
    
    % Random initialize candidate weights
    w_h = randn(d0 + num_hidden, 1) * 0.05; 
    
    % FIXED SCALE: Bu değer sabit! Adam bunu güncelleyemez.
    % 0.1 gibi bir değer veriyoruz ki türev (gradient) 0 olmasın ve w_h güncellenebilsin.
    val_out_fixed = 0.1; 

    avgGrad_wh = []; avgSqGrad_wh = [];
    iteration = 0;
    
    gradDecay = 0.9; sqGradDecay = 0.999; eps_adam = 1e-8;
    
    X0_dl = dlarray(X0_bias); U_dl = dlarray(U_seq); T_dl = dlarray(T_seq);
    w_h_dl = dlarray(w_h); 
    w_o_current_dl = dlarray(w_o_current);
    
    for epoch = 1:max_epochs
        iteration = iteration + 1;
        
        % Loss hesaplarken val_out_fixed'i skaler olarak gönderiyoruz
        [loss, grad_wh] = dlfeval(@loss_and_grad_candidate_fixed_scale, ...
            w_h_dl, val_out_fixed, X0_dl, U_dl, T_dl, W_hidden, w_o_current_dl, g, config);
        
        % Gradient Clipping (N=50 için gerekli)
        grad_wh = threshold_l2norm(grad_wh, 1.0);
        
        [w_h_dl, avgGrad_wh, avgSqGrad_wh] = adamupdate(w_h_dl, grad_wh, ...
            avgGrad_wh, avgSqGrad_wh, iteration, eta, gradDecay, sqGradDecay, eps_adam);
    end
    
    w_h = extractdata(w_h_dl);
    
    % Output hesapla
    if config.regressors.include_bias, y_prev = X0_bias(:,3); else, y_prev = X0_bias(:,2); end
    u_curr = U_seq(:,1);
    if config.regressors.include_bias, x = [ones(M,1), u_curr, y_prev]; else, x = [u_curr, y_prev]; end
    
    x_aug = x;
    for h = 1:num_hidden, v = g(x_aug * W_hidden{h}); x_aug = [x_aug, v]; end
    v_h = g(x_aug * w_h);
    
    % MSE tahmini
    W_hidden_temp = W_hidden; W_hidden_temp{end+1} = w_h;
    w_o_temp = [w_o_current; val_out_fixed];
    Yhat_seq = forwardCCNN_recursiveTrajectory(X0_bias, U_seq, W_hidden_temp, g, w_o_temp, config);
    Eseq = T_seq - Yhat_seq;
    current_mse = 0.5 * mean(Eseq.^2, 'all');
end

function [loss, grad_wh] = loss_and_grad_candidate_fixed_scale(...
    w_h, val_out_fixed, X0_dl, U_dl, T_dl, W_hidden, w_o_current, g, config)
    
    M = size(X0_dl,1); Npred = size(U_dl,2); num_hidden = numel(W_hidden);
    if config.regressors.include_bias, y_prev = X0_dl(:,3); else, y_prev = X0_dl(:,2); end
    
    Yhat_seq = dlarray(zeros(M, Npred));
    
    % Burada candidate çıkış ağırlığı SABİT ekleniyor
    w_o_temp = [w_o_current; val_out_fixed];
    
    for t = 1:Npred
        u_curr = U_dl(:,t);
        if config.regressors.include_bias, x = [dlarray(ones(M,1)), u_curr, y_prev];
        else, x = [u_curr, y_prev]; end
        
        x_aug = x;
        for h = 1:num_hidden, v = g(x_aug * W_hidden{h}); x_aug = [x_aug, v]; end
        
        v_candidate = g(x_aug * w_h);
        x_aug_final = [x_aug, v_candidate];
        
        yhat = x_aug_final * w_o_temp;
        Yhat_seq(:,t) = yhat;
        y_prev = yhat;
    end
    
    E = T_dl - Yhat_seq;
    loss = 0.5 * mean(E.^2, 'all');
    grad_wh = dlgradient(loss, w_h);
end

function gradients = threshold_l2norm(gradients, gradientThreshold)
    gradientNorm = sqrt(sum(gradients.^2, 'all'));
    if gradientNorm > gradientThreshold
        gradients = gradients * (gradientThreshold / gradientNorm);
    end
end

function [w_o_trained, E_residual, current_mse] = trainOutputLayer_GD_Autograd_Trajectory( ...
    X0_bias, U_seq, T_seq, w_initial, W_hidden, g, config, max_epochs, eta)

    X0_dl = dlarray(X0_bias); U_dl = dlarray(U_seq); T_dl = dlarray(T_seq); w = dlarray(w_initial);
    avgGrad = []; avgSqGrad = []; iteration = 0;
    gradDecay = 0.9; sqGradDecay = 0.999; eps_adam = 1e-8;

    for epoch = 1:max_epochs
        iteration = iteration + 1;
        [loss, grad_w] = dlfeval(@loss_and_grad_traj, w, X0_dl, U_dl, T_dl, W_hidden, g, config);
        
        % Output layer için de gradient clip koyuyoruz, w_o patlamasın
        grad_w = threshold_l2norm(grad_w, 2.0);
        
        [w, avgGrad, avgSqGrad] = adamupdate(w, grad_w, avgGrad, avgSqGrad, iteration, eta, gradDecay, sqGradDecay, eps_adam);
    end
    w_o_trained = extractdata(w);
    Yhat = forwardCCNN_recursiveTrajectory(extractdata(X0_dl), extractdata(U_dl), W_hidden, g, w_o_trained, config);
    Eseq = T_seq - Yhat;
    current_mse = 0.5 * mean(Eseq.^2, 'all');
    E_residual  = mean(Eseq, 2);
end

function [loss, grad_w] = loss_and_grad_traj(w, X0_dl, U_dl, T_dl, W_hidden, g, config)
    M = size(X0_dl,1); Npred = size(U_dl,2); num_hidden = numel(W_hidden);
    if config.regressors.include_bias, y_prev = X0_dl(:,3); else, y_prev = X0_dl(:,2); end
    Yhat = dlarray(zeros(M, Npred));

    for t = 1:Npred
        u_curr = U_dl(:,t);
        if config.regressors.include_bias, x = [dlarray(ones(M,1)), u_curr, y_prev];
        else, x = [u_curr, y_prev]; end
        x_aug = x;
        for h = 1:num_hidden, v = g(x_aug * W_hidden{h}); x_aug = [x_aug, v]; end
        yhat = x_aug * w;
        Yhat(:,t) = yhat;
        y_prev = yhat;
    end
    E = T_dl - Yhat;
    loss = l2loss(Yhat, T_dl, 'DataFormat', 'BC');
    grad_w = dlgradient(loss, w);
end

function Yhat_full = recursivePredictFullSeries(U, Y, W_hidden, w_o, g, config)   
    N = size(U,1); Yhat_full = zeros(N,1); num_hidden = numel(W_hidden);
    Yhat_full(1) = Y(1); y_prev = Y(1);
    for k = 2:N
        u_curr = U(k);
        if config.regressors.include_bias, x = [1, u_curr, y_prev]; else, x = [u_curr, y_prev]; end
        x_aug = x;
        for h = 1:num_hidden, v = g(x_aug * W_hidden{h}); x_aug = [x_aug, v]; end
        yhat = x_aug * w_o;
        Yhat_full(k) = yhat;
        y_prev = yhat;
    end
    
end