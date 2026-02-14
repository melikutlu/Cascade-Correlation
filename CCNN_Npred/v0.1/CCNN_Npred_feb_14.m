% CCNN_Npred.m
% CCNN where candidate units are trained to MAXIMIZE N-step residual correlation
% Model and candidate training both operate on N-step trajectory predictions.

clear; clc; close all; rng(0);

% ------------------
% CONFIG
% ------------------
config = struct();
config.data.source = 'twotankdata';
config.data.twotank.warmup_samples = 0;
config.data.twotank.sampling_time = 0.2; % s
config.data.twotank.filter_cutoff = 0.066902; % Hz (optional)

config.data.train_ratio = 0.8;
config.data.val_ratio = 0.2;

config.norm_method = 'ZScore';

config.prediction.n_steps = 15; % default N-step horizon (can be auto-adjusted)
config.prediction.auto_full_horizon = true; % set true to span full usable data length

% regressors (user can change)
config.regressors.u = [0 1 2]; % example: u(t), u(t-1)
config.regressors.y = [1 2 3]; % example: y(t-1), y(t-2)
config.regressors.include_bias = false;

% model / training
config.model.activation = 'tanh';
config.model.max_hidden_units = 10;
config.model.target_mse = 5e-5;
config.model.min_mse_improvement = 1e-6; % early stop threshold

% Adam typically saturates within 100-300 epochs; plateau guard stops early.
config.model.max_epochs_output = 100;
config.model.eta_output = 0.01;
config.model.max_epochs_candidate = 100;
config.model.eta_candidate = 0.03;
config.model.plateau_min_delta = 1e-6;   % loss/metrik iyileşmesi bu değerden küçükse
config.model.plateau_patience = 50;      % bu kadar epoch boyunca iyileşme yoksa dur

% ------------------
% DATA
% ------------------
[Utr_raw, Ytr_raw, Uva_raw, Yva_raw] = loadDataByConfig_min(config);
[Utr, Ytr, Uva, Yva, norm_stats] = normalizeData_min(config.norm_method, Utr_raw, Ytr_raw, Uva_raw, Yva_raw);

if isfield(config.prediction, 'auto_full_horizon') && config.prediction.auto_full_horizon
    maxLag = getMaxLagFromRegressors(config.regressors);
    maxStepsTr = numel(Ytr) - maxLag - 1;
    maxStepsVa = numel(Yva) - maxLag - 1;
    autoSteps = min([maxStepsTr, maxStepsVa]);
    if autoSteps < 1
        error('Not enough samples to build at least one full-horizon trajectory.');
    end
    config.prediction.n_steps = autoSteps;
end

Npred = config.prediction.n_steps;
[X0_tr, Utr_seq, Ttr_seq] = createTrajectoryDataset(Utr, Ytr, config, Npred);
[X0_va, Uva_seq, Tva_seq] = createTrajectoryDataset(Uva, Yva, config, Npred);

% activation
g = @(x) tanh(x);

% initialize
W_hidden = {};
d0 = size(X0_tr,2);
w_o = randn(d0,1)*0.01;

% Stage 1: train output weights only (N-step MSE)
[w_o, current_mse, outputTrainInfo] = trainOutputLayer_Trajectory(X0_tr, Utr_seq, Ttr_seq, w_o, W_hidden, g, config);
fprintf('Stage-1 Train MSE: %.6g\n', current_mse);

if ~isnan(outputTrainInfo.plateau_epoch)
    fprintf('Output layer plateau at epoch %d (ran %d/%d epochs).\n', ...
        outputTrainInfo.plateau_epoch, outputTrainInfo.epochs_run, config.model.max_epochs_output);
else
    fprintf('Output layer used %d/%d epochs (no plateau).\n', ...
        outputTrainInfo.epochs_run, config.model.max_epochs_output);
end

mse_hist = current_mse;
candidateEpochHistory = [];
candidatePlateauHistory = [];

% Greedy growth
while current_mse > config.model.target_mse && numel(W_hidden) < config.model.max_hidden_units
    h = numel(W_hidden) + 1;
    fprintf('\nTraining candidate #%d (maximize N-step residual correlation)\n', h);

    % train candidate to maximize correlation with residual (N-step)
    [w_h, cand_metric, candInfo] = trainCandidateUnit_Corr(X0_tr, Utr_seq, Ttr_seq, W_hidden, w_o, g, config);
    candidateEpochHistory(end+1) = candInfo.epochs_run; %#ok<AGROW>
    candidatePlateauHistory(end+1) = candInfo.plateau_epoch; %#ok<AGROW>
    fprintf('Candidate #%d | corr^2 (on train residual): %.6g\n', h, cand_metric);
    if ~isnan(candInfo.plateau_epoch)
        fprintf('Candidate #%d plateau at epoch %d (ran %d/%d epochs).\n', ...
            h, candInfo.plateau_epoch, candInfo.epochs_run, config.model.max_epochs_candidate);
    else
        fprintf('Candidate #%d used %d/%d epochs (no plateau).\n', ...
            h, candInfo.epochs_run, config.model.max_epochs_candidate);
    end

    % tentatively add candidate
    W_hidden{end+1} = w_h;
    w_o = [w_o; randn*0.01];

    prev_mse = current_mse;
    [w_o, current_mse, outputTrainInfo] = trainOutputLayer_Trajectory(X0_tr, Utr_seq, Ttr_seq, w_o, W_hidden, g, config);

    improvement = prev_mse - current_mse;
    if improvement < config.model.min_mse_improvement
        % undo
        W_hidden(end) = [];
        w_o = w_o(1:end-1);
        fprintf('Undo candidate #%d: improvement %.3g < threshold %.3g. Stopping growth.\n', h, improvement, config.model.min_mse_improvement);
        break;
    end

    mse_hist(end+1) = current_mse;
    fprintf('Hidden=%d | Train MSE=%.6g | improvement=%.3g\n', numel(W_hidden), current_mse, improvement);
    if ~isnan(outputTrainInfo.plateau_epoch)
        fprintf('Output layer re-train plateau at epoch %d (ran %d/%d epochs).\n', ...
            outputTrainInfo.plateau_epoch, outputTrainInfo.epochs_run, config.model.max_epochs_output);
    end
end

% Full-series recursive prediction and denormalize
Yhat_tr = recursivePredictFullSeries(Utr, Ytr, W_hidden, w_o, g, config);
Yhat_va = recursivePredictFullSeries(Uva, Yva, W_hidden, w_o, g, config);

Yhat_tr = Yhat_tr(2:end) * norm_stats.y_std + norm_stats.y_mu;
Yhat_va = Yhat_va(2:end) * norm_stats.y_std + norm_stats.y_mu;

fit_tr = fitPercent(Ytr_raw(2:end), Yhat_tr);
fit_va = fitPercent(Yva_raw(2:end), Yhat_va);
fprintf('\nTrain Fit: %.2f%% | Val Fit: %.2f%%\n', fit_tr, fit_va);

% Persist key hyperparameters so manual tweaks are traceable.
logInfo = struct();
logInfo.eta_output = config.model.eta_output;
logInfo.eta_candidate = config.model.eta_candidate;
logInfo.max_epochs_output = config.model.max_epochs_output;
logInfo.output_epochs_used = outputTrainInfo.epochs_run;
logInfo.output_plateau_epoch = outputTrainInfo.plateau_epoch;
logInfo.max_epochs_candidate = config.model.max_epochs_candidate;
logInfo.candidate_epochs_used = candidateEpochHistory;
logInfo.candidate_plateau_epochs = candidatePlateauHistory;
logInfo.candidate_runs = numel(candidateEpochHistory);
logInfo.plateau_min_delta = config.model.plateau_min_delta;
logInfo.plateau_patience = config.model.plateau_patience;
logInfo.hidden_units = numel(W_hidden);
logInfo.max_hidden_units = config.model.max_hidden_units;
logInfo.regressor_count = numel(config.regressors.u) + numel(config.regressors.y);
logInfo.regressors_u = config.regressors.u;
logInfo.regressors_y = config.regressors.y;
logInfo.n_steps = Npred;
logInfo.train_mse = current_mse;
logInfo.fit_train = fit_tr;
logInfo.fit_val = fit_va;
logFilePath = writeParameterLog(config, logInfo);
if ~isempty(logFilePath)
    fprintf('Parameter log saved to %s\n', logFilePath);
end

% Plots (use filtered raw data loaded earlier)
figTrain = figure('Name','TRAIN - Full Recursive','Color','w');
plot(Ytr_raw(2:end),'k','LineWidth',1.4); hold on;
plot(Yhat_tr,'b--','LineWidth',1.2); grid on;
title(sprintf('TRAIN | Hidden=%d | Fit=%.2f%%', numel(W_hidden), fit_tr)); legend('True','CCNN');

figVal = figure('Name','VAL - Full Recursive','Color','w');
plot(Yva_raw(2:end),'k','LineWidth',1.4); hold on;
plot(Yhat_va,'r--','LineWidth',1.2); grid on;
title(sprintf('VAL | Hidden=%d | Fit=%.2f%%', numel(W_hidden), fit_va)); legend('True','CCNN');

savedFigurePaths = saveFitFigures(logFilePath, struct('train', figTrain, 'val', figVal));
if ~isempty(savedFigurePaths) && ~isempty(logFilePath)
    appendFigureInfoToLog(logFilePath, savedFigurePaths);
end

% ------------------
% LOCAL FUNCTIONS
% ------------------
function [w_h, best_metric, info] = trainCandidateUnit_Corr(X0,U,T,W_hidden,w_o,g,config)
    % Train a candidate unit to MAXIMIZE correlation^2 with the N-step residual
    % Inputs are trajectory batches: X0 (Mxd0), U (MxN), T (MxN)
    % W_hidden, w_o are current model parameters (kept fixed)

    % Giriş boyutu = regresör sayısı (u + y)
    nu = numel(config.regressors.u);
    ny = numel(config.regressors.y);
    d = nu + ny;  % Temel giriş boyutu
    
    % Mevcut gizli nöronların çıktıları da girişe eklenir
    d = d + numel(W_hidden);  % TOPLAM giriş boyutu
    
    w_h = dlarray(randn(d, 1) * 0.01);

    X0_d = dlarray(X0); U_d = dlarray(U); T_d = dlarray(T);
    w_o_d = dlarray(w_o);

    avgG=[]; avgGSq=[]; it=0; best_metric = -Inf; best_w = extractdata(w_h);
    maxEpochs = config.model.max_epochs_candidate;
    metric_hist = zeros(maxEpochs,1);
    epochsSinceBest = 0;
    plateauEpoch = NaN;
    minDelta = config.model.plateau_min_delta;
    patience = config.model.plateau_patience;

    for ep=1:maxEpochs
        it = it + 1;
        [loss, metric, grad] = dlfeval(@loss_candidate_corr, w_h, X0_d, U_d, T_d, W_hidden, w_o_d, g, config);
        [w_h, avgG, avgGSq] = adamupdate(w_h, grad, avgG, avgGSq, it, config.model.eta_candidate);
        metricVal = gather(extractdata(metric));
        metric_hist(ep) = metricVal;
        if metricVal - best_metric > minDelta
            best_metric = metricVal;
            best_w = extractdata(w_h);
            epochsSinceBest = 0;
        else
            epochsSinceBest = epochsSinceBest + 1;
            if epochsSinceBest >= patience
                plateauEpoch = ep;
                break;
            end
        end
    end
    w_h = best_w;
    epochs_run = ep;
    metric_hist = metric_hist(1:epochs_run);
    info = struct('epochs_run', epochs_run, 'plateau_epoch', plateauEpoch, 'metric_history', metric_hist);
end

function [L, metric, grad] = loss_candidate_corr(w_h, X0, U, T, W_hidden, w_o, g, config)
    % compute current model N-step output without candidate
    Y_model = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config);
    R = T - Y_model; % residual (M x N)

    % compute candidate activation v (M x N)
    M = size(X0,1); N = size(U,2);
    v = dlarray(zeros(M,N));
    for t=1:N
        % build regressor vector x_t for all M samples
        x_t = buildRegressorRow(X0, U, t, W_hidden, g);
        v(:,t) = g(x_t * w_h);
    end

    % flatten and center
    r_vec = reshape(R, [], 1);
    v_vec = reshape(v, [], 1);
    r_mean = mean(r_vec);
    v_mean = mean(v_vec);
    r_c = r_vec - r_mean;
    v_c = v_vec - v_mean;

    cov_vr = sum(v_c .* r_c);
    denom = (sum(v_c.^2) + eps) .* (sum(r_c.^2) + eps);
    corr2 = (cov_vr.^2) ./ denom; % correlation squared (scalar)

    metric = corr2;
    L = -corr2; % minimize negative corr^2 -> maximize corr^2
    grad = dlgradient(L, w_h);
end

function x = buildRegressorRow(X0, U, t, W_hidden, g)
    ulags = evalin('caller', 'config.regressors.u');
    ylags = evalin('caller', 'config.regressors.y');
    ulags = ulags(:)'; 
    ylags = ylags(:)';
    nu = numel(ulags); 
    ny = numel(ylags);
    M = size(X0, 1);
    
    % uvals hesapla
    uvals = zeros(M, nu);
    for j = 1:nu
        L = ulags(j);
        if L == 0
            uvals(:, j) = U(:, t);
        else
            idx = t - L;
            if idx >= 1
                uvals(:, j) = U(:, idx);
            else
                uvals(:, j) = X0(:, j);
            end
        end
    end
    
    % yvals hesapla - BURASI ÖNEMLİ!
    yvals = zeros(M, ny);
    for j = 1:ny
        L = ylags(j);
        idx = t - L;
        if idx >= 1
            % Geçmişteki tahmin edilmiş y değerleri lazım!
            % Şimdilik X0 kullanıyoruz - BU YANLIŞ!
            yvals(:, j) = 0;  % TODO: Buraya önceki tahminler gelmeli
        else
            yvals(:, j) = X0(:, nu + j);
        end
    end
    
    % ÖNCE dlarray'e çevir
    x = dlarray([uvals, yvals]);
    
    % SONRA gizli katmanları ekle (her birini dlarray'e çevirerek)
    for h = 1:numel(W_hidden)
        w_h = dlarray(W_hidden{h});  % cell array'den dlarray'e
        hidden_output = g(x * w_h);
        x = [x, hidden_output];
    end
end
function Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config)
    M = size(X0, 1); 
    N = size(U, 2);
    Y = dlarray(zeros(M, N));
    
    ulags = config.regressors.u(:)';
    ylags = config.regressors.y(:)';
    nu = numel(ulags);
    ny = numel(ylags);
    
    % yprev: [y(t-1), y(t-2), ..., y(t-ny)] şeklinde sıralı
    % X0'un son ny sütunu: [y(t-1), y(t-2), ..., y(t-ny)] OLMALI!
    yprev = X0(:, nu+1:nu+ny);  % Bu doğru
    
    for t = 1:N
        % uvals hesapla (aynı)
        uvals = zeros(M, nu);
        for j = 1:nu
            L = ulags(j);
            if L == 0
                uvals(:, j) = U(:, t);
            else
                idx = t - L;
                if idx >= 1
                    uvals(:, j) = U(:, idx);
                else
                    uvals(:, j) = X0(:, j);
                end
            end
        end
        
        % yvals hesapla - DÜZELTİLDİ!
        yvals = zeros(M, ny);
        for j = 1:ny
            L = ylags(j);
            % ylags = [1,2,3] ise:
            % L=1 -> y(t-1) -> yprev'in 1. sütunu
            % L=2 -> y(t-2) -> yprev'in 2. sütunu
            % L=3 -> y(t-3) -> yprev'in 3. sütunu
            yvals(:, j) = yprev(:, j);  % Sıralama aynı OLMALI!
        end
        
        % Giriş vektörünü oluştur
        x = dlarray([uvals, yvals]);
        
        % Gizli katmanları ekle
        for h = 1:numel(W_hidden)
            w_h = dlarray(W_hidden{h});
            x = [x, g(x * w_h)];
        end
        
        % Çıkışı hesapla
        w_o_dl = dlarray(w_o);
        y = x * w_o_dl;
        Y(:, t) = y;
        
        % yprev'i güncelle: yeni tahmini başa ekle, en eskiyi at
        if ny > 1
            yprev = [extractdata(y), yprev(:, 1:ny-1)];
        else
            yprev = extractdata(y);
        end
    end
end

function [w_o,mse,info] = trainOutputLayer_Trajectory(X0,U,T,w_o,W_hidden,g,config)
    w_o = dlarray(w_o);
    X0 = dlarray(X0); U = dlarray(U); T = dlarray(T);
    avgG=[]; avgGSq=[]; it=0;
    maxEpochs = config.model.max_epochs_output;
    loss_hist = zeros(maxEpochs,1);
    bestLoss = inf;
    epochsSinceBest = 0;
    plateauEpoch = NaN;
    minDelta = config.model.plateau_min_delta;
    patience = config.model.plateau_patience;

    for ep=1:maxEpochs
        it = it+1;
        [L,grad] = dlfeval(@loss_output_traj, w_o, X0, U, T, W_hidden, g, config);
        [w_o, avgG, avgGSq] = adamupdate(w_o, grad, avgG, avgGSq, it, config.model.eta_output);
        lossVal = gather(extractdata(L));
        loss_hist(ep) = lossVal;
        if bestLoss - lossVal > minDelta
            bestLoss = lossVal;
            epochsSinceBest = 0;
        else
            epochsSinceBest = epochsSinceBest + 1;
            if epochsSinceBest >= patience
                plateauEpoch = ep;
                break;
            end
        end
    end
    w_o = extractdata(w_o);
    epochs_run = ep;
    loss_hist = loss_hist(1:epochs_run);
    info = struct('epochs_run', epochs_run, 'plateau_epoch', plateauEpoch, 'loss_history', loss_hist);
    Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config);
    % use l2loss with explicit DataFormat
    Yvec = reshape(Y,1,[]);
    Tvec = reshape(T,1,[]);
    mse = gather(extractdata(l2loss(Yvec, Tvec, 'DataFormat', 'CB')));
end

function [L,grad] = loss_output_traj(w, X0, U, T, W_hidden, g, config)
    Y = forwardModelTrajectory(X0, U, W_hidden, g, w, config);
    Yvec = reshape(Y,1,[]);
    Tvec = reshape(T,1,[]);
    L = l2loss(Yvec, Tvec, 'DataFormat', 'CB');
    grad = dlgradient(L, w);
end

function [Utr, Ytr, Uva, Yva] = loadDataByConfig_min(config)
    switch lower(config.data.source)
        case 'twotankdata'
            load twotankdata.mat; % must contain u,y
            u = u(:); y = y(:);
            w = config.data.twotank.warmup_samples;
            u = u(w+1:end); y = y(w+1:end);
            % optional filter
            if isfield(config.data.twotank,'filter_cutoff') && config.data.twotank.filter_cutoff>0
                fc = config.data.twotank.filter_cutoff; Ts = config.data.twotank.sampling_time;
                a = 2*pi*fc*Ts / (1 + 2*pi*fc*Ts);
                uf = zeros(size(u)); yf = zeros(size(y)); uf(1)=u(1); yf(1)=y(1);
                for k=2:length(u)
                    uf(k) = a*u(k) + (1-a)*uf(k-1);
                    yf(k) = a*y(k) + (1-a)*yf(k-1);
                end
                u = uf; y = yf;
            end
        otherwise
            error('Unknown data source');
    end
    N = length(u); Ntr = floor(config.data.train_ratio * N);
    Utr = u(1:Ntr); Ytr = y(1:Ntr); Uva = u(Ntr+1:end); Yva = y(Ntr+1:end);
end

function [Utr,Ytr,Uva,Yva,stats] = normalizeData_min(method,Utr,Ytr,Uva,Yva)
    switch lower(method)
        case 'zscore'
            stats.u_mu = mean(Utr); stats.u_std = std(Utr)+eps;
            stats.y_mu = mean(Ytr); stats.y_std = std(Ytr)+eps;
            Utr = (Utr - stats.u_mu)/stats.u_std; Uva = (Uva - stats.u_mu)/stats.u_std;
            Ytr = (Ytr - stats.y_mu)/stats.y_std; Yva = (Yva - stats.y_mu)/stats.y_std;
        otherwise
            error('Unknown normalization');
    end
end

function maxLag = getMaxLagFromRegressors(regressors)
    ulags = regressors.u(:)';
    ylags = regressors.y(:)';
    maxLag = 0;
    posULags = ulags(ulags>0);
    if ~isempty(posULags)
        maxLag = max(maxLag, max(posULags));
    end
    if ~isempty(ylags)
        maxLag = max(maxLag, max(ylags));
    end
end

function [X0, Useq, Tseq] = createTrajectoryDataset(U, Y, config, N)
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)';
    maxLag = 0; if ~isempty(ulags(ulags>0)); maxLag = max(maxLag, max(ulags(ulags>0))); end
    if ~isempty(ylags); maxLag = max(maxLag, max(ylags)); end
    Ns = length(Y) - N - maxLag; if Ns<1; error('Not enough data'); end
    nu = numel(ulags); ny = numel(ylags);
    X0 = zeros(Ns, nu+ny); Useq = zeros(Ns, N); 
    Tseq = zeros(Ns, N);
    for idx=1:Ns
        i = idx + maxLag;
        row = zeros(1,nu+ny);
        for j=1:nu; L=ulags(j); if L==0; row(j)=U(i); else row(j)=U(i+1-L); end; end
        for j=1:ny; L=ylags(j); row(nu+j)=Y(i+1-L); end
        X0(idx,:) = row; Useq(idx,:) = U(i+1:i+N)'; Tseq(idx,:) = Y(i+1:i+N)';
    end
end

function Yhat = recursivePredictFullSeries(U, Y, W_hidden, w_o, g, config)
    N = length(Y); Yhat = zeros(N,1); if N>=1; Yhat(1)=Y(1); end
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)'; nu=numel(ulags); ny=numel(ylags);
    for k=2:N
        uvals=zeros(nu,1); for j=1:nu; L=ulags(j); if L==0; uvals(j)=U(k); else idx=k-L; if idx>=1; uvals(j)=U(idx); else uvals(j)=0; end; end; end
        yvals=zeros(ny,1); for j=1:ny; L=ylags(j); idx=k-L; if idx>=1; yvals(j)=Yhat(idx); else yvals(j)=0; end; end
        x = [uvals(:)', yvals(:)']; for h=1:numel(W_hidden); x=[x, g(x*W_hidden{h})]; end
        Yhat(k)= x * w_o;
    end
end

function fit = fitPercent(y, yhat)
    fit = 100 * (1 - norm(y - yhat) / norm(y - mean(y)));
end

function logFilePath = writeParameterLog(config, logInfo)
    timestampStr = datestr(now,'yyyy-mm-dd HH:MM:SS');
    scriptFullPath = mfilename('fullpath');
    if isempty(scriptFullPath)
        scriptDir = pwd;
        scriptBase = 'CCNN_Npred';
    else
        [scriptDir, scriptBase] = fileparts(scriptFullPath);
    end

    descriptor = sprintf('eta%.3g-%.3g_ep%d-%d_hid%d_reg%d', ...
        logInfo.eta_output, logInfo.eta_candidate, ...
        logInfo.max_epochs_output, logInfo.max_epochs_candidate, ...
        logInfo.hidden_units, logInfo.regressor_count);
    descriptor = strrep(descriptor,'.','p');
    descriptor = regexprep(descriptor,'[^A-Za-z0-9_-]','_');
    if numel(descriptor) > 64
        descriptor = sprintf('log_%s', regexprep(scriptBase,'[^A-Za-z0-9_-]','_'));
    end

    fileStamp = datestr(now,'yyyymmdd_HHMMSS');
    runFolderName = sprintf('%s_%s', descriptor, fileStamp);
    runFolderPath = fullfile(scriptDir, runFolderName);
    if exist(runFolderPath,'dir') == 0
        [mkStatus, mkMsg] = mkdir(runFolderPath);
        if ~mkStatus
            warning('Could not create log folder %s (%s). Falling back to script directory.', runFolderPath, mkMsg);
            runFolderPath = scriptDir;
        end
    end

    runFolderDisplay = runFolderName;
    if strcmp(runFolderPath, scriptDir)
        runFolderDisplay = '[scriptDir]';
    end

    logFileName = sprintf('%s.log', runFolderName);
    logFilePath = fullfile(runFolderPath, logFileName);
    fid = fopen(logFilePath,'w');
    if fid == -1
        warning('Could not create parameter log at %s', logFilePath);
        logFilePath = '';
        return;
    end

    fprintf(fid, 'CCNN Parameter Log\n');
    fprintf(fid, 'Created      : %s\n', timestampStr);
    fprintf(fid, 'Script       : %s.m\n', scriptBase);
    fprintf(fid, 'Run folder   : %s\n\n', runFolderDisplay);

    summaryLine = sprintf('eta_out=%.4f | eta_cand=%.4f | output_epochs=%d/%d | hidden=%d/%d | regressors=%d | cand_runs=%d', ...
        logInfo.eta_output, logInfo.eta_candidate, ...
        logInfo.output_epochs_used, logInfo.max_epochs_output, ...
        logInfo.hidden_units, logInfo.max_hidden_units, ...
        logInfo.regressor_count, logInfo.candidate_runs);
    fprintf(fid, '%s\n', summaryLine);
    fprintf(fid, 'Output plateau epoch : %s\n', formatPlateauValue(logInfo.output_plateau_epoch));

    candEpochStr = formatArrayField(logInfo.candidate_epochs_used);
    candPlateauStr = formatArrayField(logInfo.candidate_plateau_epochs);
    fprintf(fid, 'Candidate epochs (per unit)  : %s\n', candEpochStr);
    fprintf(fid, 'Candidate plateau epochs      : %s\n', candPlateauStr);

    fprintf(fid, 'N-step horizon : %d\n', logInfo.n_steps);
    fprintf(fid, 'Train MSE       : %.6g\n', logInfo.train_mse);
    fprintf(fid, 'Train Fit (%%)   : %.2f\n', logInfo.fit_train);
    fprintf(fid, 'Val   Fit (%%)   : %.2f\n\n', logInfo.fit_val);

    fprintf(fid, 'Regressors.u : %s\n', mat2str(logInfo.regressors_u));
    fprintf(fid, 'Regressors.y : %s\n', mat2str(logInfo.regressors_y));
    fprintf(fid, 'Norm method  : %s\n', config.norm_method);
    fprintf(fid, 'Target MSE   : %.6g\n', config.model.target_mse);
    fprintf(fid, 'Plateau min delta : %.3g\n', logInfo.plateau_min_delta);
    fprintf(fid, 'Plateau patience  : %d\n', logInfo.plateau_patience);

    fclose(fid);
end

function outStr = formatArrayField(values)
    if isempty(values)
        outStr = '[]';
    else
        outStr = mat2str(values);
    end
end

function outStr = formatPlateauValue(val)
    if isempty(val) || isnan(val)
        outStr = 'none';
    else
        outStr = num2str(val);
    end
end

function savedPaths = saveFitFigures(logFilePath, figMap)
    savedPaths = {};
    if nargin < 2 || isempty(figMap)
        return;
    end

    if isempty(logFilePath)
        scriptFullPath = mfilename('fullpath');
        if isempty(scriptFullPath)
            targetDir = pwd;
            baseName = sprintf('CCNN_Npred_%s', datestr(now,'yyyymmdd_HHMMSS'));
        else
            [targetDir, scriptBase] = fileparts(scriptFullPath);
            baseName = sprintf('%s_%s', scriptBase, datestr(now,'yyyymmdd_HHMMSS'));
        end
    else
        [targetDir, baseName] = fileparts(logFilePath);
    end

    labels = fieldnames(figMap);
    for k = 1:numel(labels)
        figHandle = figMap.(labels{k});
        if isempty(figHandle) || ~ishandle(figHandle)
            continue;
        end
        cleanLabel = lower(regexprep(labels{k},'[^A-Za-z0-9]',''));
        if isempty(cleanLabel)
            cleanLabel = sprintf('fig%d', k);
        end
        fileName = sprintf('%s_%s_fit.png', baseName, cleanLabel);
        filePath = fullfile(targetDir, fileName);
        try
            exportgraphics(figHandle, filePath, 'Resolution', 150);
        catch
            try
                saveas(figHandle, filePath);
            catch
                warning('Could not save figure labeled %s', labels{k});
                continue;
            end
        end
        savedPaths{end+1,1} = filePath; %#ok<AGROW>
    end
end

function appendFigureInfoToLog(logFilePath, savedPaths)
    if isempty(logFilePath) || isempty(savedPaths)
        return;
    end
    fid = fopen(logFilePath, 'a');
    if fid == -1
        warning('Could not append figure info to %s', logFilePath);
        return;
    end
    fprintf(fid, '\nSaved figure files:\n');
    [logDir, ~, ~] = fileparts(logFilePath);
    for i = 1:numel(savedPaths)
        relPath = savedPaths{i};
        if isstring(relPath)
            relPath = relPath{1};
        end
        prefix = [logDir filesep];
        if strncmp(relPath, prefix, numel(prefix))
            relPath = relPath(numel(prefix)+1:end);
        end
        fprintf(fid, ' - %s\n', relPath);
    end
    fclose(fid);
end
