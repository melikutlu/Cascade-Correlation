% CCNN_Npred v0.2
% v0.1'den farkı: Her yeni hidden unit eklendiğinde mevcut çıkış ağırlıkları
% (w_o_frozen) DONDURULUR; sadece yeni hidden unit'e ait tek scalar çıkış
% ağırlığı (w_new) eğitilir.
% Stage-1 hâlâ tüm çıkış ağırlıklarını eğitir (gizli birim yok).

clear; clc; close all; rng(0);

% ------------------
% CONFIG
% ------------------
config = struct();
config.data.source = 'dryer2';
config.data.dryer2.sampling_time = 0.08; % s

config.data.train_ratio = 0.5;
config.data.val_ratio = 0.5;

config.norm_method = 'ZScore';

config.prediction.n_steps = 20; % default N-step horizon (override auto when disabled)
config.prediction.auto_full_horizon = false; % set true to span full usable data length

% regressors (user can change)
config.regressors.u = [1,2,3,4]; % u(t-1)..u(t-4)  (dryer2'de dead time var)
config.regressors.y = [1,2,3,4]; % y(t-1)..y(t-4)
config.regressors.include_bias = false;

% model / training
config.model.activation = 'tanh';
config.model.max_hidden_units = 10;
config.model.force_hidden_growth = false;
config.model.target_mse = 5e-4;
config.model.min_mse_improvement = 1e-6;

config.model.max_epochs_output    = 500;
config.model.eta_output           = 0.005;
config.model.max_epochs_candidate = 300;
config.model.eta_candidate        = 0.003;
config.model.plateau_min_delta    = 0;

config.model.moving_avg_window = 20;
config.model.use_plateau_stop  = false;

config.training = struct();
config.training.batch_size_output    = 32;
config.training.batch_size_candidate = 32;
config.training.candidate_pool_size  = 5;
config.training.use_parfor_pool      = true;

% ------------------
% DATA
% ------------------
[Utr_raw, Ytr_raw, Uva_raw, Yva_raw] = loadDataByConfig_min(config);
[Utr, Ytr, Uva, Yva, norm_stats] = normalizeData_min(config.norm_method, Utr_raw, Ytr_raw, Uva_raw, Yva_raw);

if isfield(config.prediction, 'auto_full_horizon') && config.prediction.auto_full_horizon
    maxLag = getMaxLagFromRegressors(config.regressors);
    maxStepsTr = numel(Ytr) - maxLag;
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

% Stage 1: train output weights only (N-step MSE) — tüm ağırlıklar serbest
[w_o, current_mse, outputTrainInfo] = trainOutputLayer_Trajectory(X0_tr, Utr_seq, Ttr_seq, w_o, W_hidden, g, config);
fprintf('Stage-1 Train MSE: %.6g\n', current_mse);

if ~isnan(outputTrainInfo.plateau_epoch)
    fprintf('Output layer plateau at epoch %d (ran %d/%d epochs).\n', ...
        outputTrainInfo.plateau_epoch, outputTrainInfo.epochs_run, config.model.max_epochs_output);
else
    fprintf('Output layer used %d/%d epochs (no plateau).\n', ...
        outputTrainInfo.epochs_run, config.model.max_epochs_output);
end

if isfield(outputTrainInfo,'stop_by_moving_avg')
    fprintf('Output train stopped by moving-avg: %d\n', double(outputTrainInfo.stop_by_moving_avg));
end

mse_hist = current_mse;
candidateEpochHistory = [];
candidatePlateauHistory = [];
lossPlotHandle = [];
lossFigHandle = [];
[lossPlotHandle, lossFigHandle] = updateLossFigure(lossPlotHandle, lossFigHandle, mse_hist);

% Greedy growth with candidate pool
% v0.2: mevcut w_o DONDURULUR, sadece yeni scalar w_new eğitilir
while numel(W_hidden) < config.model.max_hidden_units
    if ~config.model.force_hidden_growth && current_mse <= config.model.target_mse
        break;
    end

    h = numel(W_hidden) + 1;
    poolSize = max(1, round(config.training.candidate_pool_size));
    useParforPool = config.training.use_parfor_pool && license('test','Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
    fprintf('\nTraining candidate pool for hidden #%d (pool=%d, parfor=%d)\n', h, poolSize, double(useParforPool));

    candWeights = cell(poolSize,1);
    candMetrics = -inf(poolSize,1);
    candInfos   = cell(poolSize,1);

    if useParforPool
        parfor p = 1:poolSize
            [tmp_w, tmp_metric, tmp_info] = trainCandidateUnit_Corr(X0_tr, Utr_seq, Ttr_seq, W_hidden, w_o, g, config);
            candWeights{p} = tmp_w;
            candMetrics(p) = tmp_metric;
            candInfos{p}   = tmp_info;
        end
    else
        for p = 1:poolSize
            [tmp_w, tmp_metric, tmp_info] = trainCandidateUnit_Corr(X0_tr, Utr_seq, Ttr_seq, W_hidden, w_o, g, config);
            candWeights{p} = tmp_w;
            candMetrics(p) = tmp_metric;
            candInfos{p}   = tmp_info;
        end
    end

    [bestCandMetric, bestIdx] = max(candMetrics);
    bestCandW    = candWeights{bestIdx};
    bestCandInfo = candInfos{bestIdx};

    candidateEpochHistory(end+1)  = bestCandInfo.epochs_run;  %#ok<AGROW>
    candidatePlateauHistory(end+1) = bestCandInfo.plateau_epoch; %#ok<AGROW>
    fprintf('Selected candidate for #%d | score: %.6g\n', h, bestCandMetric);
    if ~isnan(bestCandInfo.plateau_epoch)
        fprintf('Selected candidate plateau at epoch %d (ran %d/%d epochs).\n', ...
            bestCandInfo.plateau_epoch, bestCandInfo.epochs_run, config.model.max_epochs_candidate);
    else
        fprintf('Selected candidate used %d/%d epochs (no plateau).\n', ...
            bestCandInfo.epochs_run, config.model.max_epochs_candidate);
    end

    % --- v0.2 DEĞİŞİKLİK: w_o_frozen koru, sadece w_new eğit ---
    w_o_frozen = w_o;          % mevcut ağırlıklar donduruldu
    W_hidden{end+1} = bestCandW;

    prev_mse = current_mse;
    [w_new_scalar, current_mse, outputTrainInfo] = trainOutputLayer_FrozenPrev( ...
        X0_tr, Utr_seq, Ttr_seq, w_o_frozen, W_hidden, g, config);
    w_o = [w_o_frozen; w_new_scalar];   % frozen + yeni scalar birleştirildi
    % -------------------------------------------------------------

    improvement = prev_mse - current_mse;

    if ~config.model.force_hidden_growth && improvement < config.model.min_mse_improvement
        % undo
        W_hidden(end) = [];
        w_o = w_o_frozen;
        fprintf('Undo candidate #%d: improvement %.3g < threshold %.3g. Stopping growth.\n', h, improvement, config.model.min_mse_improvement);
        break;
    end

    mse_hist(end+1) = current_mse;
    if config.model.force_hidden_growth
        fprintf('Hidden=%d/%d | Train MSE=%.6g | improvement=%.3g | force=ON\n', ...
            numel(W_hidden), config.model.max_hidden_units, current_mse, improvement);
    else
        fprintf('Hidden=%d | Train MSE=%.6g | improvement=%.3g | force=OFF\n', ...
            numel(W_hidden), current_mse, improvement);
    end
    if ~isnan(outputTrainInfo.plateau_epoch)
        fprintf('Output layer re-train plateau at epoch %d (ran %d/%d epochs).\n', ...
            outputTrainInfo.plateau_epoch, outputTrainInfo.epochs_run, config.model.max_epochs_output);
    end
    [lossPlotHandle, lossFigHandle] = updateLossFigure(lossPlotHandle, lossFigHandle, mse_hist);
end

% Full-series recursive prediction and denormalize
Yhat_tr = recursivePredictFullSeries(Utr, Ytr, W_hidden, w_o, g, config);
Yhat_va = recursivePredictFullSeries(Uva, Yva, W_hidden, w_o, g, config);

Yhat_tr = Yhat_tr(2:end) * norm_stats.y_std + norm_stats.y_mu;
Yhat_va = Yhat_va(2:end) * norm_stats.y_std + norm_stats.y_mu;

fit_tr = fitPercent(Ytr_raw(2:end), Yhat_tr);
fit_va = fitPercent(Yva_raw(2:end), Yhat_va);
rmse_tr = sqrt(mean((Ytr_raw(2:end) - Yhat_tr).^2));
rmse_va = sqrt(mean((Yva_raw(2:end) - Yhat_va).^2));
fprintf('\nTrain Fit: %.2f%% (RMSE=%.4g) | Val Fit: %.2f%% (RMSE=%.4g)\n', fit_tr, rmse_tr, fit_va, rmse_va);

% Log
logInfo = struct();
logInfo.eta_output           = config.model.eta_output;
logInfo.eta_candidate        = config.model.eta_candidate;
logInfo.max_epochs_output    = config.model.max_epochs_output;
logInfo.output_epochs_used   = outputTrainInfo.epochs_run;
logInfo.output_plateau_epoch = outputTrainInfo.plateau_epoch;
logInfo.max_epochs_candidate  = config.model.max_epochs_candidate;
logInfo.candidate_epochs_used  = candidateEpochHistory;
logInfo.candidate_plateau_epochs = candidatePlateauHistory;
logInfo.candidate_runs       = numel(candidateEpochHistory);
logInfo.plateau_min_delta    = config.model.plateau_min_delta;
logInfo.moving_avg_window    = config.model.moving_avg_window;
logInfo.hidden_units         = numel(W_hidden);
logInfo.max_hidden_units     = config.model.max_hidden_units;
logInfo.regressor_count      = numel(config.regressors.u) + numel(config.regressors.y);
logInfo.regressors_u         = config.regressors.u;
logInfo.regressors_y         = config.regressors.y;
logInfo.n_steps              = Npred;
logInfo.train_mse            = current_mse;
logInfo.fit_train            = fit_tr;
logInfo.fit_val              = fit_va;
logInfo.rmse_train           = rmse_tr;
logInfo.rmse_val             = rmse_va;
logInfo.activation           = config.model.activation;
logInfo.mse_history          = mse_hist;
logInfo.frozen_prev_weights  = true;   % v0.2 flag
if isfield(outputTrainInfo,'stop_by_moving_avg')
    logInfo.output_stop_by_mavg = double(outputTrainInfo.stop_by_moving_avg);
end
logFilePath = writeParameterLog(config, logInfo);
if ~isempty(logFilePath)
    fprintf('Parameter log saved to %s\n', logFilePath);
end

% Plots
figTrain = figure('Name','TRAIN - Full Recursive','Color','w');
plot(Ytr_raw(2:end),'k','LineWidth',1.4); hold on;
plot(Yhat_tr,'b--','LineWidth',1.2); grid on;
title(sprintf('TRAIN | Hidden=%d | Fit=%.2f%%', numel(W_hidden), fit_tr)); legend('True','CCNN');

figVal = figure('Name','VAL - Full Recursive','Color','w');
plot(Yva_raw(2:end),'k','LineWidth',1.4); hold on;
plot(Yhat_va,'r--','LineWidth',1.2); grid on;
title(sprintf('VAL | Hidden=%d | Fit=%.2f%%', numel(W_hidden), fit_va)); legend('True','CCNN');

savedFigurePaths = saveFitFigures(logFilePath, struct('train', figTrain, 'val', figVal, 'loss', lossFigHandle));
if ~isempty(savedFigurePaths) && ~isempty(logFilePath)
    appendFigureInfoToLog(logFilePath, savedFigurePaths);
end

% ------------------
% LOCAL FUNCTIONS
% ------------------

% ---- v0.2 YENİ FONKSİYON: sadece yeni scalar ağırlığı eğit ----
function [w_new_scalar, mse, info] = trainOutputLayer_FrozenPrev(X0, U, T, w_o_frozen, W_hidden, g, config)
    % w_o_frozen : mevcut ağırlıklar (donmuş, normal dizi)
    % W_hidden   : son eklenen hidden dahil tam liste
    % Sadece son hidden unit'e bağlı tek scalar (w_new) eğitilir.

    w_new = dlarray(randn(1,1)*0.01);   % eğitilecek tek scalar

    X0_d = dlarray(X0);
    U_d  = dlarray(U);
    T_d  = dlarray(T);

    avgG = []; avgGSq = []; it = 0;
    maxEpochs = config.model.max_epochs_output;
    loss_hist = zeros(maxEpochs,1);
    plateauEpoch = NaN;
    minDelta = config.model.plateau_min_delta;
    window   = max(1, round(config.model.moving_avg_window));
    stopByMavg = false;

    numSamples = size(X0,1);
    batchSize  = resolveBatchSize(config, 'batch_size_output', numSamples);

    for ep = 1:maxEpochs
        batches = buildMiniBatchOrder(numSamples, batchSize);
        epochLoss = 0;
        for b = 1:numel(batches)
            idx = batches{b};
            Xb = X0_d(idx,:); Ub = U_d(idx,:); Tb = T_d(idx,:);
            it = it + 1;
            [L, grad] = dlfeval(@loss_output_frozen, w_new, w_o_frozen, Xb, Ub, Tb, W_hidden, g, config);
            [w_new, avgG, avgGSq] = adamupdate(w_new, grad, avgG, avgGSq, it, config.model.eta_output);
            epochLoss = epochLoss + gather(extractdata(L)) * (numel(idx)/numSamples);
        end

        loss_hist(ep) = epochLoss;
        if config.model.use_plateau_stop && ep > window
            mavg = mean(loss_hist(ep-window:ep-1));
            if mavg - epochLoss <= minDelta
                plateauEpoch = ep;
                stopByMavg   = true;
                break;
            end
        end
    end

    w_new_scalar = gather(extractdata(w_new));
    w_o_full = [w_o_frozen; w_new_scalar];
    epochs_run = ep;
    loss_hist  = loss_hist(1:epochs_run);
    info = struct('epochs_run', epochs_run, 'plateau_epoch', plateauEpoch, ...
                  'loss_history', loss_hist, 'stop_by_moving_avg', stopByMavg);

    % MSE hesapla
    Y = forwardModelTrajectory(X0_d, U_d, W_hidden, g, w_o_full, config);
    Yvec = reshape(Y,1,[]); Tvec = reshape(T_d,1,[]);
    mse  = gather(extractdata(l2loss(Yvec, Tvec, 'DataFormat', 'CB')));
end

function [L, grad] = loss_output_frozen(w_new, w_o_frozen, X0, U, T, W_hidden, g, config)
    % w_o_frozen normal dizi olarak gelir → sabit (gradyan akımı yok)
    % w_new      dlarray → gradyan burada hesaplanır
    w_o = [w_o_frozen; w_new];   % birleştir
    Y   = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config);
    Yvec = reshape(Y, 1, []);
    Tvec = reshape(T, 1, []);
    L    = l2loss(Yvec, Tvec, 'DataFormat', 'CB');
    grad = dlgradient(L, w_new);  % sadece w_new'e göre türev
end
% -----------------------------------------------------------------

function [w_h, best_metric, info] = trainCandidateUnit_Corr(X0,U,T,W_hidden,w_o,g,config)
    d   = size(X0,2) + numel(W_hidden);
    w_h = dlarray(randn(d,1)*0.01);

    X0_d = dlarray(X0);
    U_d  = dlarray(U);
    T_d  = dlarray(T);
    w_o_d = dlarray(w_o);

    avgG=[]; avgGSq=[]; it=0;
    best_metric = 0;
    best_w      = extractdata(w_h);
    maxEpochs   = config.model.max_epochs_candidate;
    metric_hist = zeros(maxEpochs,1);
    plateauEpoch = NaN;
    minDelta = config.model.plateau_min_delta;
    window   = max(1, round(config.model.moving_avg_window));

    numSamples = size(X0,1);
    batchSize  = resolveBatchSize(config, 'batch_size_candidate', numSamples);

    for ep = 1:maxEpochs
        batches = buildMiniBatchOrder(numSamples, batchSize);
        for b = 1:numel(batches)
            idx = batches{b};
            Xb = X0_d(idx,:); Ub = U_d(idx,:); Tb = T_d(idx,:);
            it = it + 1;
            [loss, ~, grad] = dlfeval(@loss_candidate_corr, w_h, Xb, Ub, Tb, W_hidden, w_o_d, g, config);
            [w_h, avgG, avgGSq] = adamupdate(w_h, grad, avgG, avgGSq, it, config.model.eta_candidate);
        end

        metricVal = evaluateCandidateMetric(w_h, X0_d, U_d, T_d, W_hidden, w_o_d, g, config);
        metric_hist(ep) = metricVal;

        if metricVal > best_metric
            best_metric = metricVal;
            best_w      = extractdata(w_h);
        end

        if config.model.use_plateau_stop && ep > window
            mavg = mean(metric_hist(ep-window:ep-1));
            if metricVal - mavg <= minDelta
                plateauEpoch = ep;
                break;
            end
        end
    end

    w_h        = best_w;
    epochs_run = ep;
    metric_hist = metric_hist(1:epochs_run);
    info = struct('epochs_run', epochs_run, 'plateau_epoch', plateauEpoch, 'metric_history', metric_hist);
end

function metricVal = evaluateCandidateMetric(w_h, X0, U, T, W_hidden, w_o, g, config)
    metric    = candidateCorrelationMetric(w_h, X0, U, T, W_hidden, w_o, g, config);
    metricVal = gather(extractdata(metric));
end

function metric = candidateCorrelationMetric(w_h, X0, U, T, W_hidden, w_o, g, config)
    Y_model = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config);
    R = T - Y_model;

    ulags = config.regressors.u(:)';
    ylags = config.regressors.y(:)';
    nu = numel(ulags); ny = numel(ylags);
    M  = size(X0,1);  N  = size(U,2);
    v  = dlarray(zeros(M,N));

    maxLagY = max(ylags);
    yhist   = zeros(M, maxLagY);
    for j = 1:ny
        yhist(:, ylags(j)) = X0(:, nu+j);
    end

    for t = 1:N
        uvals = zeros(M, nu);
        for j = 1:nu
            L = ulags(j);
            if L == 0
                uvals(:,j) = U(:,t);
            else
                idx = t - L;
                if idx >= 1; uvals(:,j) = U(:,idx); else; uvals(:,j) = X0(:,j); end
            end
        end
        yvals = zeros(M, ny);
        for j = 1:ny
            yvals(:,j) = yhist(:, ylags(j));
        end
        x_t = [uvals, yvals];
        for hh = 1:numel(W_hidden)
            x_t = [x_t, g(x_t * W_hidden{hh})];
        end
        x_t    = dlarray(x_t);
        v(:,t) = g(x_t * w_h);
        y_t    = Y_model(:,t);
        yhist  = [y_t, yhist(:, 1:maxLagY-1)];
    end

    r_vec  = reshape(R, [], 1);
    v_vec  = reshape(v, [], 1);
    r_c    = r_vec - mean(r_vec);
    v_c    = (v_vec - mean(v_vec)) + 0.1 * sign(v_vec - mean(v_vec));
    cov_vr = sum(v_c .* r_c);
    metric = abs(cov_vr);
end

function [L, metric, grad] = loss_candidate_corr(w_h, X0, U, T, W_hidden, w_o, g, config)
    metric = candidateCorrelationMetric(w_h, X0, U, T, W_hidden, w_o, g, config);
    L      = -(metric^2);
    grad   = dlgradient(L, w_h);
end

function Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config)
    M = size(X0,1); N = size(U,2);
    Y = dlarray(zeros(M,N));

    ulags   = config.regressors.u(:)'; ylags = config.regressors.y(:)';
    nu      = numel(ulags); ny = numel(ylags);
    maxLagY = max(ylags);
    yhist   = zeros(M, maxLagY);
    for j = 1:ny
        yhist(:, ylags(j)) = X0(:, nu+j);
    end

    for t = 1:N
        uvals = zeros(M, nu);
        for j = 1:nu
            L = ulags(j);
            if L == 0
                uvals(:,j) = U(:,t);
            else
                idx = t - L;
                if idx >= 1; uvals(:,j) = U(:,idx); else; uvals(:,j) = X0(:,j); end
            end
        end
        yvals = zeros(M, ny);
        for j = 1:ny
            yvals(:,j) = yhist(:, ylags(j));
        end
        x = [uvals, yvals];
        for h = 1:numel(W_hidden)
            x = [x, g(x * W_hidden{h})];
        end
        y      = x * w_o;
        Y(:,t) = y;
        yhist  = [y, yhist(:, 1:maxLagY-1)];
    end
end

function [w_o, mse, info] = trainOutputLayer_Trajectory(X0, U, T, w_o, W_hidden, g, config)
    w_o = dlarray(w_o);
    X0  = dlarray(X0); U = dlarray(U); T = dlarray(T);
    avgG=[]; avgGSq=[]; it=0;
    maxEpochs = config.model.max_epochs_output;
    loss_hist  = zeros(maxEpochs,1);
    plateauEpoch = NaN;
    minDelta = config.model.plateau_min_delta;
    window   = max(1, round(config.model.moving_avg_window));
    stopByMavg = false;

    numSamples = size(X0,1);
    batchSize  = resolveBatchSize(config, 'batch_size_output', numSamples);

    for ep = 1:maxEpochs
        batches   = buildMiniBatchOrder(numSamples, batchSize);
        epochLoss = 0;
        for b = 1:numel(batches)
            idx = batches{b};
            Xb = X0(idx,:); Ub = U(idx,:); Tb = T(idx,:);
            it = it + 1;
            [L, grad] = dlfeval(@loss_output_traj, w_o, Xb, Ub, Tb, W_hidden, g, config);
            [w_o, avgG, avgGSq] = adamupdate(w_o, grad, avgG, avgGSq, it, config.model.eta_output);
            epochLoss = epochLoss + gather(extractdata(L)) * (numel(idx)/numSamples);
        end
        loss_hist(ep) = epochLoss;
        if config.model.use_plateau_stop && ep > window
            mavg = mean(loss_hist(ep-window:ep-1));
            if mavg - epochLoss <= minDelta
                plateauEpoch = ep;
                stopByMavg   = true;
                break;
            end
        end
    end
    w_o        = extractdata(w_o);
    epochs_run = ep;
    loss_hist  = loss_hist(1:epochs_run);
    info = struct('epochs_run', epochs_run, 'plateau_epoch', plateauEpoch, ...
                  'loss_history', loss_hist, 'stop_by_moving_avg', stopByMavg);
    Y    = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config);
    Yvec = reshape(Y,1,[]); Tvec = reshape(T,1,[]);
    mse  = gather(extractdata(l2loss(Yvec, Tvec, 'DataFormat', 'CB')));
end

function [L, grad] = loss_output_traj(w, X0, U, T, W_hidden, g, config)
    Y    = forwardModelTrajectory(X0, U, W_hidden, g, w, config);
    Yvec = reshape(Y,1,[]); Tvec = reshape(T,1,[]);
    L    = l2loss(Yvec, Tvec, 'DataFormat', 'CB');
    grad = dlgradient(L, w);
end

function [Utr, Ytr, Uva, Yva] = loadDataByConfig_min(config)
    switch lower(config.data.source)
        case 'twotankdata'
            load twotankdata.mat;
            u = u(:); y = y(:);
            w = config.data.twotank.warmup_samples;
            u = u(w+1:end); y = y(w+1:end);
            if isfield(config.data.twotank,'filter_cutoff') && config.data.twotank.filter_cutoff>0
                fc = config.data.twotank.filter_cutoff; Ts = config.data.twotank.sampling_time;
                a  = 2*pi*fc*Ts / (1 + 2*pi*fc*Ts);
                uf = zeros(size(u)); yf = zeros(size(y)); uf(1)=u(1); yf(1)=y(1);
                for k = 2:length(u)
                    uf(k) = a*u(k) + (1-a)*uf(k-1);
                    yf(k) = a*y(k) + (1-a)*yf(k-1);
                end
                u = uf; y = yf;
            end
            N = length(u); Ntr = floor(config.data.train_ratio * N);
            Utr = u(1:Ntr); Ytr = y(1:Ntr); Uva = u(Ntr+1:end); Yva = y(Ntr+1:end);
        case 'dryer2'
            load dryer2;
            Ts    = config.data.dryer2.sampling_time;
            z_full = iddata(y2(:), u2(:), Ts);
            N_total   = length(z_full.y);
            train_end = floor(config.data.train_ratio * N_total);
            val_end   = train_end + floor(config.data.val_ratio * N_total);
            z1f = detrend(z_full(1:train_end));
            z2f = detrend(z_full(train_end+1:val_end));
            Utr = z1f.u; Ytr = z1f.y;
            Uva = z2f.u; Yva = z2f.y;
        otherwise
            error('Unknown data source: %s', config.data.source);
    end
end

function [Utr, Ytr, Uva, Yva, stats] = normalizeData_min(method, Utr, Ytr, Uva, Yva)
    switch lower(method)
        case 'zscore'
            stats.u_mu  = mean(Utr); stats.u_std = std(Utr)+eps;
            stats.y_mu  = mean(Ytr); stats.y_std = std(Ytr)+eps;
            Utr = (Utr - stats.u_mu)/stats.u_std; Uva = (Uva - stats.u_mu)/stats.u_std;
            Ytr = (Ytr - stats.y_mu)/stats.y_std; Yva = (Yva - stats.y_mu)/stats.y_std;
        otherwise
            error('Unknown normalization');
    end
end

function maxLag = getMaxLagFromRegressors(regressors)
    ulags   = regressors.u(:)'; ylags = regressors.y(:)';
    maxLag  = 0;
    posULags = ulags(ulags>0);
    if ~isempty(posULags); maxLag = max(maxLag, max(posULags)); end
    if ~isempty(ylags);    maxLag = max(maxLag, max(ylags));    end
end

function [X0, Useq, Tseq] = createTrajectoryDataset(U, Y, config, N)
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)';
    maxLag = 0;
    if ~isempty(ulags(ulags>0)); maxLag = max(maxLag, max(ulags(ulags>0))); end
    if ~isempty(ylags);          maxLag = max(maxLag, max(ylags));          end
    Ns = length(Y) - N - maxLag + 1;
    if Ns < 1; error('Not enough data'); end
    nu = numel(ulags); ny = numel(ylags);
    X0 = zeros(Ns, nu+ny); Useq = zeros(Ns, N); Tseq = zeros(Ns, N);
    for idx = 1:Ns
        i   = idx + maxLag - 1;
        row = zeros(1, nu+ny);
        for j = 1:nu; L=ulags(j); if L==0; row(j)=U(i); else; row(j)=U(i+1-L); end; end
        for j = 1:ny; L=ylags(j); row(nu+j)=Y(i+1-L); end
        X0(idx,:)   = row;
        Useq(idx,:) = U(i+1:i+N)';
        Tseq(idx,:) = Y(i+1:i+N)';
    end
end

function Yhat = recursivePredictFullSeries(U, Y, W_hidden, w_o, g, config)
    N = length(Y); Yhat = zeros(N,1); if N>=1; Yhat(1)=Y(1); end
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)';
    nu = numel(ulags); ny = numel(ylags);
    for k = 2:N
        uvals = zeros(nu,1);
        for j = 1:nu; L=ulags(j); if L==0; uvals(j)=U(k); else; idx=k-L; if idx>=1; uvals(j)=U(idx); else; uvals(j)=0; end; end; end
        yvals = zeros(ny,1);
        for j = 1:ny; L=ylags(j); idx=k-L; if idx>=1; yvals(j)=Yhat(idx); else; yvals(j)=0; end; end
        x = [uvals(:)', yvals(:)'];
        for h = 1:numel(W_hidden); x = [x, g(x*W_hidden{h})]; end
        Yhat(k) = x * w_o;
    end
end

function fit = fitPercent(y, yhat)
    fit = 100 * (1 - norm(y - yhat) / norm(y - mean(y)));
end

function batchSize = resolveBatchSize(config, fieldName, numSamples)
    batchSize = numSamples;
    if isfield(config,'training') && isfield(config.training, fieldName)
        c = config.training.(fieldName);
        if isnumeric(c) && c > 0
            batchSize = min(numSamples, max(1, round(c)));
        end
    end
end

function batches = buildMiniBatchOrder(numSamples, batchSize)
    if batchSize >= numSamples; batches = {1:numSamples}; return; end
    order     = randperm(numSamples);
    numBatches = ceil(numSamples / batchSize);
    batches   = cell(numBatches,1);
    for k = 1:numBatches
        s = (k-1)*batchSize+1; e = min(k*batchSize, numSamples);
        batches{k} = order(s:e);
    end
end

function [plotHandle, figHandle] = updateLossFigure(plotHandle, figHandle, mse_hist)
    xVals = 0:numel(mse_hist)-1;
    if isempty(figHandle) || ~ishandle(figHandle)
        figHandle = figure('Name','Train MSE vs Hidden Units','Color','w');
    else
        figure(figHandle);
    end
    if isempty(plotHandle) || ~isvalid(plotHandle)
        clf(figHandle);
        plotHandle = plot(xVals, mse_hist, '-o', 'LineWidth', 1.4);
        grid on; xlabel('Hidden Units'); ylabel('Train MSE'); title('Train MSE vs Hidden Units');
    else
        set(plotHandle, 'XData', xVals, 'YData', mse_hist);
    end
    drawnow;
end

function logFilePath = writeParameterLog(config, logInfo)
    timestampStr   = datestr(now,'yyyy-mm-dd HH:MM:SS');
    scriptFullPath = mfilename('fullpath');
    if isempty(scriptFullPath)
        scriptDir  = pwd; scriptBase = 'CCNN_Npred';
    else
        [scriptDir, scriptBase] = fileparts(scriptFullPath);
    end

    fitTrStr   = strrep(sprintf('%.1f', logInfo.fit_train), '.', 'p');
    fitVaStr   = strrep(sprintf('%.1f', logInfo.fit_val),   '.', 'p');
    descriptor = sprintf('fitTr%s_fitVa%s', fitTrStr, fitVaStr);
    descriptor = regexprep(strrep(descriptor,'.','p'), '[^A-Za-z0-9_-]', '_');
    if numel(descriptor) > 64
        descriptor = sprintf('log_%s', regexprep(scriptBase,'[^A-Za-z0-9_-]','_'));
    end

    fileStamp     = datestr(now,'yyyymmdd_HHMMSS');
    runFolderName = sprintf('%s_%s', descriptor, fileStamp);
    runFolderPath = fullfile(scriptDir, runFolderName);
    if ~exist(runFolderPath,'dir')
        [ok, msg] = mkdir(runFolderPath);
        if ~ok
            warning('Could not create log folder %s (%s).', runFolderPath, msg);
            runFolderPath = scriptDir;
        end
    end

    logFileName = sprintf('%s.log', runFolderName);
    logFilePath = fullfile(runFolderPath, logFileName);
    fid = fopen(logFilePath,'w');
    if fid == -1
        warning('Could not create parameter log at %s', logFilePath);
        logFilePath = ''; return;
    end

    fprintf(fid, 'CCNN Parameter Log (v0.2 - frozen prev weights)\n');
    fprintf(fid, 'Created      : %s\n', timestampStr);
    fprintf(fid, 'Script       : %s.m\n', scriptBase);
    fprintf(fid, 'Run folder   : %s\n\n', runFolderName);

    summaryLine = sprintf('eta_out=%.4f | eta_cand=%.4f | output_epochs=%d/%d | hidden=%d/%d | regressors=%d | cand_runs=%d', ...
        logInfo.eta_output, logInfo.eta_candidate, ...
        logInfo.output_epochs_used, logInfo.max_epochs_output, ...
        logInfo.hidden_units, logInfo.max_hidden_units, ...
        logInfo.regressor_count, logInfo.candidate_runs);
    fprintf(fid, '%s\n', summaryLine);
    fprintf(fid, 'Frozen prev weights : true\n');
    fprintf(fid, 'Output plateau epoch : %s\n', formatPlateauValue(logInfo.output_plateau_epoch));
    if isfield(logInfo,'output_stop_by_mavg')
        fprintf(fid, 'Output stopped by moving-avg : %d\n', logInfo.output_stop_by_mavg);
    end
    fprintf(fid, 'Candidate epochs (per unit)  : %s\n', formatArrayField(logInfo.candidate_epochs_used));
    fprintf(fid, 'Candidate plateau epochs      : %s\n', formatArrayField(logInfo.candidate_plateau_epochs));
    fprintf(fid, 'N-step horizon : %d\n',  logInfo.n_steps);
    fprintf(fid, 'Train obj MSE  : %.6g\n', logInfo.train_mse);
    fprintf(fid, 'Train RMSE     : %.6g\n', logInfo.rmse_train);
    fprintf(fid, 'Val   RMSE     : %.6g\n', logInfo.rmse_val);
    fprintf(fid, 'Train Fit (%%)  : %.2f\n', logInfo.fit_train);
    fprintf(fid, 'Val   Fit (%%)  : %.2f\n\n', logInfo.fit_val);
    if isfield(logInfo,'mse_history') && ~isempty(logInfo.mse_history)
        fprintf(fid, 'MSE history (per hidden unit added): %s\n', formatArrayField(logInfo.mse_history));
    end
    fprintf(fid, 'Regressors.u : %s\n', mat2str(logInfo.regressors_u));
    fprintf(fid, 'Regressors.y : %s\n', mat2str(logInfo.regressors_y));
    fprintf(fid, 'Norm method  : %s\n', config.norm_method);
    fprintf(fid, 'Activation   : %s\n', logInfo.activation);
    fprintf(fid, 'Target MSE   : %.6g\n', config.model.target_mse);
    fprintf(fid, 'Plateau min delta  : %.3g\n', logInfo.plateau_min_delta);
    fprintf(fid, 'Moving avg window  : %d\n', logInfo.moving_avg_window);
    fclose(fid);
end

function outStr = formatArrayField(values)
    if isempty(values); outStr = '[]'; else; outStr = mat2str(values); end
end

function outStr = formatPlateauValue(val)
    if isempty(val) || isnan(val); outStr = 'none'; else; outStr = num2str(val); end
end

function savedPaths = saveFitFigures(logFilePath, figMap)
    savedPaths = {};
    if nargin < 2 || isempty(figMap); return; end
    if isempty(logFilePath)
        scriptFullPath = mfilename('fullpath');
        if isempty(scriptFullPath)
            targetDir = pwd; baseName = sprintf('CCNN_Npred_%s', datestr(now,'yyyymmdd_HHMMSS'));
        else
            [targetDir, scriptBase] = fileparts(scriptFullPath);
            baseName = sprintf('%s_%s', scriptBase, datestr(now,'yyyymmdd_HHMMSS'));
        end
    else
        [targetDir, baseName] = fileparts(logFilePath);
    end
    labels = fieldnames(figMap);
    for k = 1:numel(labels)
        fh = figMap.(labels{k});
        if isempty(fh) || ~ishandle(fh); continue; end
        cleanLabel = lower(regexprep(labels{k},'[^A-Za-z0-9]',''));
        if isempty(cleanLabel); cleanLabel = sprintf('fig%d',k); end
        filePath = fullfile(targetDir, sprintf('%s_%s_fit.png', baseName, cleanLabel));
        try
            exportgraphics(fh, filePath, 'Resolution', 150);
        catch
            try; saveas(fh, filePath); catch; warning('Could not save figure %s', labels{k}); continue; end
        end
        savedPaths{end+1,1} = filePath; %#ok<AGROW>
    end
end

function appendFigureInfoToLog(logFilePath, savedPaths)
    if isempty(logFilePath) || isempty(savedPaths); return; end
    fid = fopen(logFilePath,'a');
    if fid == -1; warning('Could not append figure info to %s', logFilePath); return; end
    fprintf(fid, '\nSaved figure files:\n');
    [logDir, ~, ~] = fileparts(logFilePath);
    for i = 1:numel(savedPaths)
        relPath = savedPaths{i};
        if isstring(relPath); relPath = relPath{1}; end
        prefix = [logDir filesep];
        if strncmp(relPath, prefix, numel(prefix)); relPath = relPath(numel(prefix)+1:end); end
        fprintf(fid, ' - %s\n', relPath);
    end
    fclose(fid);
end
