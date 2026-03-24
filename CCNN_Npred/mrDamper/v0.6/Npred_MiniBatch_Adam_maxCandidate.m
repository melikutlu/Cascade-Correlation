% % CCNN_Npred.m
% CCNN where candidate units are trained to MAXIMIZE N-step residual correlation
% Model and candidate training both operate on N-step trajectory predictions.

clear; clc; close all; rng(0);

% Ensure local `function` folder is on MATLAB path (created if missing)
scriptFullPath = mfilename('fullpath');
if isempty(scriptFullPath)
    scriptDir = pwd;
else
    [scriptDir, ~] = fileparts(scriptFullPath);
end
funcFolder = fullfile(scriptDir, 'function');
if exist(funcFolder, 'dir') ~= 7
    mkdir(funcFolder);
end
addpath(funcFolder, '-begin');

% ------------------
% CONFIG
% ----------------
config = struct();
config.data.source = 'mrDamper';
config.data.dryer2.sampling_time = 0.08; % s

config.data.train_ratio = 0.5;
config.data.val_ratio = 0.5;

config.norm_method = 'zscore';

config.prediction.n_steps = 20; % default N-step horizon (override auto when disabled)
config.prediction.auto_full_horizon = false; % set true to span full usable data length

% regressors (user can change)
config.regressors.u = [1]; % example: u(t), u(t-1) (u(t) kaldır, dryer2'de dead time var)
config.regressors.y = [1]; % example: y(t-1), y(t-2)
config.regressors.include_bias = false;

% model / training
% activation options: 'tanh' (default), 'diff' (time diff of z), 'diff-tanh' (time diff then tanh)
config.model.activation = 'diff';
config.model.max_hidden_units = 10;
config.model.force_hidden_growth = true; % true: always add up to max_hidden_units
config.model.target_mse = 5e-4;  % true MSE — adjust if needed
config.model.min_mse_improvement = 1e-4; % early stop threshold


% Adam typically saturates within -300 epochs; plateau guard stops early.
config.model.max_epochs_output = 100;
config.model.eta_output = 0.005;
config.model.max_epochs_candidate = 100;
config.model.eta_candidate = 0.03;
config.model.plateau_min_delta = 0;   % stop if improvement over prev-window mean is <= this

% Moving-average plateau stop: after each epoch, compare current loss/metric
% against the mean of the previous `moving_avg_window` epochs.
% If improvement <= plateau_min_delta, training stops (plateau detected).
config.model.moving_avg_window = 20;      % number of previous epochs to average
config.model.use_plateau_stop = false;

config.training = struct();
config.training.batch_size_output = 32;     % mini-batch size for output layer updates
config.training.batch_size_candidate = 32;  % mini-batch size for candidate unit search
config.training.candidate_pool_size = 0;    % train this many candidates, pick best scored
config.training.use_parfor_pool = false ;     % true: train candidate pool with parfor (if available)

% load raw data according to config, then normalize
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

% activation selected by config: do not define `g` here

% initialize
W_hidden = {};
% X0 shape: (Ns, nWarmupSteps, nFeatures)
% Feature dimension is in the 3rd dimension
d0 = size(X0_tr,3);
w_o = randn(d0,1)*0.01;

% Stage 1: train output weights only (N-step MSE)
[w_o, ~, outputTrainInfo, lossHistoryFig] = trainOutputLayer_Trajectory(X0_tr, Utr_seq, Ttr_seq, w_o, W_hidden, config.model.activation, config, 'b');
% Full-series recursive MSE (tutarlı olması için büyüme kararları bununla alınır)
Yhat_tmp = recursivePredictFullSeries(Utr, Ytr, W_hidden, w_o, config.model.activation, config);
current_mse = mean((Ytr(2:end) - Yhat_tmp(2:end)).^2);
Yhat_tmp_raw = Yhat_tmp(2:end) * norm_stats.y_std + norm_stats.y_mu;
stage0_rmse = sqrt(mean((Ytr_raw(2:end) - Yhat_tmp_raw).^2));
stage0_fit  = fitPercent(Ytr_raw(2:end), Yhat_tmp_raw);
fprintf('--- Hidden=0 | MSE=%.6g | RMSE=%.4g | Fit=%.2f%% ---\n', current_mse, stage0_rmse, stage0_fit);

if ~isnan(outputTrainInfo.plateau_epoch)
    fprintf('Output layer plateau at epoch %d (ran %d/%d epochs).\n', ...
        outputTrainInfo.plateau_epoch, outputTrainInfo.epochs_run, config.model.max_epochs_output);
else
    fprintf('Output layer used %d/%d epochs (no plateau).\n', ...
        outputTrainInfo.epochs_run, config.model.max_epochs_output);
end

% print whether training stopped due to moving-average stop (if present)
if isfield(outputTrainInfo,'stop_by_moving_avg')
    fprintf('Output train stopped by moving-avg: %d\n', double(outputTrainInfo.stop_by_moving_avg));
end

mse_hist = current_mse;

candidateEpochHistory = [];
candidatePlateauHistory = [];
lossPlotHandle = [];
lossFigHandle = [];
corrPlotHandle = [];
corrFigHandle = [];
candidateCorrHistory = [];
[lossPlotHandle, lossFigHandle] = updateLossFigure(lossPlotHandle, lossFigHandle, mse_hist);

% Greedy growth with candidate pool
while numel(W_hidden) < config.model.max_hidden_units
    if ~config.model.force_hidden_growth && current_mse <= config.model.target_mse
        break;
    end

    h = numel(W_hidden) + 1;
    poolSize = max(1, round(config.training.candidate_pool_size));
    useParforPool = config.training.use_parfor_pool && license('test','Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
    fprintf('\nTraining candidate pool for hidden #%d (pool=%d, parfor=%d)\n', h, poolSize, double(useParforPool));

    % train a pool of candidates and select the best-scored one
    bestCandMetric = -inf;
    bestCandW = [];
    bestCandInfo = struct('epochs_run', 0, 'plateau_epoch', NaN, 'metric_history', []);

    candWeights = cell(poolSize,1);
    candMetrics = -inf(poolSize,1);
    candInfos = cell(poolSize,1);

    if useParforPool
        parfor p = 1:poolSize
            [tmp_w, tmp_metric, tmp_info] = trainCandidateUnit_Corr(X0_tr, Utr_seq, Ttr_seq, W_hidden, w_o, config.model.activation, config);
            candWeights{p} = tmp_w;
            candMetrics(p) = tmp_metric;
            candInfos{p} = tmp_info;
            
        end
    else
        for p = 1:poolSize
            [tmp_w, tmp_metric, tmp_info] = trainCandidateUnit_Corr(X0_tr, Utr_seq, Ttr_seq, W_hidden, w_o, config.model.activation, config);
            candWeights{p} = tmp_w;
            candMetrics(p) = tmp_metric;
            candInfos{p} = tmp_info;
        end
    end

    [bestCandMetric, bestIdx] = max(candMetrics);
    if isfinite(bestCandMetric)
        bestCandW = candWeights{bestIdx};
        bestCandInfo = candInfos{bestIdx};
    end

    w_h = bestCandW;
    cand_metric = bestCandMetric;
    candInfo = bestCandInfo;
    candidateEpochHistory(end+1) = candInfo.epochs_run; %#ok<AGROW>
    candidatePlateauHistory(end+1) = candInfo.plateau_epoch; %#ok<AGROW>
    fprintf('Selected candidate for #%d | score: %.6g\n', h, cand_metric);
    if ~isnan(candInfo.plateau_epoch)
        fprintf('Selected candidate plateau at epoch %d (ran %d/%d epochs).\n', ...
            candInfo.plateau_epoch, candInfo.epochs_run, config.model.max_epochs_candidate);
    else
        fprintf('Selected candidate used %d/%d epochs (no plateau).\n', ...
            candInfo.epochs_run, config.model.max_epochs_candidate);
    end

    % Append best candidate correlation history to a persistent figure
    if isfield(candInfo, 'metric_history') && ~isempty(candInfo.metric_history)
        candidateCorrHistory = [candidateCorrHistory; candInfo.metric_history(:)];
        [corrPlotHandle, corrFigHandle] = updateCorrelationFigure(corrPlotHandle, corrFigHandle, candidateCorrHistory);
    end

    % tentatively add candidate
    w_o_prev = w_o;
    W_hidden{end+1} = w_h;
    % Warm-start: mevcut output agirliklarini aynen koru,
    % sadece yeni candidate icin bir cikis agirligi ekle.
    w_o = [w_o_prev; dlarray(0)];

    prev_mse = current_mse;
    [w_o, ~, outputTrainInfo, lossHistoryFig] = trainOutputLayer_Trajectory(X0_tr, Utr_seq, Ttr_seq, w_o, W_hidden, config.model.activation, config, 'r');
    % Full-series recursive MSE (tutarlı olması için büyüme kararları bununla alınır)
    Yhat_tmp = recursivePredictFullSeries(Utr, Ytr, W_hidden, w_o, config.model.activation, config);
    current_mse = mean((Ytr(2:end) - Yhat_tmp(2:end)).^2);

    improvement = prev_mse - current_mse;

    if ~config.model.force_hidden_growth && improvement < config.model.min_mse_improvement
        % undo: w_o_prev'i geri yükle (retrain sonrası tüm elemanlar
        % yeni hidden unit için güncellendi, w_o(1:end-1) yanlış olur)
        W_hidden(end) = [];
        w_o = w_o_prev;
        fprintf('Undo candidate #%d: improvement %.3g < threshold %.3g. Stopping growth.\n', h, improvement, config.model.min_mse_improvement);

        break;
      
    end

    mse_hist(end+1) = current_mse;
    Yhat_stage_raw = Yhat_tmp(2:end) * norm_stats.y_std + norm_stats.y_mu;
    stage_rmse = sqrt(mean((Ytr_raw(2:end) - Yhat_stage_raw).^2));
    stage_fit  = fitPercent(Ytr_raw(2:end), Yhat_stage_raw);
    if config.model.force_hidden_growth
        fprintf('--- Hidden=%d/%d | MSE=%.6g | RMSE=%.4g | Fit=%.2f%% | improvement=%.3g | force=ON ---\n', ...
            numel(W_hidden), config.model.max_hidden_units, current_mse, stage_rmse, stage_fit, improvement);
    else
        fprintf('--- Hidden=%d | MSE=%.6g | RMSE=%.4g | Fit=%.2f%% | improvement=%.3g ---\n', ...
            numel(W_hidden), current_mse, stage_rmse, stage_fit, improvement);
    end
    if ~isnan(outputTrainInfo.plateau_epoch)
        fprintf('Output layer re-train plateau at epoch %d (ran %d/%d epochs).\n', ...
            outputTrainInfo.plateau_epoch, outputTrainInfo.epochs_run, config.model.max_epochs_output);
    else
        fprintf('Output layer re-train used %d/%d epochs (no plateau).\n', ...
            outputTrainInfo.epochs_run, config.model.max_epochs_output);
    end
    config.model.eta_output = config.model.eta_output ;
    fprintf('Output learning rate reduced to %.2e for next hidden unit.\n', config.model.eta_output);
    [lossPlotHandle, lossFigHandle] = updateLossFigure(lossPlotHandle, lossFigHandle, mse_hist);
end

% Full-series recursive prediction and denormalize
Yhat_tr = recursivePredictFullSeries(Utr, Ytr, W_hidden, w_o, config.model.activation, config);
Yhat_va = recursivePredictFullSeries(Uva, Yva, W_hidden, w_o, config.model.activation, config);

Yhat_tr = Yhat_tr(2:end) * norm_stats.y_std + norm_stats.y_mu;
Yhat_va = Yhat_va(2:end) * norm_stats.y_std + norm_stats.y_mu;

fit_tr = fitPercent(Ytr_raw(2:end), Yhat_tr);
fit_va = fitPercent(Yva_raw(2:end), Yhat_va);
rmse_tr = sqrt(mean((Ytr_raw(2:end) - Yhat_tr).^2));
rmse_va = sqrt(mean((Yva_raw(2:end) - Yhat_va).^2));
fprintf('\nTrain Fit: %.2f%% (RMSE=%.4g) | Val Fit: %.2f%% (RMSE=%.4g)\n', fit_tr, rmse_tr, fit_va, rmse_va);

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
logInfo.moving_avg_window = config.model.moving_avg_window;
logInfo.hidden_units = numel(W_hidden);
logInfo.max_hidden_units = config.model.max_hidden_units;
logInfo.regressor_count = numel(config.regressors.u) + numel(config.regressors.y);
logInfo.regressors_u = config.regressors.u;
logInfo.regressors_y = config.regressors.y;
logInfo.n_steps = Npred;
logInfo.train_mse = current_mse;
logInfo.fit_train = fit_tr;
logInfo.fit_val = fit_va;
logInfo.rmse_train = rmse_tr;
logInfo.rmse_val = rmse_va;
logInfo.activation = config.model.activation;
% include training progress history so it is available to the log writer
logInfo.mse_history = mse_hist;
if isfield(outputTrainInfo,'stop_by_moving_avg')
    logInfo.output_stop_by_mavg = double(outputTrainInfo.stop_by_moving_avg);
end
logFilePath = writeParameterLog(config, logInfo);

% Move the run's folder into a centralized `logs` directory for easier collection
if ~isempty(logFilePath)
    [runFolderPath, ~, ~] = fileparts(logFilePath);
    [parentDir, runFolderName] = fileparts(runFolderPath);
    logsDir = fullfile(scriptDir, 'logs');
    if exist(logsDir, 'dir') == 0
        try
            mkdir(logsDir);
        catch
            warning('Could not create logs directory %s', logsDir);
        end
    end
    destPath = fullfile(logsDir, runFolderName);
    if ~strcmpi(runFolderPath, destPath)
        try
            movefile(runFolderPath, destPath);
            logFilePath = fullfile(destPath, [runFolderName '.log']);
        catch ME
            warning('Could not move run folder to %s: %s', destPath, ME.message);
            try
                copyfile(runFolderPath, destPath);
                logFilePath = fullfile(destPath, [runFolderName '.log']);
            catch ME2
                warning('Could not copy run folder to %s: %s', destPath, ME2.message);
            end
        end
    end
end
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

% also save the loss-vs-units figure into the run folder so it's available visually
% include the output loss-history figure (if present) when saving run figures
figMap = struct('train', figTrain, 'val', figVal, 'loss', lossFigHandle, 'candidate_corr', corrFigHandle);
if exist('lossHistoryFig','var') && ~isempty(lossHistoryFig) && ishandle(lossHistoryFig)
    figMap.loss_history = lossHistoryFig;
end
savedFigurePaths = saveFitFigures(logFilePath, figMap);
if ~isempty(savedFigurePaths) && ~isempty(logFilePath)
    appendFigureInfoToLog(logFilePath, savedFigurePaths);
end

% Local helper functions are located in the `function` subfolder.
