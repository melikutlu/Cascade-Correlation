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
config.data.source = 'twotankdata'; % veri seti adı: twotankdata | dryer2 | mrDamper

config.data.train_ratio = 0.5; % eğitim verisi oranı
config.data.val_ratio = 0.5; % doğrulama verisi oranı

config.norm_method = 'zscore'; % veriyi z-score ile normalize et

config.prediction.n_steps = 20; % tahmin ufku; otomatik mod kapalıysa bu değeri kullan
config.prediction.auto_full_horizon = false; % true ise kullanılabilir tüm veri uzunluğunu hedefle

% regressors (user can change)
config.regressors.u = [1]; % giriş gecikmeleri; u(t), u(t-1) gibi terimleri seçer
config.regressors.y = [1]; % çıkış gecikmeleri; y(t-1), y(t-2) gibi terimleri seçer
config.regressors.include_bias = false; % sabit bias regressor'ü ekle veya çıkar

% model / training
% activation options: 'tanh' (default), 'diff' (time diff of z), 'diff-tanh' (time diff then tanh)
config.model.activation = 'diff'; % gizli katman aktivasyon tipi
config.model.diff_clip_lower = -10; % diff aktivasyonunda alt kırpma sınırı
config.model.diff_clip_upper = 10; % diff aktivasyonunda üst kırpma sınırı
config.model.hidden_bootstrap_count = 4; % ilk kaç gizli birimi zorunlu eklenir 
config.model.hidden_acceptance_window = 3; % kabul kararı için kaç önceki gizli birimin ortalamasını kullanır
config.model.max_hidden_units = 15; % en fazla kaç gizli birim ekleneceği
config.model.force_hidden_growth = false; % true ise hedefe bakmadan gizli birim eklemeye devam eder
config.model.target_mse = 5e-4;  % durdurma / hedefleme için istenen MSE seviyesi


% Output-layer recursive simulation loss is evaluated every N epochs.
config.model.sim_loss_eval_interval = 10; % recursive sim-loss kaç epochta bir ölçülecek
config.model.sim_loss_min_blocks = 3; % plato kararından önce en az kaç blok çalışacak
config.model.output_max_epochs = 1000; % output katmanı için toplam epoch bütçesi
config.model.max_epochs_output = config.model.sim_loss_eval_interval; % tek blokta çalıştırılacak varsayılan epoch sayısı
config.model.eta_output = 0.005; % output katmanı öğrenme oranı
config.model.max_epochs_candidate = 300; % aday gizli biriminin en çok kaç epoch eğitileceği
config.model.eta_candidate = 0.003; % aday gizli biriminin öğrenme oranı
config.model.plateau_min_delta = 0;   % önceki pencere ortalamasına göre en küçük iyileşme eşiği

config.model.moving_avg_window = 20;      % plato kontrolünde kaç önceki epochun ortalamasının kullanılacağı
config.model.use_plateau_stop = true; % aday eğitiminde plato durdurma açık, output katmanında blok tabanlı mantık kullanılır

config.training = struct();
config.training.batch_size_output = 32;     % output katmanı güncellemesinde mini-batch boyutu
config.training.batch_size_candidate = 32;  % aday birim aramasında mini-batch boyutu
config.training.candidate_pool_size = 1;    % aynı anda eğitilecek aday sayısı
config.training.use_parfor_pool = false ;     % true ise aday havuzunu paralel eğitir (varsa)

% Dataset'e özgü tüm varsayılanlar source adına göre otomatik doldurulur.
config = applyDatasetDefaults(config);

% load raw data according to config, then normalize
[Utr_raw, Ytr_raw, Uva_raw, Yva_raw] = loadDataByConfig_min(config);
[Utr, Ytr, Uva, Yva, norm_stats] = normalizeData_min(config.norm_method, Utr_raw, Ytr_raw, Uva_raw, Yva_raw);

if isfield(config.prediction, 'auto_full_horizon') && config.prediction.auto_full_horizon
    maxLag = getMaxLagFromRegressors(config.regressors);
    maxStepsTr = numel(Ytr) - maxLag;
    maxStepsVa = numel(Yva) - maxLag;
    autoSteps = min([maxStepsTr, maxStepsVa]);
    if autoSteps < 1
        error('Not enough samples to build at least one full-horizon trajectory.');
    end
    config.prediction.n_steps = autoSteps; % otomatik modda tahmin ufkunu veri uzunluğuna göre ayarla
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

% Stage 1: train output weights only using block-wise simulation plateau logic
[w_o, ~, outputTrainInfo, lossHistoryFig] = trainOutputLayer_TrajectorySimPlateau(X0_tr, Utr_seq, Ttr_seq, Utr, Ytr, w_o, W_hidden, config.model.activation, config, 'b');
% Full-series recursive MSE (tutarlı olması için büyüme kararları bununla alınır)
Yhat_tmp = recursivePredictFullSeries(Utr, Ytr, W_hidden, w_o, config.model.activation, config);
if any(isnan(Yhat_tmp(:))) || any(isinf(Yhat_tmp(:)))
    fprintf('WARNING: Yhat_tmp contains NaN or Inf values! Activation explosion detected.\n');
    fprintf('  NaN count: %d, Inf count: %d\n', sum(isnan(Yhat_tmp(:))), sum(isinf(Yhat_tmp(:))));
end
current_mse = mean((Ytr(2:end) - Yhat_tmp(2:end)).^2);
Yhat_tmp_raw = Yhat_tmp(2:end) * norm_stats.y_std + norm_stats.y_mu;
stage0_rmse = sqrt(mean((Ytr_raw(2:end) - Yhat_tmp_raw).^2));
stage0_fit  = fitPercent(Ytr_raw(2:end), Yhat_tmp_raw);
fprintf('--- Hidden=0 | MSE=%.6g | RMSE=%.4g | Fit=%.2f%% ---\n', current_mse, stage0_rmse, stage0_fit);

baselineW_hidden = W_hidden;
baselineW_o = w_o;
baselineMse = current_mse;
hiddenBootstrapCount = max(1, round(config.model.hidden_bootstrap_count));
hiddenAcceptanceWindow = max(1, round(config.model.hidden_acceptance_window));
acceptedHiddenMseHistory = zeros(0, 1);
hiddenGrowthRevertedToBaseline = false;
hiddenGrowthStoppedByMse = false;

if isfield(outputTrainInfo,'stop_by_sim_plateau') && outputTrainInfo.stop_by_sim_plateau
    fprintf('Output layer simulation plateau at epoch %d (ran %d epochs across %d blocks).\n', ...
        outputTrainInfo.plateau_epoch, outputTrainInfo.epochs_run, outputTrainInfo.block_count);
else
    fprintf('Output layer used %d epochs across %d blocks (no simulation plateau).\n', ...
        outputTrainInfo.epochs_run, outputTrainInfo.block_count);
end

if isfield(outputTrainInfo,'sim_loss_history') && ~isempty(outputTrainInfo.sim_loss_history)
    fprintf('Output sim-loss history per block: %s\n', mat2str(outputTrainInfo.sim_loss_history, 5));
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
    candidateEpochHistory(end+1) = candInfo.epochs_run; 
    candidatePlateauHistory(end+1) = candInfo.plateau_epoch; 
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
    [w_o, ~, outputTrainInfo, lossHistoryFig] = trainOutputLayer_TrajectorySimPlateau(X0_tr, Utr_seq, Ttr_seq, Utr, Ytr, w_o, W_hidden, config.model.activation, config, 'r');
    % Full-series recursive MSE (tutarlı olması için büyüme kararları bununla alınır)
    Yhat_tmp = recursivePredictFullSeries(Utr, Ytr, W_hidden, w_o, config.model.activation, config);
    current_mse = mean((Ytr(2:end) - Yhat_tmp(2:end)).^2);

    improvement = prev_mse - current_mse;

    acceptedHiddenCount = numel(acceptedHiddenMseHistory);
    acceptCandidate = true;
    referenceMse = NaN;

    if ~config.model.force_hidden_growth
        if acceptedHiddenCount < hiddenBootstrapCount
            acceptCandidate = true;
        else
            if acceptedHiddenCount < hiddenAcceptanceWindow
                referenceMse = mean(acceptedHiddenMseHistory);
            else
                referenceMse = mean(acceptedHiddenMseHistory(end-hiddenAcceptanceWindow+1:end));
            end
            acceptCandidate = current_mse < referenceMse;
        end
    end

    if ~acceptCandidate
        W_hidden(end) = [];
        w_o = w_o_prev;
        fprintf('Reject hidden #%d: full-recursive MSE=%.6g did not beat rolling mean %.6g. Stopping growth.\n', ...
            numel(W_hidden) + 1, current_mse, referenceMse);
        hiddenGrowthStoppedByMse = true;
        break;
    end

    acceptedHiddenMseHistory(end+1, 1) = current_mse; %#ok<AGROW>
    mse_hist(end+1) = current_mse;
    Yhat_stage_raw = Yhat_tmp(2:end) * norm_stats.y_std + norm_stats.y_mu;
    stage_rmse = sqrt(mean((Ytr_raw(2:end) - Yhat_stage_raw).^2));
    stage_fit  = fitPercent(Ytr_raw(2:end), Yhat_stage_raw);

    if config.model.force_hidden_growth
        fprintf('--- Hidden=%d/%d | MSE=%.6g | RMSE=%.4g | Fit=%.2f%% | improvement=%.3g | force=ON ---\n', ...
            numel(W_hidden), config.model.max_hidden_units, current_mse, stage_rmse, stage_fit, improvement);
    elseif acceptedHiddenCount < hiddenBootstrapCount
        fprintf('--- Hidden=%d/%d | bootstrap trial | MSE=%.6g | RMSE=%.4g | Fit=%.2f%% | improvement=%.3g ---\n', ...
            numel(W_hidden), config.model.max_hidden_units, current_mse, stage_rmse, stage_fit, improvement);
    else
        fprintf('--- Hidden=%d | accepted | MSE=%.6g | RMSE=%.4g | Fit=%.2f%% | improvement=%.3g | refAvg=%.6g ---\n', ...
            numel(W_hidden), current_mse, stage_rmse, stage_fit, improvement, referenceMse);
    end

    if ~config.model.force_hidden_growth && numel(acceptedHiddenMseHistory) == hiddenBootstrapCount
        bootstrapAvg = mean(acceptedHiddenMseHistory(end-hiddenBootstrapCount+1:end));
        fprintf('Bootstrap check: mean of first %d accepted hidden MSEs = %.6g | baseline = %.6g\n', ...
            hiddenBootstrapCount, bootstrapAvg, baselineMse);
        if bootstrapAvg >= baselineMse
            W_hidden = baselineW_hidden;
            w_o = baselineW_o;
            acceptedHiddenMseHistory = zeros(0, 1);
            current_mse = baselineMse;
            mse_hist = baselineMse;
            hiddenGrowthRevertedToBaseline = true;
            fprintf('Bootstrap block did not improve baseline. Reverting to hidden=0 and stopping.\n');
            [lossPlotHandle, lossFigHandle] = updateLossFigure(lossPlotHandle, lossFigHandle, mse_hist);
            break;
        end
    end

    if ~config.model.force_hidden_growth
        if acceptedHiddenCount < hiddenBootstrapCount
            fprintf('Accepted hidden #%d as bootstrap unit (%d/%d).\n', numel(W_hidden), numel(acceptedHiddenMseHistory), hiddenBootstrapCount);
        else
            fprintf('Accepted hidden #%d against rolling window mean %.6g.\n', numel(W_hidden), referenceMse);
        end
    end

    if isfield(outputTrainInfo,'stop_by_sim_plateau') && outputTrainInfo.stop_by_sim_plateau
        fprintf('Output layer re-train simulation plateau at epoch %d (ran %d epochs across %d blocks).\n', ...
            outputTrainInfo.plateau_epoch, outputTrainInfo.epochs_run, outputTrainInfo.block_count);
    else
        fprintf('Output layer re-train used %d epochs across %d blocks (no simulation plateau).\n', ...
            outputTrainInfo.epochs_run, outputTrainInfo.block_count);
    end
    config.model.eta_output = config.model.eta_output ; % sonraki hidden birim için mevcut output öğrenme oranını koru
    fprintf('Output learning rate reduced to %.2e for next hidden unit.\n', config.model.eta_output);
    [lossPlotHandle, lossFigHandle] = updateLossFigure(lossPlotHandle, lossFigHandle, mse_hist);
end

% Full-series recursive prediction and denormalize
Yhat_tr = recursivePredictFullSeries(Utr, Ytr, W_hidden, w_o, config.model.activation, config);
Yhat_va = recursivePredictFullSeries(Uva, Yva, W_hidden, w_o, config.model.activation, config);

% Son kontrol
if any(isnan(Yhat_tr(:))) || any(isinf(Yhat_tr(:)))
    fprintf('WARNING: Yhat_tr contains NaN/Inf (%.0f NaN, %.0f Inf)\n', sum(isnan(Yhat_tr(:))), sum(isinf(Yhat_tr(:))));
end

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
logInfo.sim_loss_eval_interval = config.model.sim_loss_eval_interval;
logInfo.sim_loss_min_blocks = config.model.sim_loss_min_blocks;
logInfo.output_max_epochs = config.model.output_max_epochs;
logInfo.output_epochs_used = outputTrainInfo.epochs_run;
logInfo.output_plateau_epoch = outputTrainInfo.plateau_epoch;
logInfo.output_block_count = outputTrainInfo.block_count;
logInfo.max_epochs_candidate = config.model.max_epochs_candidate;
logInfo.hidden_bootstrap_count = config.model.hidden_bootstrap_count;
logInfo.hidden_acceptance_window = config.model.hidden_acceptance_window;
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
logInfo.hidden_baseline_mse = baselineMse;
logInfo.hidden_accepted_mse_history = acceptedHiddenMseHistory;
logInfo.hidden_growth_reverted_to_baseline = double(hiddenGrowthRevertedToBaseline);
logInfo.hidden_growth_stopped_by_mse = double(hiddenGrowthStoppedByMse);
% include training progress history so it is available to the log writer
logInfo.mse_history = mse_hist;
if isfield(outputTrainInfo,'sim_loss_history')
    logInfo.output_sim_loss_history = outputTrainInfo.sim_loss_history;
end
if isfield(outputTrainInfo,'stop_by_sim_plateau')
    logInfo.output_stop_by_sim_plateau = double(outputTrainInfo.stop_by_sim_plateau);
end
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
