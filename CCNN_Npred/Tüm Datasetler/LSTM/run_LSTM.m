% LSTM baseline for the CCNN datasets.
% Change only config.data.source to select a dataset.

clear; clc; close all; rng(0);

scriptFullPath = mfilename('fullpath');
if isempty(scriptFullPath)
    scriptDir = pwd;
else
    [scriptDir, ~] = fileparts(scriptFullPath);
end
projectDir = fileparts(scriptDir);
ccnnVersionDir = fullfile(projectDir, 'v1.1');
funcFolder = fullfile(ccnnVersionDir, 'function');
addpath(funcFolder, '-begin');
addpath(ccnnVersionDir, '-begin');

% ------------------
% CONFIG
% ------------------
config = struct();
config.data.source = 'robotarmdata'; % twotankdata | dryer2 | mrDamper | robotarmdata
config.data.train_ratio = 0.5;
config.data.val_ratio = 0.5;

config.norm_method = 'zscore';

% Regressors follow the same SISO convention as CCNN:
% u lag 0 means u(t); y lag 0 is not allowed for recursive prediction.
config.regressors.u = [1 2 3];
config.regressors.y = [1 2 3];

% LSTM model parameters.
config.model.hidden_units = 64;
config.model.num_layers = 1;
config.model.dropout = 0.0;
config.model.output_activation = 'linear'; % linear | tanh | relu

% Training parameters.
config.training.max_epochs = 300;
config.training.initial_learn_rate = 1e-3;
config.training.mini_batch_size = 1;
config.training.loss_mode = 'simulation'; % one-step | simulation
config.training.simulation_horizon = 20;
config.training.simulation_batch_size = 64;
config.training.gradient_threshold = 1;
config.training.shuffle = 'never';
config.training.validation_frequency = 20; % simulation mode: epochs; one-step mode: iterations
config.training.execution_environment = 'auto'; % auto | cpu | gpu
config.training.show_training_plot = true;

config = applyDatasetDefaults(config);
validateSisoRecurrentConfig(config);

[Utr_raw, Ytr_raw, Uva_raw, Yva_raw] = loadDataByConfig_min(config);
[Utr, Ytr, Uva, Yva, norm_stats] = normalizeData_min(config.norm_method, Utr_raw, Ytr_raw, Uva_raw, Yva_raw);

maxLag = getMaxLagFromRecurrentConfig(config);
trainStartIdx = maxLag + 1;
valStartIdx = maxLag + 1;

switch lower(config.training.loss_mode)
    case 'one-step'
        [XTrain, TTrain, trainStartIdx] = buildLaggedSequence(Utr, Ytr, config, true);
        [XVal, TVal, valStartIdx] = buildLaggedSequence(Uva, Yva, config, true);
        XTrainCell = {XTrain};
        TTrainCell = {TTrain};
        XValCell = {XVal};
        TValCell = {TVal};

        layers = buildRecurrentLayers(size(XTrain, 1), config, 'lstm');
        options = trainingOptions('adam', ...
            'MaxEpochs', config.training.max_epochs, ...
            'InitialLearnRate', config.training.initial_learn_rate, ...
            'MiniBatchSize', config.training.mini_batch_size, ...
            'GradientThreshold', config.training.gradient_threshold, ...
            'Shuffle', config.training.shuffle, ...
            'ValidationData', {XValCell, TValCell}, ...
            'ValidationFrequency', config.training.validation_frequency, ...
            'ExecutionEnvironment', config.training.execution_environment, ...
            'Verbose', true, ...
            'Plots', trainingPlotOption(config.training.show_training_plot));

        [net, trainInfo] = trainNetwork(XTrainCell, TTrainCell, layers, options);
    case 'simulation'
        [net, trainInfo] = trainRecurrentSimulation(Utr, Ytr, Uva, Yva, config, 'lstm');
    otherwise
        error('Unknown config.training.loss_mode: %s', config.training.loss_mode);
end

Yhat_tr_norm = recursivePredictRecurrent(net, Utr, Ytr, config);
Yhat_va_norm = recursivePredictRecurrent(net, Uva, Yva, config);

Yhat_tr = Yhat_tr_norm(trainStartIdx:end) * norm_stats.y_std + norm_stats.y_mu;
Yhat_va = Yhat_va_norm(valStartIdx:end) * norm_stats.y_std + norm_stats.y_mu;
Ytr_eval = Ytr_raw(trainStartIdx:end);
Yva_eval = Yva_raw(valStartIdx:end);

rmse_tr = sqrt(mean((Ytr_eval - Yhat_tr).^2));
rmse_va = sqrt(mean((Yva_eval - Yhat_va).^2));
fit_tr = fitPercent(Ytr_eval, Yhat_tr);
fit_va = fitPercent(Yva_eval, Yhat_va);

fprintf('\nLSTM results (%s)\n', char(config.data.source_label));
fprintf('Train RMSE: %.6g | Train Fit: %.2f%%\n', rmse_tr, fit_tr);
fprintf('Val   RMSE: %.6g | Val   Fit: %.2f%%\n', rmse_va, fit_va);

runFolder = createRunFolder(scriptDir, config, fit_tr, fit_va);
save(fullfile(runFolder, 'LSTM_Model.mat'), 'net', 'trainInfo', 'config', 'norm_stats', 'rmse_tr', 'rmse_va', 'fit_tr', 'fit_va');
writeBaselineLog(fullfile(runFolder, 'LSTM_Log.txt'), 'LSTM', config, rmse_tr, rmse_va, fit_tr, fit_va);
saveTrainingLossFigure(runFolder, 'LSTM', trainInfo);
saveBaselineFigures(runFolder, 'LSTM', Ytr_eval, Yhat_tr, Yva_eval, Yhat_va, fit_tr, fit_va);

function validateSisoRecurrentConfig(config)
    if any(config.regressors.y(:) <= 0)
        error('config.regressors.y must contain positive lags only. Use [1 2 3], not [0 1 2 3].');
    end
    if any(config.regressors.u(:) < 0)
        error('config.regressors.u cannot contain negative lags.');
    end
    if isfield(config.training, 'loss_mode') && strcmpi(config.training.loss_mode, 'simulation') && config.model.num_layers ~= 1
        error('Simulation-loss training currently supports config.model.num_layers = 1.');
    end
end

function maxLag = getMaxLagFromRecurrentConfig(config)
    ulags = config.regressors.u(:)';
    ylags = config.regressors.y(:)';
    maxLag = max([ulags(ulags > 0), ylags]);
end

function [params, info] = trainRecurrentSimulation(Utr, Ytr, Uva, Yva, config, modelType)
    inputSize = numel(config.regressors.u) + numel(config.regressors.y);
    hiddenSize = config.model.hidden_units;
    horizon = config.training.simulation_horizon;
    batchSize = config.training.simulation_batch_size;
    maxLag = getMaxLagFromRecurrentConfig(config);

    params = initializeCustomRecurrent(inputSize, hiddenSize, modelType);

    trainStarts = maxLag:(numel(Ytr) - horizon);
    valStarts = maxLag:(numel(Yva) - horizon);
    if isempty(trainStarts) || isempty(valStarts)
        error('Not enough samples for simulation_horizon=%d and max lag=%d.', horizon, maxLag);
    end

    avgGrad = [];
    avgSqGrad = [];
    iter = 0;
    trainLossHist = zeros(config.training.max_epochs, 1);
    valLossHist = nan(config.training.max_epochs, 1);

    for epoch = 1:config.training.max_epochs
        if strcmpi(config.training.shuffle, 'never')
            order = trainStarts;
        else
            order = trainStarts(randperm(numel(trainStarts)));
        end

        epochLoss = 0;
        batchCount = 0;
        for first = 1:batchSize:numel(order)
            batchStarts = order(first:min(first + batchSize - 1, numel(order)));
            iter = iter + 1;
            [loss, grads] = dlfeval(@simulationGradients, params, Utr, Ytr, batchStarts, config, modelType);
            grads = clipGradientStruct(grads, config.training.gradient_threshold);
            [params, avgGrad, avgSqGrad] = adamupdate(params, grads, avgGrad, avgSqGrad, iter, config.training.initial_learn_rate);
            epochLoss = epochLoss + double(gather(extractdata(loss)));
            batchCount = batchCount + 1;
        end

        trainLossHist(epoch) = epochLoss / max(batchCount, 1);
        if mod(epoch, config.training.validation_frequency) == 0 || epoch == 1 || epoch == config.training.max_epochs
            valSubset = valStarts(round(linspace(1, numel(valStarts), min(numel(valStarts), 512))));
            valLoss = simulationLoss(params, Uva, Yva, valSubset, config, modelType);
            valLossHist(epoch) = double(gather(extractdata(valLoss)));
            fprintf('%s simulation epoch %d/%d | train loss %.6g | val loss %.6g\n', ...
                upper(modelType), epoch, config.training.max_epochs, trainLossHist(epoch), valLossHist(epoch));
        else
            fprintf('%s simulation epoch %d/%d | train loss %.6g\n', ...
                upper(modelType), epoch, config.training.max_epochs, trainLossHist(epoch));
        end
    end

    info = struct();
    info.Iteration = (1:config.training.max_epochs)';
    info.TrainingLoss = trainLossHist;
    info.ValidationLoss = valLossHist;
    params.custom_type = modelType;
end

function params = initializeCustomRecurrent(inputSize, hiddenSize, modelType)
    scaleIn = 0.1;
    scaleRec = 0.1;
    switch lower(modelType)
        case 'gru'
            params.Wz = dlarray(randn(hiddenSize, inputSize) * scaleIn);
            params.Uz = dlarray(randn(hiddenSize, hiddenSize) * scaleRec);
            params.bz = dlarray(zeros(hiddenSize, 1));
            params.Wr = dlarray(randn(hiddenSize, inputSize) * scaleIn);
            params.Ur = dlarray(randn(hiddenSize, hiddenSize) * scaleRec);
            params.br = dlarray(zeros(hiddenSize, 1));
            params.Wh = dlarray(randn(hiddenSize, inputSize) * scaleIn);
            params.Uh = dlarray(randn(hiddenSize, hiddenSize) * scaleRec);
            params.bh = dlarray(zeros(hiddenSize, 1));
        case 'lstm'
            params.Wi = dlarray(randn(hiddenSize, inputSize) * scaleIn);
            params.Ui = dlarray(randn(hiddenSize, hiddenSize) * scaleRec);
            params.bi = dlarray(zeros(hiddenSize, 1));
            params.Wf = dlarray(randn(hiddenSize, inputSize) * scaleIn);
            params.Uf = dlarray(randn(hiddenSize, hiddenSize) * scaleRec);
            params.bf = dlarray(ones(hiddenSize, 1));
            params.Wo = dlarray(randn(hiddenSize, inputSize) * scaleIn);
            params.Uo = dlarray(randn(hiddenSize, hiddenSize) * scaleRec);
            params.bo = dlarray(zeros(hiddenSize, 1));
            params.Wg = dlarray(randn(hiddenSize, inputSize) * scaleIn);
            params.Ug = dlarray(randn(hiddenSize, hiddenSize) * scaleRec);
            params.bg = dlarray(zeros(hiddenSize, 1));
        otherwise
            error('Unknown model type: %s', modelType);
    end
    params.Wy = dlarray(randn(1, hiddenSize) * scaleIn);
    params.by = dlarray(0);
end

function [loss, grads] = simulationGradients(params, U, Y, starts, config, modelType)
    loss = simulationLoss(params, U, Y, starts, config, modelType);
    grads = dlgradient(loss, params);
end

function loss = simulationLoss(params, U, Y, starts, config, modelType)
    [Ypred, Ytrue] = forwardSimulation(params, U, Y, starts, config, modelType);
    err = Ypred - Ytrue;
    loss = mean(err(:).^2);
end

function [Ypred, Ytrue] = forwardSimulation(params, U, Y, starts, config, modelType)
    ulags = config.regressors.u(:)';
    ylags = config.regressors.y(:)';
    horizon = config.training.simulation_horizon;
    B = numel(starts);
    hiddenSize = config.model.hidden_units;

    h = dlarray(zeros(hiddenSize, B));
    c = dlarray(zeros(hiddenSize, B));
    Ypred = dlarray(zeros(B, horizon));
    Ytrue = dlarray(zeros(B, horizon));

    for t = 1:horizon
        k = starts + t;
        x = dlarray(zeros(numel(ulags) + numel(ylags), B));
        for j = 1:numel(ulags)
            L = ulags(j);
            if L == 0
                x(j, :) = dlarray(U(k)');
            else
                x(j, :) = dlarray(U(k - L)');
            end
        end
        for j = 1:numel(ylags)
            L = ylags(j);
            predIdx = t - L;
            if predIdx >= 1
                x(numel(ulags) + j, :) = Ypred(:, predIdx)';
            else
                x(numel(ulags) + j, :) = dlarray(Y(k - L)');
            end
        end

        switch lower(modelType)
            case 'gru'
                [h, y] = gruStep(params, x, h, config);
            case 'lstm'
                [h, c, y] = lstmStep(params, x, h, c, config);
        end
        Ypred(:, t) = y';
        Ytrue(:, t) = dlarray(Y(k)');
    end
end

function [h, y] = gruStep(params, x, h, config)
    z = sigmoidLocal(params.Wz * x + params.Uz * h + params.bz);
    r = sigmoidLocal(params.Wr * x + params.Ur * h + params.br);
    hCand = tanh(params.Wh * x + params.Uh * (r .* h) + params.bh);
    h = (1 - z) .* h + z .* hCand;
    y = applyOutputActivation(params.Wy * h + params.by, config);
end

function [h, c, y] = lstmStep(params, x, h, c, config)
    i = sigmoidLocal(params.Wi * x + params.Ui * h + params.bi);
    f = sigmoidLocal(params.Wf * x + params.Uf * h + params.bf);
    o = sigmoidLocal(params.Wo * x + params.Uo * h + params.bo);
    g = tanh(params.Wg * x + params.Ug * h + params.bg);
    c = f .* c + i .* g;
    h = o .* tanh(c);
    y = applyOutputActivation(params.Wy * h + params.by, config);
end

function y = applyOutputActivation(y, config)
    switch lower(config.model.output_activation)
        case 'linear'
        case 'tanh'
            y = tanh(y);
        case 'relu'
            y = max(y, 0);
        otherwise
            error('Unsupported config.model.output_activation: %s', config.model.output_activation);
    end
end

function y = sigmoidLocal(x)
    y = 1 ./ (1 + exp(-x));
end

function grads = clipGradientStruct(grads, threshold)
    if isempty(threshold) || threshold <= 0
        return;
    end
    gradNorm = sqrt(sumGradSquares(grads));
    if gradNorm > threshold
        scale = threshold / (gradNorm + eps);
        grads = dlupdate(@(g) g * scale, grads);
    end
end

function total = sumGradSquares(grads)
    total = 0;
    names = fieldnames(grads);
    for i = 1:numel(names)
        value = grads.(names{i});
        if isa(value, 'dlarray')
            total = total + double(gather(extractdata(sum(value(:).^2))));
        end
    end
end

function [X, T, startIdx] = buildLaggedSequence(U, Y, config, useMeasuredY)
    ulags = config.regressors.u(:)';
    ylags = config.regressors.y(:)';
    maxLag = max([ulags(ulags > 0), ylags]);
    startIdx = maxLag + 1;
    N = numel(Y);
    X = zeros(numel(ulags) + numel(ylags), N - maxLag);
    T = zeros(1, N - maxLag);

    for k = startIdx:N
        col = k - maxLag;
        uvals = zeros(1, numel(ulags));
        for j = 1:numel(ulags)
            L = ulags(j);
            if L == 0
                uvals(j) = U(k);
            else
                uvals(j) = U(k - L);
            end
        end

        yvals = zeros(1, numel(ylags));
        for j = 1:numel(ylags)
            L = ylags(j);
            if useMeasuredY
                yvals(j) = Y(k - L);
            else
                error('Recursive sequence construction is handled by recursivePredictRecurrent.');
            end
        end

        X(:, col) = [uvals, yvals]';
        T(:, col) = Y(k);
    end
end

function layers = buildRecurrentLayers(featureCount, config, layerType)
    layers = [
        sequenceInputLayer(featureCount, 'Name', 'input')
    ];

    for layerIdx = 1:config.model.num_layers
        layerName = sprintf('%s_%d', layerType, layerIdx);
        switch lower(layerType)
            case 'lstm'
                recurrentLayer = lstmLayer(config.model.hidden_units, 'OutputMode', 'sequence', 'Name', layerName);
            case 'gru'
                recurrentLayer = gruLayer(config.model.hidden_units, 'OutputMode', 'sequence', 'Name', layerName);
            otherwise
                error('Unknown recurrent layer type: %s', layerType);
        end
        layers = [layers; recurrentLayer]; %#ok<AGROW>
        if config.model.dropout > 0
            layers = [layers; dropoutLayer(config.model.dropout, 'Name', sprintf('dropout_%d', layerIdx))]; %#ok<AGROW>
        end
    end

    switch lower(config.model.output_activation)
        case 'linear'
            layers = [layers; fullyConnectedLayer(1, 'Name', 'fc'); regressionLayer('Name', 'regression')];
        case 'tanh'
            layers = [layers; fullyConnectedLayer(1, 'Name', 'fc'); tanhLayer('Name', 'output_tanh'); regressionLayer('Name', 'regression')];
        case 'relu'
            layers = [layers; fullyConnectedLayer(1, 'Name', 'fc'); reluLayer('Name', 'output_relu'); regressionLayer('Name', 'regression')];
        otherwise
            error('Unsupported config.model.output_activation: %s', config.model.output_activation);
    end
end

function plotMode = trainingPlotOption(showPlot)
    if showPlot
        plotMode = 'training-progress';
    else
        plotMode = 'none';
    end
end

function Yhat = recursivePredictRecurrent(net, U, Y, config)
    if isstruct(net) && isfield(net, 'custom_type')
        Yhat = recursivePredictCustomRecurrent(net, U, Y, config);
        return;
    end

    ulags = config.regressors.u(:)';
    ylags = config.regressors.y(:)';
    maxLag = max([ulags(ulags > 0), ylags]);
    N = numel(Y);
    Yhat = zeros(N, 1);
    Yhat(1:maxLag) = Y(1:maxLag);

    netState = resetState(net);
    for k = maxLag + 1:N
        uvals = zeros(1, numel(ulags));
        for j = 1:numel(ulags)
            L = ulags(j);
            if L == 0
                uvals(j) = U(k);
            else
                uvals(j) = U(k - L);
            end
        end

        yvals = zeros(1, numel(ylags));
        for j = 1:numel(ylags)
            L = ylags(j);
            yvals(j) = Yhat(k - L);
        end

        x = [uvals, yvals]';
        [netState, yhat] = predictAndUpdateState(netState, x, 'ExecutionEnvironment', config.training.execution_environment);
        Yhat(k) = gather(yhat);
    end
end

function Yhat = recursivePredictCustomRecurrent(params, U, Y, config)
    ulags = config.regressors.u(:)';
    ylags = config.regressors.y(:)';
    maxLag = getMaxLagFromRecurrentConfig(config);
    N = numel(Y);
    Yhat = zeros(N, 1);
    Yhat(1:maxLag) = Y(1:maxLag);

    params = gatherCustomParams(params);
    hiddenSize = config.model.hidden_units;
    h = zeros(hiddenSize, 1);
    c = zeros(hiddenSize, 1);

    for k = maxLag + 1:N
        x = zeros(numel(ulags) + numel(ylags), 1);
        for j = 1:numel(ulags)
            L = ulags(j);
            if L == 0
                x(j) = U(k);
            else
                x(j) = U(k - L);
            end
        end
        for j = 1:numel(ylags)
            L = ylags(j);
            x(numel(ulags) + j) = Yhat(k - L);
        end

        switch lower(params.custom_type)
            case 'gru'
                [h, y] = gruStepNumeric(params, x, h, config);
            case 'lstm'
                [h, c, y] = lstmStepNumeric(params, x, h, c, config);
        end
        Yhat(k) = y;
    end
end

function params = gatherCustomParams(params)
    names = fieldnames(params);
    for i = 1:numel(names)
        value = params.(names{i});
        if isa(value, 'dlarray')
            params.(names{i}) = gather(extractdata(value));
        end
    end
end

function [h, y] = gruStepNumeric(params, x, h, config)
    z = sigmoidNumeric(params.Wz * x + params.Uz * h + params.bz);
    r = sigmoidNumeric(params.Wr * x + params.Ur * h + params.br);
    hCand = tanh(params.Wh * x + params.Uh * (r .* h) + params.bh);
    h = (1 - z) .* h + z .* hCand;
    y = applyOutputActivationNumeric(params.Wy * h + params.by, config);
end

function [h, c, y] = lstmStepNumeric(params, x, h, c, config)
    i = sigmoidNumeric(params.Wi * x + params.Ui * h + params.bi);
    f = sigmoidNumeric(params.Wf * x + params.Uf * h + params.bf);
    o = sigmoidNumeric(params.Wo * x + params.Uo * h + params.bo);
    g = tanh(params.Wg * x + params.Ug * h + params.bg);
    c = f .* c + i .* g;
    h = o .* tanh(c);
    y = applyOutputActivationNumeric(params.Wy * h + params.by, config);
end

function y = applyOutputActivationNumeric(y, config)
    switch lower(config.model.output_activation)
        case 'linear'
        case 'tanh'
            y = tanh(y);
        case 'relu'
            y = max(y, 0);
    end
end

function y = sigmoidNumeric(x)
    y = 1 ./ (1 + exp(-x));
end

function runFolder = createRunFolder(scriptDir, config, fitTr, fitVa)
    dataLabel = regexprep(char(config.data.source_label), '[^A-Za-z0-9_-]', '_');
    runFolder = fullfile(scriptDir, 'logs', dataLabel, sprintf('fitTr%d_fitVa%d', round(fitTr), round(fitVa)));
    if exist(runFolder, 'dir') == 0
        mkdir(runFolder);
    end
end

function writeBaselineLog(logPath, modelName, config, rmseTr, rmseVa, fitTr, fitVa)
    fid = fopen(logPath, 'w');
    if fid == -1
        warning('Could not write log file: %s', logPath);
        return;
    end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '%s Baseline Log\n', modelName);
    fprintf(fid, 'Created      : %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, 'Data source  : %s\n', char(config.data.source_label));
    fprintf(fid, 'Regressors u : %s\n', mat2str(config.regressors.u));
    fprintf(fid, 'Regressors y : %s\n', mat2str(config.regressors.y));
    fprintf(fid, 'Hidden units : %d\n', config.model.hidden_units);
    fprintf(fid, 'Layers       : %d\n', config.model.num_layers);
    fprintf(fid, 'Dropout      : %.6g\n', config.model.dropout);
    fprintf(fid, 'Output activation : %s\n', config.model.output_activation);
    fprintf(fid, 'Loss mode    : %s\n', config.training.loss_mode);
    if strcmpi(config.training.loss_mode, 'simulation')
        fprintf(fid, 'Simulation horizon : %d\n', config.training.simulation_horizon);
        fprintf(fid, 'Simulation batch size : %d\n', config.training.simulation_batch_size);
    end
    fprintf(fid, 'Max epochs   : %d\n', config.training.max_epochs);
    fprintf(fid, 'Learn rate   : %.6g\n', config.training.initial_learn_rate);
    fprintf(fid, 'Validation frequency : %d\n', config.training.validation_frequency);
    fprintf(fid, 'Loss history figure : LossHistory.png\n');
    fprintf(fid, 'Train RMSE   : %.6g\n', rmseTr);
    fprintf(fid, 'Val RMSE     : %.6g\n', rmseVa);
    fprintf(fid, 'Train Fit %%  : %.2f\n', fitTr);
    fprintf(fid, 'Val Fit %%    : %.2f\n', fitVa);
end

function saveBaselineFigures(runFolder, modelName, Ytr, YhatTr, Yva, YhatVa, fitTr, fitVa)
    hTrain = figure('Name', [modelName ' Train']);
    plot(Ytr, 'k', 'LineWidth', 1.0); hold on;
    plot(YhatTr, 'b--', 'LineWidth', 1.2); grid on;
    title(sprintf('%s TRAIN | Fit=%.2f%%', modelName, fitTr));
    legend('True', modelName);
    saveas(hTrain, fullfile(runFolder, 'Train.png'));

    hVal = figure('Name', [modelName ' Validation']);
    plot(Yva, 'k', 'LineWidth', 1.0); hold on;
    plot(YhatVa, 'r--', 'LineWidth', 1.2); grid on;
    title(sprintf('%s VAL | Fit=%.2f%%', modelName, fitVa));
    legend('True', modelName);
    saveas(hVal, fullfile(runFolder, 'Val.png'));
end

function saveTrainingLossFigure(runFolder, modelName, trainInfo)
    trainLoss = getInfoField(trainInfo, 'TrainingLoss');
    valLoss = getInfoField(trainInfo, 'ValidationLoss');
    iteration = getInfoField(trainInfo, 'Iteration');

    if isempty(trainLoss)
        warning('Training loss history was not available; LossHistory.png was not saved.');
        return;
    end
    if isempty(iteration)
        iteration = 1:numel(trainLoss);
    end

    hLoss = figure('Name', [modelName ' Loss History']);
    plot(iteration, trainLoss, 'b-', 'LineWidth', 1.2); hold on; grid on;
    if ~isempty(valLoss)
        if numel(valLoss) == numel(iteration)
            valIteration = iteration;
        else
            valIteration = 1:numel(valLoss);
        end
        validVal = ~isnan(valLoss);
        plot(valIteration(validVal), valLoss(validVal), 'r--', 'LineWidth', 1.2);
        legend('Training loss', 'Validation loss');
    else
        legend('Training loss');
    end
    xlabel('Iteration');
    ylabel('Loss');
    title(sprintf('%s Loss History', modelName));
    saveas(hLoss, fullfile(runFolder, 'LossHistory.png'));
end

function value = getInfoField(trainInfo, fieldName)
    value = [];
    if isstruct(trainInfo) && isfield(trainInfo, fieldName)
        value = trainInfo.(fieldName);
    elseif istable(trainInfo) && any(strcmp(trainInfo.Properties.VariableNames, fieldName))
        value = trainInfo.(fieldName);
    end
    if ~isempty(value)
        value = gather(value(:))';
    end
end
