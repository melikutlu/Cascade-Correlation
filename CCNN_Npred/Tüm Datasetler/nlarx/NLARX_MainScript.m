% NLARX Training Script with Dataset and Model Type Selection
% This script supports Cascade-Correlation and NLARX neural network models
% Performance metrics are logged for later analysis

clear; clc; close all; rng(0);

% Ensure local `function` folder is on MATLAB path
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

% =========================================
% DATASET SELECTION MENU
% =========================================
fprintf('\n=====================================\n');
fprintf('  NLARX Training - Configuration Menu\n');
fprintf('=====================================\n\n');
fprintf('--- Dataset Selection ---\n');

availableDatasets = {'twotankdata', 'dryer2', 'mrdamper', 'robotarmdata'};
fprintf('Available datasets:\n');
for i = 1:length(availableDatasets)
    fprintf('  %d: %s\n', i, availableDatasets{i});
end
fprintf('\n');

datasetChoice = input('Select dataset (1-4): ');
if datasetChoice < 1 || datasetChoice > length(availableDatasets)
    error('Invalid choice. Please select 1-4.');
end
datasetName = availableDatasets{datasetChoice};
fprintf('\nSelected: %s\n\n', datasetName);

% =========================================
% MODEL TYPE SELECTION MENU
% =========================================
fprintf('=====================================\n');
fprintf('  Model Type Selection\n');
fprintf('=====================================\n\n');

availableModels = {'Cascade-Correlation Neural Network', 'NLARX with Sigmoid NN', 'NLARX with Neural Network'};
fprintf('Available models:\n');
for i = 1:length(availableModels)
    fprintf('  %d: %s\n', i, availableModels{i});
end
fprintf('\n');

modelChoice = input('Select model type (1-3): ');
if modelChoice < 1 || modelChoice > length(availableModels)
    error('Invalid choice. Please select 1-3.');
end
modelName = availableModels{modelChoice};
fprintf('\nSelected: %s\n\n', modelName);

% =========================================
% CONFIGURE DATA LOADING
% =========================================
% Setup configuration (same structure as cascade correlation code)
config = struct();
config.data.source = datasetName;
config.data.train_ratio = 0.5;
config.data.val_ratio = 0.5;
config.norm_method = 'zscore';  % Z-score normalization

% Additional configuration for robotarmdata
if strcmpi(datasetName, 'robotarmdata')
    fprintf('RobotArm dataset requires additional configuration:\n');
    fprintf('Available validation experiments: 1, 2, 3 (MathWorks example uses 3)\n');
    valExp = input('Select validation experiment (1-3, default 3): ');
    if isempty(valExp)
        config.data.robotarm.validation_experiment = 3;
    else
        config.data.robotarm.validation_experiment = valExp;
    end
    fprintf('Downsample factor (default 10, matching MathWorks example): ');
    ds_factor = input('');
    if isempty(ds_factor)
        config.data.robotarm.downsample_factor = 10;
    else
        config.data.robotarm.downsample_factor = ds_factor;
    end
end

% =========================================
% LOAD RAW DATA
% =========================================
fprintf('Loading raw data from %s...\n', datasetName);
try
    [Utr_raw, Ytr_raw, Uva_raw, Yva_raw] = loadDataForNLARX(config);
    fprintf('Raw data loaded successfully.\n');
    fprintf('  Training samples: %d\n', length(Ytr_raw));
    fprintf('  Validation samples: %d\n', length(Yva_raw));
catch ME
    fprintf('Error loading data: %s\n', ME.message);
    return;
end

% =========================================
% NORMALIZE DATA (Z-SCORE)
% =========================================
fprintf('\nNormalizing data with Z-score method...\n');
[Utr, Ytr, Uva, Yva, norm_stats] = normalizeData_min(config.norm_method, Utr_raw, Ytr_raw, Uva_raw, Yva_raw);
fprintf('Data normalized successfully.\n');
fprintf('  U mean: %.6g, std: %.6g\n', norm_stats.u_mu, norm_stats.u_std);
fprintf('  Y mean: %.6g, std: %.6g\n\n', norm_stats.y_mu, norm_stats.y_std);

% =========================================
% CREATE IDDATA OBJECTS FOR NLARX
% =========================================
% Get sampling time from config defaults
config = applyDatasetDefaults(config);
switch lower(config.data.source)
    case 'twotankdata'
        Ts = config.data.twotank.sampling_time;
    case 'dryer2'
        Ts = config.data.dryer2.sampling_time;
    case 'mrdamper'
        Ts = 0.01;  % default for mrdamper
    case 'robotarmdata'
        Ts = config.data.robotarm.original_sampling_time * config.data.robotarm.downsample_factor;
    otherwise
        Ts = 1;
end

dataTraining = iddata(Ytr, Utr, Ts, 'OutputName', 'output', 'InputName', 'input');
dataValidation = iddata(Yva, Uva, Ts, 'OutputName', 'output', 'InputName', 'input');

fprintf('Created iddata objects for NLARX training.\n');

% =========================================
% VISUALIZATION OF INPUT DATA
% =========================================
fprintf('Plotting normalized input data...\n');
figure('Name', 'Input Data Visualization (Normalized)', 'Color', 'w');
idplot(dataTraining, dataValidation);
legend('Training (Normalized)', 'Validation (Normalized)', 'Location', 'best');
title(sprintf('%s - Normalized Input Data (Z-score)', datasetName));
grid on;

% Plot raw data for reference
figure('Name', 'Input Data Visualization (Raw)', 'Color', 'w');
subplot(2,1,1);
plot(Utr_raw, 'LineWidth', 1.2); hold on;
plot(length(Ytr_raw) + (1:length(Uva_raw)), Uva_raw, 'LineWidth', 1.2);
xlabel('Sample'); ylabel('Input (U)');
legend('Training', 'Validation');
title(sprintf('%s - Raw Input Data', datasetName));
grid on;

subplot(2,1,2);
plot(Ytr_raw, 'LineWidth', 1.2); hold on;
plot(length(Ytr_raw) + (1:length(Yva_raw)), Yva_raw, 'LineWidth', 1.2);
xlabel('Sample'); ylabel('Output (Y)');
legend('Training', 'Validation');
title(sprintf('%s - Raw Output Data', datasetName));
grid on;

% =========================================
% MODEL CONFIGURATION
% =========================================
fprintf('\n--- Model Configuration ---\n');

% Configure model based on selection
switch modelChoice
    case 1  % Cascade-Correlation Neural Network
        fprintf('Model Type: Cascade-Correlation Neural Network\n');
        activation = 'tanh';
        maxHiddenUnits = 20;
        fprintf('Activation function: %s\n', activation);
        fprintf('Max hidden units: %d\n', maxHiddenUnits);
        
        % Create cascade-correlation network
        f = idNeuralNetwork("cascade-correlation", activation, 0, 0, ...
            MaxNumActLayers=maxHiddenUnits, SizeSelection='off');
        fprintf('Cascade-Correlation neural network created.\n');
        
    case 2  % NLARX with Sigmoid Neural Network
        fprintf('Model Type: NLARX with Sigmoid Neural Network\n');
        activation = 'sigmoid';
        layerSizes = [8 5];  % 2 hidden layers: 8 and 5 units (reduced for stability)
        fprintf('Activation function: %s\n', activation);
        fprintf('Layer sizes: %s\n', mat2str(layerSizes));
        
        % Create feedforward neural network with sigmoid activation
        % idNeuralNetwork requires layer sizes as vector, not string
        f = idNeuralNetwork(layerSizes, activation, 0, 0);
        fprintf('Sigmoid Neural Network created.\n');
        
    case 3  % NLARX with Neural Network (default activation - typically ReLU)
        fprintf('Model Type: NLARX with Neural Network (ReLU activation)\n');
        activation = 'relu';
        layerSizes = [8 5];  % 2 hidden layers: 8 and 5 units (reduced for stability)
        fprintf('Activation function: %s\n', activation);
        fprintf('Layer sizes: %s\n', mat2str(layerSizes));
        
        % Create feedforward neural network with ReLU activation
        f = idNeuralNetwork(layerSizes, activation, 0, 0);
        fprintf('Neural Network (ReLU) created.\n');
end

% =========================================
% REGRESSOR ORDERS CONFIGURATION
% =========================================
fprintf('\n--- Regressor Orders Selection ---\n');
fprintf('Regressor orders [na, nb, nk]:\n');
fprintf('  na = output lag order\n');
fprintf('  nb = input lag order\n');
fprintf('  nk = dead time (delay)\n\n');

fprintf('Preset options:\n');
fprintf('  1: [2, 2, 0] - minimal orders\n');
fprintf('  2: [2, 2, 1] - minimal with delay\n');
fprintf('  3: [3, 3, 0] - moderate orders\n');
fprintf('  4: [3, 3, 1] - moderate with delay (recommended)\n');
fprintf('  5: [4, 4, 1] - higher order\n\n');

orderChoice = input('Select regressor orders (1-5, default 4): ');
if isempty(orderChoice)
    orderChoice = 4;
end

orderOptions = {[2, 2, 0], [2, 2, 1], [3, 3, 0], [3, 3, 1], [4, 4, 1]};
if orderChoice >= 1 && orderChoice <= length(orderOptions)
    orders = orderOptions{orderChoice};
else
    orders = [3, 3, 1];  % Default fallback
    fprintf('Invalid choice. Using default: [3, 3, 1]\n');
end

fprintf('Selected orders: [%d, %d, %d]\n\n', orders(1), orders(2), orders(3));

% =========================================
% TRAINING OPTIONS - WITHOUT CROSS-VALIDATION
% =========================================
fprintf('--- Training Phase: WITHOUT Cross-Validation ---\n');

opt2 = nlarxOptions;
opt2.Focus = 'simulation';
opt2.Display = 'on';
opt2.SearchOptions.MaxIterations = 20;
opt2.NormalizationOptions.NormalizationMethod = 'zscore';
opt2.CrossValidate = false;  % Disable cross-validation

fprintf('Training model without cross-validation...\n');
sys2 = nlarx(dataTraining, orders, f, opt2);
fprintf('Training completed.\n\n');

% Get output function structure
outputFcn = sys2.OutputFcn;
fprintf('Output function: %s\n', class(outputFcn));

% Evaluate performance
fprintf('Evaluating performance...\n');
[yhat_tr, fit_tr, yhat_va, fit_va, yhat_tr_raw, yhat_va_raw] = evaluateNLARXPerformance(dataTraining, dataValidation, sys2, norm_stats);

% Check for NaN values in output
if any(isnan(yhat_tr_raw)) || any(isnan(yhat_va_raw))
    fprintf('\nWarning: Model simulation produced NaN values.\n');
    fprintf('This may indicate numerical instability in the model training.\n');
    fprintf('Skipping performance visualization.\n\n');
    % Skip further processing
    return;
end

rmse_tr = calculateRMSE(Ytr_raw, yhat_tr_raw);
rmse_va = calculateRMSE(Yva_raw, yhat_va_raw);

fprintf('  Training Fit: %.2f%%\n', fit_tr);
fprintf('  Training RMSE (raw data): %.6g\n', rmse_tr);
fprintf('  Validation Fit: %.2f%%\n', fit_va);
fprintf('  Validation RMSE (raw data): %.6g\n\n', rmse_va);

% Create log directory if it doesn't exist
% Log directory includes both dataset and model type for organization
modelDirName = '';
switch modelChoice
    case 1
        modelDirName = 'cascadeCorrelation';
    case 2
        modelDirName = 'nlarxSigmoid';
    case 3
        modelDirName = 'nlarxReLU';
end

logDir = fullfile(scriptDir, 'logs', datasetName, modelDirName);
if ~exist(logDir, 'dir')
    mkdir(logDir);
end

% Plotting results (using raw data and predictions)
fig1 = figure('Name', 'Training Fit', 'Color', 'w');
plot(Ytr_raw, 'k', 'LineWidth', 1.4); hold on;
plot(yhat_tr_raw, 'b--', 'LineWidth', 1.2); grid on;
title(sprintf('%s - %s - Training Data (Fit=%.2f%%, RMSE=%.4g)', datasetName, modelName, fit_tr, rmse_tr));
legend('True Data', 'Model Simulation', 'Location', 'best');
trainingFigPath = fullfile(logDir, sprintf('%s_%s_Training_Fit.png', datasetName, modelDirName));
saveas(fig1, trainingFigPath, 'png');
fprintf('Training figure saved: %s\n', trainingFigPath);

fig2 = figure('Name', 'Validation Fit', 'Color', 'w');
plot(Yva_raw, 'k', 'LineWidth', 1.4); hold on;
plot(yhat_va_raw, 'r--', 'LineWidth', 1.2); grid on;
title(sprintf('%s - %s - Validation Data (Fit=%.2f%%, RMSE=%.4g)', datasetName, modelName, fit_va, rmse_va));
legend('True Data', 'Model Simulation', 'Location', 'best');
validationFigPath = fullfile(logDir, sprintf('%s_%s_Validation_Fit.png', datasetName, modelDirName));
saveas(fig2, validationFigPath, 'png');
fprintf('Validation figure saved: %s\n', validationFigPath);

% Plotting loss metrics (MSE-based)
fig3 = figure('Name', 'Loss Metrics', 'Color', 'w');
mse_tr = mean((Ytr_raw - yhat_tr_raw).^2);
mse_va = mean((Yva_raw - yhat_va_raw).^2);
metrics = [mse_tr, mse_va];

% Remove NaN values for plotting
metrics(isnan(metrics)) = 0;

bar([1, 2], metrics, 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
xlabel('Data Set'); ylabel('MSE (Mean Squared Error)');
set(gca, 'XTickLabel', {'Training', 'Validation'});
title(sprintf('%s - %s - Loss Metrics (MSE)', datasetName, modelName));

% Set ylim safely (check for valid values)
if ~all(isnan(metrics)) && max(metrics) > 0
    ylim([0, max(metrics)*1.2]);
else
    ylim([0, 1]);
end

for i = 1:2
    if ~isnan(metrics(i))
        text(i, metrics(i) + max(metrics(~isnan(metrics)))*0.05, sprintf('%.6g', metrics(i)), 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
end
grid on; set(gca, 'GridLineStyle', ':');
lossFigPath = fullfile(logDir, sprintf('%s_%s_Loss_Metrics.png', datasetName, modelDirName));
saveas(fig3, lossFigPath, 'png');
fprintf('Loss metrics figure saved: %s\n', lossFigPath);

% =========================================
% TRAINING SUMMARY
% =========================================
fprintf('\n=====================================\n');
fprintf('  TRAINING SUMMARY\n');
fprintf('=====================================\n\n');

fprintf('Training Results:\n');
fprintf('  Train Fit: %.2f%% | Train RMSE (raw): %.6g\n', fit_tr, rmse_tr);
fprintf('  Val Fit:   %.2f%% | Val RMSE (raw):   %.6g\n\n', fit_va, rmse_va);

% =========================================
% LOG RESULTS
% =========================================
fprintf('\nSaving training logs...\n\n');

% Log training information
trainInfo = struct();
trainInfo.dataset = datasetName;
trainInfo.modelType = modelName;
trainInfo.modelChoice = modelChoice;
trainInfo.activation = activation;
trainInfo.orders = orders;
trainInfo.crossValidation = false;
trainInfo.trainFit = fit_tr;
trainInfo.trainRMSE = rmse_tr;
trainInfo.valFit = fit_va;
trainInfo.valRMSE = rmse_va;
trainInfo.trainSamples = length(Ytr_raw);
trainInfo.valSamples = length(Yva_raw);
trainInfo.normMethod = 'zscore';
trainInfo.normStats = norm_stats;
trainInfo.trainingFigure = trainingFigPath;
trainInfo.validationFigure = validationFigPath;
trainInfo.lossFigure = lossFigPath;
trainInfo.notes = sprintf('Model: %s. Trained without cross-validation (Z-score normalized). All figures saved as PNG.', modelName);


logPath = writeNLARXLog(fullfile('logs', datasetName, modelDirName), datasetName, trainInfo);
fprintf('Log saved: %s\n', logPath);

fprintf('\nTraining completed successfully!\n');
fprintf('Results are saved in the logs folder.\n');
