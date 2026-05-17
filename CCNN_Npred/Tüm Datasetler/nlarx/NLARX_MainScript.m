% NLARX Training Script with Dataset Selection
% This script uses MATLAB's idNeuralNetwork and nlarx for modeling
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
fprintf('  NLARX Training - Dataset Selection\n');
fprintf('=====================================\n\n');

availableDatasets = {'twotankdata', 'dryer2', 'mrdamper'};
fprintf('Available datasets:\n');
for i = 1:length(availableDatasets)
    fprintf('  %d: %s\n', i, availableDatasets{i});
end
fprintf('\n');

datasetChoice = input('Select dataset (1-3): ');
if datasetChoice < 1 || datasetChoice > length(availableDatasets)
    error('Invalid choice. Please select 1-3.');
end
datasetName = availableDatasets{datasetChoice};
fprintf('\nSelected: %s\n\n', datasetName);

% =========================================
% CONFIGURE DATA LOADING
% =========================================
% Setup configuration (same structure as cascade correlation code)
config = struct();
config.data.source = datasetName;
config.data.train_ratio = 0.5;
config.data.val_ratio = 0.5;
config.norm_method = 'zscore';  % Z-score normalization

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

% Neural network setup: cascade-correlation, sigmoid
activation = 'sigmoid';
maxHiddenUnits = 20;
fprintf('Activation function: %s\n', activation);
fprintf('Max hidden units: %d\n', maxHiddenUnits);

% Create network
f = idNeuralNetwork("cascade-correlation", activation, 0, 0, ...
    MaxNumActLayers=maxHiddenUnits, SizeSelection='off');
fprintf('Neural network created.\n');

% Regressor orders: [na, nb, nk]
% na = output lag order
% nb = input lag order  
% nk = dead time (delay)
orders = [3, 3, 1];  % Typical choice
fprintf('Regressor orders (na, nb, nk): [%d, %d, %d]\n\n', orders(1), orders(2), orders(3));

% =========================================
% TRAINING OPTIONS - WITH CROSS-VALIDATION
% =========================================
fprintf('--- Training Phase 1: WITH Cross-Validation ---\n');

opt1 = nlarxOptions;
opt1.SearchOptions.MaxIterations = 0;  % Let algorithm decide iterations
opt1.NormalizationOptions.NormalizationMethod = 'zscore';
opt1.CrossValidationOptions.HoldoutFraction = 0.1;

fprintf('Training model with cross-validation...\n');
sys1 = nlarx(dataTraining, orders, f, opt1);
fprintf('Training completed.\n\n');

% Get output function structure
outputFcn1 = sys1.OutputFcn;
fprintf('Output function (CV): %s\n', class(outputFcn1));

% Evaluate performance with cross-validation
fprintf('Evaluating performance with cross-validation...\n');
[yhat_tr1, fit_tr1, yhat_va1, fit_va1, yhat_tr1_raw, yhat_va1_raw] = evaluateNLARXPerformance(dataTraining, dataValidation, sys1, norm_stats);
rmse_tr1 = calculateRMSE(Ytr_raw, yhat_tr1_raw);
rmse_va1 = calculateRMSE(Yva_raw, yhat_va1_raw);

fprintf('  Training Fit: %.2f%%\n', fit_tr1);
fprintf('  Training RMSE (raw data): %.6g\n', rmse_tr1);
fprintf('  Validation Fit: %.2f%%\n', fit_va1);
fprintf('  Validation RMSE (raw data): %.6g\n\n', rmse_va1);

% Plotting results with CV (using raw data and predictions)
figure('Name', 'Training Fit - WITH CV', 'Color', 'w');
plot(Ytr_raw, 'k', 'LineWidth', 1.4); hold on;
plot(yhat_tr1_raw, 'b--', 'LineWidth', 1.2); grid on;
title(sprintf('%s - Training Data (WITH CV, Fit=%.2f%%, RMSE=%.4g)', datasetName, fit_tr1, rmse_tr1));
legend('True Data', 'Model Simulation', 'Location', 'best');

figure('Name', 'Validation Fit - WITH CV', 'Color', 'w');
plot(Yva_raw, 'k', 'LineWidth', 1.4); hold on;
plot(yhat_va1_raw, 'r--', 'LineWidth', 1.2); grid on;
title(sprintf('%s - Validation Data (WITH CV, Fit=%.2f%%, RMSE=%.4g)', datasetName, fit_va1, rmse_va1));
legend('True Data', 'Model Simulation', 'Location', 'best');

% =========================================
% TRAINING OPTIONS - WITHOUT CROSS-VALIDATION
% =========================================
fprintf('--- Training Phase 2: WITHOUT Cross-Validation ---\n');

opt2 = nlarxOptions;
opt2.SearchOptions.MaxIterations = 0;
opt2.NormalizationOptions.NormalizationMethod = 'zscore';
opt2.CrossValidate = false;  % Disable cross-validation

fprintf('Training model without cross-validation...\n');
sys2 = nlarx(dataTraining, orders, f, opt2);
fprintf('Training completed.\n\n');

% Get output function structure
outputFcn2 = sys2.OutputFcn;
fprintf('Output function (No CV): %s\n', class(outputFcn2));

% Evaluate performance without cross-validation
fprintf('Evaluating performance without cross-validation...\n');
[yhat_tr2, fit_tr2, yhat_va2, fit_va2, yhat_tr2_raw, yhat_va2_raw] = evaluateNLARXPerformance(dataTraining, dataValidation, sys2, norm_stats);
rmse_tr2 = calculateRMSE(Ytr_raw, yhat_tr2_raw);
rmse_va2 = calculateRMSE(Yva_raw, yhat_va2_raw);

fprintf('  Training Fit: %.2f%%\n', fit_tr2);
fprintf('  Training RMSE (raw data): %.6g\n', rmse_tr2);
fprintf('  Validation Fit: %.2f%%\n', fit_va2);
fprintf('  Validation RMSE (raw data): %.6g\n\n', rmse_va2);

% Plotting results without CV (using raw data and predictions)
figure('Name', 'Training Fit - WITHOUT CV', 'Color', 'w');
plot(Ytr_raw, 'k', 'LineWidth', 1.4); hold on;
plot(yhat_tr2_raw, 'b--', 'LineWidth', 1.2); grid on;
title(sprintf('%s - Training Data (WITHOUT CV, Fit=%.2f%%, RMSE=%.4g)', datasetName, fit_tr2, rmse_tr2));
legend('True Data', 'Model Simulation', 'Location', 'best');

figure('Name', 'Validation Fit - WITHOUT CV', 'Color', 'w');
plot(Yva_raw, 'k', 'LineWidth', 1.4); hold on;
plot(yhat_va2_raw, 'r--', 'LineWidth', 1.2); grid on;
title(sprintf('%s - Validation Data (WITHOUT CV, Fit=%.2f%%, RMSE=%.4g)', datasetName, fit_va2, rmse_va2));
legend('True Data', 'Model Simulation', 'Location', 'best');

% =========================================
% COMPARISON SUMMARY
% =========================================
fprintf('\n=====================================\n');
fprintf('  COMPARISON SUMMARY\n');
fprintf('=====================================\n\n');

fprintf('WITH Cross-Validation:\n');
fprintf('  Train Fit: %.2f%% | Train RMSE (raw): %.6g\n', fit_tr1, rmse_tr1);
fprintf('  Val Fit:   %.2f%% | Val RMSE (raw):   %.6g\n\n', fit_va1, rmse_va1);

fprintf('WITHOUT Cross-Validation:\n');
fprintf('  Train Fit: %.2f%% | Train RMSE (raw): %.6g\n', fit_tr2, rmse_tr2);
fprintf('  Val Fit:   %.2f%% | Val RMSE (raw):   %.6g\n\n', fit_va2, rmse_va2);

% =========================================
% LOG RESULTS
% =========================================
fprintf('Saving training logs...\n\n');

% Log WITH cross-validation
trainInfo1 = struct();
trainInfo1.dataset = datasetName;
trainInfo1.activation = activation;
trainInfo1.maxHiddenUnits = maxHiddenUnits;
trainInfo1.orders = orders;
trainInfo1.crossValidation = true;
trainInfo1.trainFit = fit_tr1;
trainInfo1.trainRMSE = rmse_tr1;
trainInfo1.valFit = fit_va1;
trainInfo1.valRMSE = rmse_va1;
trainInfo1.trainSamples = length(Ytr_raw);
trainInfo1.valSamples = length(Yva_raw);
trainInfo1.normMethod = 'zscore';
trainInfo1.normStats = norm_stats;
trainInfo1.notes = 'Model trained WITH cross-validation (Z-score normalized)';

logPath1 = writeNLARXLog('logs', [datasetName '_CV'], trainInfo1);
fprintf('Log saved: %s\n', logPath1);

% Log WITHOUT cross-validation
trainInfo2 = struct();
trainInfo2.dataset = datasetName;
trainInfo2.activation = activation;
trainInfo2.maxHiddenUnits = maxHiddenUnits;
trainInfo2.orders = orders;
trainInfo2.crossValidation = false;
trainInfo2.trainFit = fit_tr2;
trainInfo2.trainRMSE = rmse_tr2;
trainInfo2.valFit = fit_va2;
trainInfo2.valRMSE = rmse_va2;
trainInfo2.trainSamples = length(Ytr_raw);
trainInfo2.valSamples = length(Yva_raw);
trainInfo2.normMethod = 'zscore';
trainInfo2.normStats = norm_stats;
trainInfo2.notes = 'Model trained WITHOUT cross-validation (Z-score normalized)';

logPath2 = writeNLARXLog('logs', [datasetName '_NoCV'], trainInfo2);
fprintf('Log saved: %s\n', logPath2);

fprintf('\nTraining completed successfully!\n');
fprintf('Results are saved in the logs folder.\n');
