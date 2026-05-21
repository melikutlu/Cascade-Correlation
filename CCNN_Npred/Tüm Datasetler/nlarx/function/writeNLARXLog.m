function logFilePath = writeNLARXLog(baseLogDir, datasetName, trainInfo)
    % WRITENLARXLOG Write NLARX training results to log file
    % trainInfo should contain: trainRMSE, valRMSE, trainFit, valFit, epochs, 
    % orders, activation, maxHiddenUnits, actualHiddenUnits, etc.
    
    if nargin < 1 || isempty(baseLogDir)
        baseLogDir = 'logs';
    end
    if nargin < 2 || isempty(datasetName)
        datasetName = 'unknown';
    end
    
    % Create dataset-specific log directory
    datasetLogDir = fullfile(baseLogDir, datasetName);
    if ~exist(datasetLogDir, 'dir')
        mkdir(datasetLogDir);
    end
    
    % Generate timestamp-based filename
    timestamp = datetime('now', 'Format', 'yyMMdd_HHmmss');
    logFilename = sprintf('nlarx_%s_%s.txt', datasetName, timestamp);
    logFilePath = fullfile(datasetLogDir, logFilename);
    
    % Open file for writing
    fid = fopen(logFilePath, 'w');
    if fid == -1
        error('Could not open log file: %s', logFilePath);
    end
    
    % Write header
    fprintf(fid, '=====================================\n');
    fprintf(fid, '    NLARX Training Results Log\n');
    fprintf(fid, '=====================================\n');
    fprintf(fid, 'Dataset: %s\n', datasetName);
    fprintf(fid, 'Timestamp: %s\n', datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    fprintf(fid, '\n');
    
    % Write configuration
    fprintf(fid, '--- Configuration ---\n');
    if isfield(trainInfo, 'activation')
        fprintf(fid, 'Activation: %s\n', trainInfo.activation);
    end
    if isfield(trainInfo, 'orders')
        fprintf(fid, 'Orders: %s\n', mat2str(trainInfo.orders));
    end
    if isfield(trainInfo, 'maxHiddenUnits')
        fprintf(fid, 'Max Hidden Units: %d\n', trainInfo.maxHiddenUnits);
    end
    if isfield(trainInfo, 'actualHiddenUnits')
        fprintf(fid, 'Actual Hidden Units Added: %d\n', trainInfo.actualHiddenUnits);
    end
    if isfield(trainInfo, 'maxIterations')
        fprintf(fid, 'Max Iterations: %d\n', trainInfo.maxIterations);
    end
    fprintf(fid, '\n');
    
    % Write performance metrics
    fprintf(fid, '--- Performance Metrics ---\n');
    if isfield(trainInfo, 'trainRMSE')
        fprintf(fid, 'Training RMSE: %.6g\n', trainInfo.trainRMSE);
    end
    if isfield(trainInfo, 'valRMSE')
        fprintf(fid, 'Validation RMSE: %.6g\n', trainInfo.valRMSE);
    end
    if isfield(trainInfo, 'trainFit')
        fprintf(fid, 'Training Fit: %.2f%%\n', trainInfo.trainFit);
    end
    if isfield(trainInfo, 'valFit')
        fprintf(fid, 'Validation Fit: %.2f%%\n', trainInfo.valFit);
    end
    if isfield(trainInfo, 'trainMSE')
        fprintf(fid, 'Training MSE: %.6g\n', trainInfo.trainMSE);
    end
    if isfield(trainInfo, 'valMSE')
        fprintf(fid, 'Validation MSE: %.6g\n', trainInfo.valMSE);
    end
    fprintf(fid, '\n');
    
    % Write data information
    fprintf(fid, '--- Data Information ---\n');
    if isfield(trainInfo, 'trainSamples')
        fprintf(fid, 'Training Samples: %d\n', trainInfo.trainSamples);
    end
    if isfield(trainInfo, 'valSamples')
        fprintf(fid, 'Validation Samples: %d\n', trainInfo.valSamples);
    end
    fprintf(fid, '\n');
    
    % Write additional details
    fprintf(fid, '--- Additional Details ---\n');
    if isfield(trainInfo, 'notes')
        fprintf(fid, 'Notes: %s\n', trainInfo.notes);
    end
    
    fprintf(fid, '=====================================\n');
    
    fclose(fid);
    
    % Also save as MAT file for later analysis
    matFilename = strrep(logFilename, '.txt', '.mat');
    matFilePath = fullfile(datasetLogDir, matFilename);
    save(matFilePath, 'trainInfo');
end
