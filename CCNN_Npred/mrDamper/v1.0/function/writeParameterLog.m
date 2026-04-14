function logFilePath = writeParameterLog(config, logInfo)
    timestampStr = datestr(now,'yyyy-mm-dd HH:MM:SS');
    scriptFullPath = mfilename('fullpath');
    if isempty(scriptFullPath)
        scriptDir = pwd;
        scriptBase = 'CCNN_Npred';
    else
        [scriptDir, scriptBase] = fileparts(scriptFullPath);
    end

        % Fit değerlerini hazırla (noktaları 'p' ile değiştir)
    fitTrStr = strrep(sprintf('%.1f', logInfo.fit_train), '.', 'p');
    fitVaStr = strrep(sprintf('%.1f', logInfo.fit_val), '.', 'p');
    
    % Sadece fit değerlerinden oluşan descriptor
    descriptor = sprintf('fitTr%s_fitVa%s', fitTrStr, fitVaStr);
    
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
        if isfield(logInfo,'output_stop_by_mavg')
            fprintf(fid, 'Output stopped by moving-avg : %d\n', logInfo.output_stop_by_mavg);
        end

    candEpochStr = formatArrayField(logInfo.candidate_epochs_used);
    candPlateauStr = formatArrayField(logInfo.candidate_plateau_epochs);
    fprintf(fid, 'Candidate epochs (per unit)  : %s\n', candEpochStr);
    fprintf(fid, 'Candidate plateau epochs      : %s\n', candPlateauStr);

    fprintf(fid, 'N-step horizon : %d\n', logInfo.n_steps);
    fprintf(fid, 'Train obj MSE  : %.6g\n', logInfo.train_mse);
    fprintf(fid, 'Train series RMSE : %.6g\n', logInfo.rmse_train);
    fprintf(fid, 'Val   RMSE      : %.6g\n', logInfo.rmse_val);
    fprintf(fid, 'Train Fit (%%)   : %.2f\n', logInfo.fit_train);
    fprintf(fid, 'Val   Fit (%%)   : %.2f\n\n', logInfo.fit_val);

    % include per-unit MSE history so the log contains the loss progression
    if isfield(logInfo,'mse_history') && ~isempty(logInfo.mse_history)
        fprintf(fid, 'MSE history (per hidden unit added): %s\n', formatArrayField(logInfo.mse_history));
    end

    fprintf(fid, 'Regressors.u : %s\n', mat2str(logInfo.regressors_u));
    fprintf(fid, 'Regressors.y : %s\n', mat2str(logInfo.regressors_y));
    fprintf(fid, 'Norm method  : %s\n', config.norm_method);
    fprintf(fid, 'Activation   : %s\n', logInfo.activation);
    fprintf(fid, 'Diff clip lower : %.6g\n', config.model.diff_clip_lower);
    fprintf(fid, 'Diff clip upper : %.6g\n', config.model.diff_clip_upper);
    fprintf(fid, 'Target MSE   : %.6g\n', config.model.target_mse);
    fprintf(fid, 'Plateau min delta  : %.3g\n', logInfo.plateau_min_delta);
    fprintf(fid, 'Moving avg window  : %d\n', logInfo.moving_avg_window);

    fclose(fid);
end
