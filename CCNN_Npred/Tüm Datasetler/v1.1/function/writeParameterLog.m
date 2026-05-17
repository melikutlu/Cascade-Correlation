function logFilePath = writeParameterLog(config, logInfo)
    timestampStr = datestr(now,'yyyy-mm-dd HH:MM:SS');
    fileStamp = datestr(now,'yyyymmdd_HHMMSS');
    scriptFullPath = mfilename('fullpath');
    if isempty(scriptFullPath)
        scriptDir = pwd;
        scriptBase = 'CCNN_Npred';
    else
        [scriptDir, scriptBase] = fileparts(scriptFullPath);
    end

    projectDir = fileparts(scriptDir);
    if isfield(config, 'data') && isfield(config.data, 'source_label') && ~isempty(config.data.source_label)
        dataLabel = char(config.data.source_label);
    else
        dataLabel = char(config.data.source);
    end
    dataLabel = regexprep(dataLabel, '[^A-Za-z0-9_-]', '_');

    % Fit değerlerini hazırla (folder adı sade kalsın)
    fitTrStr = sprintf('%d', round(logInfo.fit_train));
    fitVaStr = sprintf('%d', round(logInfo.fit_val));
    
    runFolderName = sprintf('fitTr%s_fitVa%s', fitTrStr, fitVaStr);
    runFolderPath = fullfile(projectDir, 'logs', dataLabel, runFolderName);
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

    logFileName = sprintf('%s_%s.log', runFolderName, fileStamp);
    logFilePath = fullfile(runFolderPath, logFileName);
    fid = fopen(logFilePath,'w');
    if fid == -1
        warning('Could not create parameter log at %s', logFilePath);
        logFilePath = '';
        return;
    end

    fprintf(fid, 'Data source : %s\n', dataLabel);
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

    fprintf(fid, 'Data source   : %s\n', dataLabel);
    if isfield(config.data, 'twotank')
        if isfield(config.data.twotank, 'sampling_time')
            fprintf(fid, 'Twotank sampling time : %.6g\n', config.data.twotank.sampling_time);
        end
        if isfield(config.data.twotank, 'warmup_samples')
            fprintf(fid, 'Twotank warmup samples : %d\n', config.data.twotank.warmup_samples);
        end
        if isfield(config.data.twotank, 'filter_cutoff')
            fprintf(fid, 'Twotank filter cutoff  : %.6g\n', config.data.twotank.filter_cutoff);
        end
    end
    if isfield(config.data, 'dryer2') && isfield(config.data.dryer2, 'sampling_time')
        fprintf(fid, 'Dryer2 sampling time   : %.6g\n', config.data.dryer2.sampling_time);
    end
    if isfield(config.data, 'robotarm')
        if isfield(config.data.robotarm, 'sampling_time')
            fprintf(fid, 'Robot arm sampling time : %.6g\n', config.data.robotarm.sampling_time);
        end
        if isfield(config.data.robotarm, 'validation_experiment')
            fprintf(fid, 'Robot arm validation experiment : %d\n', config.data.robotarm.validation_experiment);
        end
    end

    fprintf(fid, 'N-step horizon : %d\n', logInfo.n_steps);
    fprintf(fid, 'Train obj MSE  : %.6g\n', logInfo.train_mse);
    if isfield(logInfo, 'val_mse')
        fprintf(fid, 'Val   obj MSE  : %.6g\n', logInfo.val_mse);
    end
    fprintf(fid, 'Train series RMSE : %.6g\n', logInfo.rmse_train);
    fprintf(fid, 'Val   RMSE      : %.6g\n', logInfo.rmse_val);
    fprintf(fid, 'Train Fit (%%)   : %.2f\n', logInfo.fit_train);
    fprintf(fid, 'Val   Fit (%%)   : %.2f\n\n', logInfo.fit_val);

    if isfield(logInfo, 'hidden_stage_counts') && ~isempty(logInfo.hidden_stage_counts)
        fprintf(fid, 'Hidden units per stage : %s\n', formatArrayField(logInfo.hidden_stage_counts));
    end
    if isfield(logInfo, 'train_mse_history') && ~isempty(logInfo.train_mse_history)
        fprintf(fid, 'Train obj MSE history  : %s\n', formatArrayField(logInfo.train_mse_history));
    end
    if isfield(logInfo, 'val_mse_history') && ~isempty(logInfo.val_mse_history)
        fprintf(fid, 'Val   obj MSE history  : %s\n', formatArrayField(logInfo.val_mse_history));
    end
    if isfield(logInfo, 'train_rmse_history') && ~isempty(logInfo.train_rmse_history)
        fprintf(fid, 'Train RMSE history     : %s\n', formatArrayField(logInfo.train_rmse_history));
    end
    if isfield(logInfo, 'val_rmse_history') && ~isempty(logInfo.val_rmse_history)
        fprintf(fid, 'Val   RMSE history     : %s\n', formatArrayField(logInfo.val_rmse_history));
    end
    if isfield(logInfo, 'train_fit_history') && ~isempty(logInfo.train_fit_history)
        fprintf(fid, 'Train Fit history (%%)  : %s\n', formatArrayField(logInfo.train_fit_history));
    end
    if isfield(logInfo, 'val_fit_history') && ~isempty(logInfo.val_fit_history)
        fprintf(fid, 'Val   Fit history (%%)  : %s\n', formatArrayField(logInfo.val_fit_history));
    end
    if isfield(logInfo, 'best_validation_stage_index')
        fprintf(fid, 'Best validation stage  : #%d (hidden=%d)\n', logInfo.best_validation_stage_index, logInfo.best_validation_stage_hidden_units);
        fprintf(fid, 'Best validation metric : %s = %.6g\n', logInfo.best_validation_selection_metric, logInfo.best_validation_score_value);
    end
    if isfield(logInfo, 'hidden_growth_reverted_to_baseline') && logInfo.hidden_growth_reverted_to_baseline
        fprintf(fid, 'Hidden growth reverted to baseline during search.\n');
    end
    if isfield(logInfo, 'hidden_stage_counts') && isfield(logInfo, 'best_validation_stage_index') ...
            && logInfo.best_validation_stage_index < numel(logInfo.hidden_stage_counts)
        fprintf(fid, 'Final model note : selected from an earlier validation stage, not the last hidden-added stage.\n');
    end

    % include per-unit MSE history so the final selected prefix is visible in the log
    if isfield(logInfo,'mse_history') && ~isempty(logInfo.mse_history)
        fprintf(fid, 'MSE history (final selected prefix): %s\n', formatArrayField(logInfo.mse_history));
    end

    fprintf(fid, 'Regressors.u : %s\n', mat2str(logInfo.regressors_u));
    fprintf(fid, 'Regressors.y : %s\n', mat2str(logInfo.regressors_y));
    fprintf(fid, 'Norm method  : %s\n', config.norm_method);
    fprintf(fid, 'Activation   : %s\n', logInfo.activation);
    activationClipping = 1;
    if isfield(config.model, 'use_activation_clipping') && ~isempty(config.model.use_activation_clipping)
        activationClipping = double(logical(config.model.use_activation_clipping));
    end
    fprintf(fid, 'Activation clipping : %d\n', activationClipping);
    if isfield(config.model, 'tustin_sample_time')
        fprintf(fid, 'Tustin sample time : %.6g\n', config.model.tustin_sample_time);
    end
    fprintf(fid, 'Diff clip lower : %.6g\n', config.model.diff_clip_lower);
    fprintf(fid, 'Diff clip upper : %.6g\n', config.model.diff_clip_upper);
    if isfield(config.model, 'sim_loss_eval_interval')
        fprintf(fid, 'Sim loss eval interval : %d\n', config.model.sim_loss_eval_interval);
    end
    if isfield(config.model, 'sim_loss_min_blocks')
        fprintf(fid, 'Sim loss min blocks : %d\n', config.model.sim_loss_min_blocks);
    end
    if isfield(config.model, 'output_max_epochs')
        fprintf(fid, 'Output max epochs : %d\n', config.model.output_max_epochs);
    end
    fprintf(fid, 'Target MSE   : %.6g\n', config.model.target_mse);
    fprintf(fid, 'Plateau min delta  : %.3g\n', logInfo.plateau_min_delta);
    fprintf(fid, 'Moving avg window  : %d\n', logInfo.moving_avg_window);
    fprintf(fid, 'Hidden bootstrap count : %d\n', logInfo.hidden_bootstrap_count);
    fprintf(fid, 'Hidden acceptance window : %d\n', logInfo.hidden_acceptance_window);

    fclose(fid);
end
