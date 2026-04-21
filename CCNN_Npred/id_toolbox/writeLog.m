function results = writeLog(results, config, dataInfo)

    if ~exist(config.logFolder, 'dir')
        mkdir(config.logFolder);
    end

    runName = sprintf('%s_%s_%s_%s', ...
        dataInfo.name, lower(config.modelType), lower(config.mode), results.timestamp);

    runFolder = fullfile(config.logFolder, runName);
    if ~exist(runFolder, 'dir')
        mkdir(runFolder);
    end

    % results içine ekle
    results.runFolder = runFolder;

    %% save figures
    figPaths = {};
    if isfield(results, 'figHandles') && ~isempty(results.figHandles) && config.savePlots
        if length(results.figHandles) >= 1 && isgraphics(results.figHandles(1))
            p1 = fullfile(runFolder, 'output_comparison.png');
            saveas(results.figHandles(1), p1);
            figPaths{end+1} = p1;
        end
        if length(results.figHandles) >= 2 && isgraphics(results.figHandles(2))
            p2 = fullfile(runFolder, 'error_signal.png');
            saveas(results.figHandles(2), p2);
            figPaths{end+1} = p2;
        end
    end

    results.figPaths = figPaths;

    %% save mat file
    % Figure handle'ları save etme
    resultsToSave = results;
    if isfield(resultsToSave, 'figHandles')
        resultsToSave = rmfield(resultsToSave, 'figHandles');
    end

    save(fullfile(runFolder, 'results.mat'), 'resultsToSave', 'config', 'dataInfo');

    %% text log
    logFile = fullfile(runFolder, 'run_log.txt');
    fid = fopen(logFile, 'w');

    if fid == -1
        warning('Could not create log file.');
        return;
    end

    fprintf(fid, '============================================\n');
    fprintf(fid, 'SYSTEM IDENTIFICATION RUN LOG\n');
    fprintf(fid, '============================================\n\n');

    fprintf(fid, 'Timestamp           : %s\n', results.timestamp);
    fprintf(fid, 'Dataset             : %s\n', dataInfo.name);
    fprintf(fid, 'Model type          : %s\n', config.modelType);
    fprintf(fid, 'Mode                : %s\n', config.mode);
    fprintf(fid, 'Sampling time       : %.6f\n', dataInfo.Ts);
    fprintf(fid, 'Train samples       : %d\n', dataInfo.Ntrain);
    fprintf(fid, 'Validation samples  : %d\n\n', dataInfo.Nval);

    if strcmpi(config.modelType, 'tf')
        fprintf(fid, 'TF numerator order  : %d\n', config.tf_num_order);
        fprintf(fid, 'TF denominator order: %d\n', config.tf_den_order);
        fprintf(fid, 'TF input delay      : %d\n\n', config.tf_io_delay);
    elseif strcmpi(config.modelType, 'ss')
        fprintf(fid, 'SS order            : %d\n\n', config.ss_order);
    end

    fprintf(fid, 'RMSE                : %.10f\n', results.rmse);
    fprintf(fid, 'MSE                 : %.10f\n', results.mse);
    fprintf(fid, 'MAE                 : %.10f\n', results.mae);
    fprintf(fid, 'Fit (%%)             : %.10f\n\n', results.fit);

    fprintf(fid, '--------------------------------------------\n');
    fprintf(fid, 'IDENTIFIED MODEL SUMMARY\n');
    fprintf(fid, '--------------------------------------------\n');
    fprintf(fid, '%s\n', results.modelText);

    fprintf(fid, '\n--------------------------------------------\n');
    fprintf(fid, 'SAVED FIGURES\n');
    fprintf(fid, '--------------------------------------------\n');
    if isempty(figPaths)
        fprintf(fid, 'No figures saved.\n');
    else
        for i = 1:numel(figPaths)
            fprintf(fid, '%s\n', figPaths{i});
        end
    end

    fclose(fid);

    fprintf('\nLog written to: %s\n', runFolder);
end