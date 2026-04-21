function appendFigureInfoToLog(logFilePath, savedPaths)
    if isempty(logFilePath) || isempty(savedPaths)
        return;
    end
    fid = fopen(logFilePath, 'a');
    if fid == -1
        warning('Could not append figure info to %s', logFilePath);
        return;
    end
    fprintf(fid, '\nSaved figure files:\n');
    [logDir, ~, ~] = fileparts(logFilePath);
    for i = 1:numel(savedPaths)
        relPath = savedPaths{i};
        if isstring(relPath)
            relPath = relPath{1};
        end
        prefix = [logDir filesep];
        if strncmp(relPath, prefix, numel(prefix))
            relPath = relPath(numel(prefix)+1:end);
        end
        fprintf(fid, ' - %s\n', relPath);
    end
    fclose(fid);
end
