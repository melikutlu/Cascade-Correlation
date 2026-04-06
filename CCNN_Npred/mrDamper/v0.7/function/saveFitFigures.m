function savedPaths = saveFitFigures(logFilePath, figMap)
    savedPaths = {};
    if nargin < 2 || isempty(figMap)
        return;
    end

    if isempty(logFilePath)
        scriptFullPath = mfilename('fullpath');
        if isempty(scriptFullPath)
            targetDir = pwd;
            baseName = sprintf('CCNN_Npred_%s', datestr(now,'yyyymmdd_HHMMSS'));
        else
            [targetDir, scriptBase] = fileparts(scriptFullPath);
            baseName = sprintf('%s_%s', scriptBase, datestr(now,'yyyymmdd_HHMMSS'));
        end
    else
        [targetDir, baseName] = fileparts(logFilePath);
    end

    labels = fieldnames(figMap);
    for k = 1:numel(labels)
        figHandle = figMap.(labels{k});
        if isempty(figHandle) || ~ishandle(figHandle)
            continue;
        end
        cleanLabel = lower(regexprep(labels{k},'[^A-Za-z0-9]',''));
        if isempty(cleanLabel)
            cleanLabel = sprintf('fig%d', k);
        end
        fileName = sprintf('%s_%s_fit.png', baseName, cleanLabel);
        filePath = fullfile(targetDir, fileName);
        try
            exportgraphics(figHandle, filePath, 'Resolution', 150);
        catch
            try
                saveas(figHandle, filePath);
            catch
                warning('Could not save figure labeled %s', labels{k});
                continue;
            end
        end
        savedPaths{end+1,1} = filePath; %#ok<AGROW>
    end
end
