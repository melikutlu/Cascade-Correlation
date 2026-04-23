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
        % Create readable filename from label
        displayLabel = makeReadableLabel(labels{k});
        fileName = sprintf('%s.png', displayLabel);
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

function readableLabel = makeReadableLabel(label)
    % Convert label like 'loss_history' or 'candidateCorr' to 'Loss History'
    label = strtrim(string(label));
    
    % Replace underscores with spaces
    label = replace(label, '_', ' ');
    
    % Insert space before capital letters (for camelCase)
    label = regexprep(label, '([a-z])([A-Z])', '$1 $2');
    
    % Capitalize first letter of each word
    label = regexprep(label, '(^|\s)(\w)', '${upper($2)}');
    
    readableLabel = char(label);
end
