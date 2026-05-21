function savedPaths = saveFitFigures(logFilePath, figMap, activation, regressorCount)
    savedPaths = {};
    if nargin < 2 || isempty(figMap)
        return;
    end
    
    % activation parametresi opsiyonel, boşsa varsayılan olarak boş string
    if nargin < 3
        activation = '';
    end
    
    % regressorCount parametresi opsiyonel
    if nargin < 4
        regressorCount = '';
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

    % Regresör bilgisini hazırla (dosya isimlerine eklenecek)
    regressorSuffix = '';
    if ~isempty(regressorCount)
        regressorSuffix = sprintf('_reg%d', regressorCount);
    end
    
    % Activation bilgisini hazırla (dosya isimlerine eklenecek)
    activationSuffix = '';
    if ~isempty(activation)
        activation = char(activation);
        activationClean = regexprep(activation, '[^A-Za-z0-9_-]', '_');
        activationClean = lower(activationClean);
        activationSuffix = sprintf('_%s', activationClean);
    end

    labels = fieldnames(figMap);
    for k = 1:numel(labels)
        figHandle = figMap.(labels{k});
        if isempty(figHandle) || ~ishandle(figHandle)
            continue;
        end
        % Create readable filename from label
        displayLabel = makeReadableLabel(labels{k});
        % Regresör ve activation bilgisini dosya ismine ekle (overwrite'yi önle)
        fileName = sprintf('%s%s%s.png', displayLabel, regressorSuffix, activationSuffix);
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
