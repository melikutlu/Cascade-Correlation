function [U_train, Y_train, U_val, Y_val, U_test, Y_test] = loadDataByConfig(config)
    % Farklı veri kaynaklarını yükleyen ana fonksiyon
    
    U_test = [];
    Y_test = [];
    
    switch lower(config.data.source)
        case 'twotankdata'
            [U_train, Y_train, U_val, Y_val] = loadTwotankData(config);
            
        case 'csv'
            [U_train, Y_train, U_val, Y_val] = loadCSVData(config);
            
        case 'mat'
            [U_train, Y_train, U_val, Y_val] = loadMATData(config);
            
        case 'workspace'
            [U_train, Y_train, U_val, Y_val] = loadFromWorkspace(config);
            
        otherwise
            error('Desteklenmeyen veri kaynağı: %s', config.data.source);
    end
end
