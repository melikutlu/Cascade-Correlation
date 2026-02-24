
function [U_train, Y_train, U_val, Y_val] = loadCSVData(config)
    % CSV dosyasından veri yükleme
    
    if isempty(config.data.filepath)
        error('CSV dosya yolu belirtilmeli: config.data.filepath');
    end
    
    data = readtable(config.data.filepath);
    
    % Sütun belirleme
    if isfield(config.data, 'input_columns') && ~isempty(config.data.input_columns)
        input_cols = config.data.input_columns;
    else
        % Varsayılan: ilk num_inputs sütun
        input_cols = 1:config.model.num_inputs;
    end
    
    if isfield(config.data, 'output_columns') && ~isempty(config.data.output_columns)
        output_cols = config.data.output_columns;
    else
        % Varsayılan: sonraki num_outputs sütun
        start_idx = max(input_cols) + 1;
        output_cols = start_idx:(start_idx + config.model.num_outputs - 1);
    end
    
    U_all = table2array(data(:, input_cols));
    Y_all = table2array(data(:, output_cols));
    
    % Veriyi böl
    N = size(U_all, 1);
    train_end = floor(N * config.data.train_ratio);
    val_end = train_end + floor(N * config.data.val_ratio);
    
    U_train = U_all(1:train_end, :);
    Y_train = Y_all(1:train_end, :);
    U_val = U_all(train_end+1:val_end, :);
    Y_val = Y_all(train_end+1:val_end, :);
end
