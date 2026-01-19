
function [U_train, Y_train, U_val, Y_val] = loadFromWorkspace(config)
    % Workspace'ten değişkenleri yükleme
    
    fprintf('Workspace''ten yükleme:\n');
    
    % Giriş değişkenleri
    if isfield(config.data, 'input_var')
        U_all = evalin('base', config.data.input_var);
    else
        error('Workspace için config.data.input_var belirtilmeli');
    end
    
    % Çıkış değişkenleri
    if isfield(config.data, 'output_var')
        Y_all = evalin('base', config.data.output_var);
    else
        error('Workspace için config.data.output_var belirtilmeli');
    end
    
    % Veriyi böl
    N = size(U_all, 1);
    train_end = floor(N * config.data.train_ratio);
    val_end = train_end + floor(N * config.data.val_ratio);
    
    U_train = U_all(1:train_end, :);
    Y_train = Y_all(1:train_end, :);
    U_val = U_all(train_end+1:val_end, :);
    Y_val = Y_all(train_end+1:val_end, :);
end