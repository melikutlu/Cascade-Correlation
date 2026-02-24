
function [U_train, Y_train, U_val, Y_val] = loadMATData(config)
    % .mat dosyasından veri yükleme
    
    if isempty(config.data.filepath)
        error('MAT dosya yolu belirtilmeli: config.data.filepath');
    end
    
    data = load(config.data.filepath);
    
    % Değişken isimlerini belirle
    if isfield(config.data, 'input_var')
        U_all = data.(config.data.input_var);
    else
        % Varsayılan değişken isimleri
        if isfield(data, 'U')
            U_all = data.U;
        elseif isfield(data, 'u')
            U_all = data.u;
        elseif isfield(data, 'input')
            U_all = data.input;
        else
            error('Giriş verisi değişkeni bulunamadı');
        end
    end
    
    if isfield(config.data, 'output_var')
        Y_all = data.(config.data.output_var);
    else
        % Varsayılan değişken isimleri
        if isfield(data, 'Y')
            Y_all = data.Y;
        elseif isfield(data, 'y')
            Y_all = data.y;
        elseif isfield(data, 'output')
            Y_all = data.output;
        else
            error('Çıkış verisi değişkeni bulunamadı');
        end
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