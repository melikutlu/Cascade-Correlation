
function [U_train, Y_train, U_val, Y_val] = loadTwotankData(config)
    % Twotankdata için özel yükleyici
    
    load dryer2;
    z_full = iddata(y2, u2, config.data.twotank.sampling_time);
    
    % Veriyi böl
    N_total = length(z_full.y);
    train_end = floor(N_total * config.data.train_ratio);
    val_end = train_end + floor(N_total * config.data.val_ratio);
    
    % Eğitim verisi
    if config.data.train_ratio > 0
        z1 = z_full(1:train_end);
        z1f = detrend(z1);
        U_train = z1f.u;
        Y_train = z1f.y;
    else
        U_train = [];
        Y_train = [];
    end
    
    % Doğrulama verisi
    if config.data.val_ratio > 0
        z2 = z_full(train_end+1:val_end);
        z2f = detrend(z2);
        U_val = z2f.u;
        Y_val = z2f.y;
    else
        U_val = [];
        Y_val = [];
    end
end
