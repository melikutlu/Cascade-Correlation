function data_real = denormalizeData(data_norm, method, stats_struct)
    % Veriyi eski haline çevirir (Sadece tek bir kanal için, örn: Output)

    stats.method = method;
    stats.u = [];
    stats.y = [];
    
    switch method
        case 'ZScore'
            % x = z * std + mean
            data_real = data_norm * stats_struct.std + stats_struct.mean;
            
    end
end
