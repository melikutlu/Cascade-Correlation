function [u_tr_n, y_tr_n, u_val_n, y_val_n, stats] = normalizeData(method, u_tr, y_tr, u_val, y_val)
    % Bu fonksiyon istatistikleri SADECE u_tr ve y_tr'den çıkarır.
    % Hem training hem validation verisine aynı istatistiği uygular.
    
    stats.method = method;
    stats.u = [];
    stats.y = [];
    
    switch method
        case 'ZScore'
            % İstatistikleri Hesapla (Eğitimden)
            stats.u.mean = mean(u_tr); stats.u.std = std(u_tr);
            stats.y.mean = mean(y_tr); stats.y.std = std(y_tr);
            
            % Uygula
            u_tr_n = (u_tr - stats.u.mean) ./ stats.u.std;
            y_tr_n = (y_tr - stats.y.mean) ./ stats.y.std;
            u_val_n = (u_val - stats.u.mean) ./ stats.u.std;
            y_val_n = (y_val - stats.y.mean) ./ stats.y.std;
            
            fprintf('   -> İstatistikler: Mean_U=%.2f, Std_U=%.2f\n', stats.u.mean, stats.u.std);
            
            
        otherwise
            error('Bilinmeyen normalizasyon yöntemi: %s', method);
    end
end
