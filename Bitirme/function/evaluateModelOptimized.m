function evaluateModelOptimized(X_val, T_val, w_stage1, w_final, W_hidden, g, plot_title, config,num_hidden_units)
    % Model değerlendirme fonksiyonu
    
    num_hidden = length(W_hidden);
    
    % 1. Aşama 1 (Gizli Katmansız) Tahmin
    Y_stage1 = X_val * w_stage1;
    fit_stage1 = (1 - (sum((T_val - Y_stage1).^2) / sum((T_val - mean(T_val)).^2))) * 100;
    
    % 2. Aşama 2 (Tam Model) Tahmin
    X_curr = X_val;
    X_cand = X_val;
    for k = 1:num_hidden
        V_h = g(X_cand * W_hidden{k});
        X_curr = [X_curr, V_h];
        X_cand = [X_cand, V_h];
    end
    
    Y_final = X_curr * w_final;
    fit_final = (1 - (sum((T_val - Y_final).^2) / sum((T_val - mean(T_val)).^2))) * 100;
    
    fprintf('%s -> Başlangıç Fit: %.2f%% | Final Fit: %.2f%%\n', ...
        plot_title, fit_stage1, fit_final);
    
    if config.plotting.enabled
        figure('Name', plot_title, 'Color', 'w');
        plot(T_val, 'k', 'LineWidth', 1.5); hold on;
        plot(Y_stage1, 'r--', 'DisplayName', sprintf('Gizli Katman Yok - Fit: %.2f%%', fit_stage1));
        plot(Y_final, 'b-', 'DisplayName', sprintf('%d Gizli Katman - Fit: %.2f%%', num_hidden_units, fit_final));
        legend('show', 'Location', 'best');
        title(sprintf('%s (Fit: %.2f%%)', plot_title, fit_final));
        xlabel('Zaman Örneği');
        ylabel('Çıkış (Normalize)');
        grid on;
    end
end
