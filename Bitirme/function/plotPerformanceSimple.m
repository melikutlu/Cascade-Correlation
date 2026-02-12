
function plotPerformanceSimple(T, Y_stage1, X_final_input, w_final, title_txt, config,num_hidden_units)
    % Performans grafiği
    
    Y_final = X_final_input * w_final;
    fit_final = (1 - (sum((T - Y_final).^2) / sum((T - mean(T)).^2))) * 100;
    fit_stage1 = (1 - (sum((T - Y_stage1).^2) / sum((T - mean(T)).^2))) * 100;
    
    % Gizli katman sayısı hesapla
    num_hidden_calc = size(X_final_input, 2) - size(Y_stage1, 2);
    
    if config.plotting.enabled
        figure('Name', title_txt, 'Color', 'w');
        plot(T, 'k', 'LineWidth', 1.5); hold on;
        plot(Y_stage1, 'r--', 'LineWidth', 1.2, ...
            'DisplayName', sprintf('Gizli Katman Yok - Fit: %.2f%%', fit_stage1));
        plot(Y_final, 'b-', 'LineWidth', 1.2, ...
            'DisplayName', sprintf('%d Gizli Katman - Fit: %.2f%%', num_hidden_units, fit_final));
        legend('show', 'Location', 'best');
        title(sprintf('%s (Fit: %.2f%%)', title_txt, fit_final));
        xlabel('Zaman Örneği');
        ylabel('Çıkış (Normalize)');
        grid on;
    end
end
