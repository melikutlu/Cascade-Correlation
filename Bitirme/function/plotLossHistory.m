function plotLossHistory(mse_history, target_mse)
% plotLossHistory: Modelin MSE'sinin her gizli birim eklendikçe
% nasıl geliştiğini gösteren bir grafik çizer.
%
% GİRDİLER:
%   mse_history: Her adımın sonunda kaydedilen MSE değerlerini içeren vektör
%   target_mse: Grafikte bir hedef çizgisi olarak gösterilecek MSE hedefi

fprintf('\n--- AŞAMA 6: Kayıp (Loss) Gelişim Grafiği Çiziliyor ---\n');

if isempty(mse_history)
    warning('MSE geçmişi (mse_history) boş, grafik çizilemiyor.');
    return;
end

% X ekseni etiketlerini oluştur (0'dan başla)
% mse_history[1] -> 0 gizli birim
% mse_history[2] -> 1 gizli birim
% ...
num_steps = length(mse_history);
x_axis_labels = 0:(num_steps - 1);

figure;
hold on;

% Ana MSE gelişim çizigisi
plot(x_axis_labels, mse_history, 'bo-', 'LineWidth', 2, 'MarkerFaceColor', 'b', ...
     'DisplayName', 'Model MSE');

% Hedef MSE çizgisi (yatay)
plot(x_axis_labels, ones(1, num_steps) * target_mse, 'r--', 'LineWidth', 1.5, ...
     'DisplayName', sprintf('Hedef MSE (%.4f)', target_mse));

hold off;

title('Model Hatasının (MSE) Gizli Birimlere Göre Gelişimi');
xlabel('Eklenen Gizli Birim Sayısı');
ylabel('Ortalama Kare Hata (MSE)');
legend('show', 'Location', 'northeast');
grid on;
set(gca, 'YScale', 'log'); % MSE'yi logaritmik ölçekte görmek genellikle daha iyidir
set(gca, 'XTick', x_axis_labels); % X ekseninde sadece tam sayıları (0, 1, 2...) göster

fprintf('Grafik tamamlandı. Başlangıç MSE: %f -> Final MSE: %f\n', ...
        mse_history(1), mse_history(end));

end