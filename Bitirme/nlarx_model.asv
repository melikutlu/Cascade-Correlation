load twotankdata
z1f = iddata(y,u,0.2,'Name','Two-tank system');
z1 = z1f(1:1500);
z2 = z1f(1501:3000);

z1f = idfilt(z1,3,0.066902);
z2f = idfilt(z2,3,0.066902);
%Remove the effect of the filter initial conditions from the data by trimming the initial data points from each data set.
z1f = z1f(20:end);
z2f = z2f(20:end);
f = idNeuralNetwork("cascade-correlation", "tanh", false, false, ...
    MaxNumActLayers = 1, ...
    SizeSelection   = "on");
% 2) ADAM optimizer ayarları
f.EstimationOptions.Solver = "ADAM";

% Tüm ağı eğitirken epoch sayısı (varsayılan zaten 100 ama açıkça yazıyoruz)
f.EstimationOptions.SolverOptions.MaxEpochs = 100;


% 3) nlarx seçenekleri
opt = nlarxOptions;

% Cascade-correlation için default: CrossValidate = true
% ama sen kapatmak istemişsin:
opt.CrossValidate = false;

% Dış arama (Levenberg-Maraquardt vs.) iterasyon sayısı
% 0 dersen, ağın ağırlıkları sadece cascade-correlation + ADAM ile init edilir,
% nlarx'ın dış search'ü hiçbir ekstra update yapmaz.
opt.SearchOptions.MaxIterations = 20;

% Normalizasyon
opt.NormalizationOptions.NormalizationMethod = 'zscore';

orders = [1 1 0];
sys1 = nlarx(z1f, orders, f, opt);

% compare çıktısını al
[yhat, fit] = compare(z2f, sys1);   % yhat da bir iddata

% Zaman vektörü (örnek sayısı * Ts)
t = z2f.SamplingInstants;   % veya: t = (0:length(z2f.y)-1)' * z2f.Ts;

% Ölçülen ve model çıktıları
y_meas = z2f.y;
y_mod  = yhat.y;

figure;
plot(t, y_meas, 'LineWidth', 1.2); hold on;
plot(t, y_mod,  'LineWidth', 1.2);
grid on;
xlabel('Time (s)');
ylabel('Output level');
legend('Measured output','Model output','Location','best');
title(sprintf('Model fit = %.1f %%', fit));


