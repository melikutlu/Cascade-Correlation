% Copy_February_12.m
% CCNN variant where MSE-style losses use MATLAB's built-in l2loss.
% Candidate training still maximizes N-step residual correlation^2.
clear; clc; close all; rng(0);
% ------------------
% CONFIG
% ------------------
config = struct();
config.data.source = 'twotankdata';
config.data.twotank.warmup_samples = 20;
config.data.twotank.sampling_time = 0.2; % s
config.data.twotank.filter_cutoff = 0.066902; % Hz (optional)
config.data.train_ratio = 0.5;
config.data.val_ratio = 0.5;
config.norm_method = 'ZScore';
config.prediction.n_steps = 480;
% regressors (user can change)
config.regressors.u = [0 1];
config.regressors.y = [1 2];
config.regressors.include_bias = false;
% model / training
config.model.activation = 'tanh';
config.model.max_hidden_units = 10;
config.model.target_mse = 5e-5;
config.model.min_mse_improvement = 1e-6;
config.model.max_epochs_output = 100;
config.model.eta_output = 0.02;
config.model.max_epochs_candidate = 150;
config.model.eta_candidate = 0.02;
% ------------------
% DATA
% ------------------
[Utr_raw, Ytr_raw, Uva_raw, Yva_raw] = loadDataByConfig_min(config);
[Utr, Ytr, Uva, Yva, norm_stats] = normalizeData_min(config.norm_method, Utr_raw, Ytr_raw, Uva_raw, Yva_raw);
Npred = config.prediction.n_steps;
[X0_tr, Utr_seq, Ttr_seq] = createTrajectoryDataset(Utr, Ytr, config, Npred);
[X0_va, Uva_seq, Tva_seq] = createTrajectoryDataset(Uva, Yva, config, Npred);
g = @(x) tanh(x);
W_hidden = {};
d0 = size(X0_tr,2);
w_o = randn(d0,1)*0.01;
% Stage 1: output-only (use l2loss in loss)
[w_o, current_mse] = trainOutputLayer_Trajectory(X0_tr, Utr_seq, Ttr_seq, w_o, W_hidden, g, config);
fprintf('Stage-1 Train MSE (l2loss): %.6g\n', current_mse);
mse_hist = current_mse;
% Greedy growth loop
while current_mse > config.model.target_mse && numel(W_hidden) < config.model.max_hidden_units
    h = numel(W_hidden) + 1;
    fprintf('\nTraining candidate #%d (maximize N-step residual correlation)\n', h);
    [w_h, cand_metric] = trainCandidateUnit_Corr(X0_tr, Utr_seq, Ttr_seq, W_hidden, w_o, g, config);
    fprintf('Candidate #%d | corr^2: %.6g\n', h, cand_metric);
    % tentatively add
    W_hidden{end+1} = w_h;
    w_o = [w_o; randn*0.01];
    prev_mse = current_mse;
    [w_o, current_mse] = trainOutputLayer_Trajectory(X0_tr, Utr_seq, Ttr_seq, w_o, W_hidden, g, config);
    improvement = prev_mse - current_mse;
    if improvement < config.model.min_mse_improvement
        % undo
        W_hidden(end) = [];
        w_o = w_o(1:end-1);
        fprintf('Undo candidate #%d: improvement %.3g < threshold %.3g. Stop.\n', h, improvement, config.model.min_mse_improvement);
        break;
    end
    mse_hist(end+1) = current_mse;
    fprintf('Hidden=%d | Train MSE (l2loss)=%.6g | improvement=%.3g\n', numel(W_hidden), current_mse, improvement);
end
% Full-series recursive predictions and denormalize
Yhat_tr = recursivePredictFullSeries(Utr, Ytr, W_hidden, w_o, g, config);
Yhat_va = recursivePredictFullSeries(Uva, Yva, W_hidden, w_o, g, config);
Yhat_tr = Yhat_tr(2:end) * norm_stats.y_std + norm_stats.y_mu;
Yhat_va = Yhat_va(2:end) * norm_stats.y_std + norm_stats.y_mu;
fit_tr = fitPercent(Ytr_raw(2:end), Yhat_tr);
fit_va = fitPercent(Yva_raw(2:end), Yhat_va);
fprintf('\nTrain Fit: %.2f%% | Val Fit: %.2f%%\n', fit_tr, fit_va);
figure('Name','TRAIN - Full Recursive','Color','w');
plot(Ytr_raw(2:end),'k','LineWidth',1.4); hold on; plot(Yhat_tr,'b--','LineWidth',1.2); grid on;
title(sprintf('TRAIN | Hidden=%d | Fit=%.2f%%', numel(W_hidden), fit_tr)); legend('True','CCNN');
figure('Name','VAL - Full Recursive','Color','w');
plot(Yva_raw(2:end),'k','LineWidth',1.4); hold on; plot(Yhat_va,'r--','LineWidth',1.2); grid on;
title(sprintf('VAL | Hidden=%d | Fit=%.2f%%', numel(W_hidden), fit_va)); legend('True','CCNN');
% ------------------
% LOCAL FUNCTIONS
% ------------------
function [w_h, best_metric] = trainCandidateUnit_Corr(X0,U,T,W_hidden,w_o,g,config)
    d = size(X0,2) + numel(W_hidden);
    w_h = dlarray(randn(d,1)*0.01);
    X0_d = dlarray(X0); U_d = dlarray(U); T_d = dlarray(T); w_o_d = dlarray(w_o);
    avgG=[]; avgGSq=[]; it=0; best_metric=-Inf; best_w = extractdata(w_h);
    for ep=1:config.model.max_epochs_candidate
        it = it + 1;
        [loss, metric, grad] = dlfeval(@loss_candidate_corr, w_h, X0_d, U_d, T_d, W_hidden, w_o_d, g, config);
        [w_h, avgG, avgGSq] = adamupdate(w_h, grad, avgG, avgGSq, it, config.model.eta_candidate);
        metric_val = gather(extractdata(metric));
        if metric_val > best_metric
            best_metric = metric_val;
            best_w = extractdata(w_h);
        end
    end
    w_h = best_w;
end
function [L, metric, grad] = loss_candidate_corr(w_h, X0, U, T, W_hidden, w_o, g, config)
    Y_model = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config);
    R = T - Y_model;
    M = size(X0,1); N = size(U,2);
    v = dlarray(zeros(M,N));
    for t=1:N
        x_t = buildRegressorRow(X0, U, t, W_hidden, g, config);
        v(:,t) = g(x_t * w_h);
    end
    r_vec = reshape(R, [], 1);
    v_vec = reshape(v, [], 1);
    r_c = r_vec - mean(r_vec);
    v_c = v_vec - mean(v_vec);
    cov_vr = sum(v_c .* r_c);
    denom = (sum(v_c.^2) + eps) .* (sum(r_c.^2) + eps);
    corr2 = (cov_vr.^2) ./ denom;
    metric = corr2;
    L = -corr2;
    grad = dlgradient(L, w_h);
end
function x = buildRegressorRow(X0, U, t, W_hidden, g, config)
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)';
    nu = numel(ulags); ny = numel(ylags); M = size(X0,1);
    uvals = zeros(M, nu);
    for j=1:nu
        L = ulags(j);
        if L==0
            uvals(:,j) = U(:,t);
        else
            idx = t - L;
            if idx >= 1
                uvals(:,j) = U(:, idx);
            else
                uvals(:,j) = X0(:, j);
            end
        end
    end
    yvals = zeros(M, ny);
    for j=1:ny
        L = ylags(j); idx = t - L;
        if idx >= 1
            yvals(:,j) = 0; % for initial times use X0
            yvals(:,j) = X0(:, nu + j);
        else
            yvals(:,j) = X0(:, nu + j);
        end
    end
    x = [uvals, yvals];
    for h=1:numel(W_hidden)
        x = [x, g(x * W_hidden{h})];
    end
    x = dlarray(x);
end
function Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config)
    M = size(X0,1); N = size(U,2);
    Y = dlarray(zeros(M,N));
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)'; nu = numel(ulags); ny = numel(ylags);
    yprev = X0(:, nu+1:nu+ny);
    for t=1:N
        uvals = zeros(M, nu);
        for j=1:nu
            L=ulags(j);
            if L==0
                uvals(:,j) = U(:,t);
            else
                idx = t - L;
                if idx>=1; uvals(:,j) = U(:, idx); else uvals(:,j) = X0(:, j); end
            end
        end
        yvals = zeros(M, ny);
        for j=1:ny; yvals(:,j) = yprev(:,j); end
        x = [uvals, yvals];
        for h=1:numel(W_hidden)
            x = [x, g(x * W_hidden{h})];
        end
        y = x * w_o;
        Y(:,t) = y;
        if ny>1; yprev = [y, yprev(:,1:ny-1)]; else yprev = y; end
    end
end
function [w_o,mse] = trainOutputLayer_Trajectory(X0,U,T,w_o,W_hidden,g,config)
    w_o = dlarray(w_o);
    X0 = dlarray(X0); U = dlarray(U); T = dlarray(T);
    avgG=[]; avgGSq=[]; it=0;
    for ep=1:config.model.max_epochs_output
        it=it+1;
        [L,grad] = dlfeval(@loss_output_traj, w_o, X0, U, T, W_hidden, g, config);
        [w_o, avgG, avgGSq] = adamupdate(w_o, grad, avgG, avgGSq, it, config.model.eta_output);
    end
    w_o = extractdata(w_o);
    Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config);
    % reshape to 1 x (M*N) and specify DataFormat for l2loss so dlarray is valid
    Yvec = reshape(Y,1,[]);
    Tvec = reshape(T,1,[]);
    mse = gather(extractdata(l2loss(Yvec, Tvec, 'DataFormat', 'CB')));
end
function [L,grad] = loss_output_traj(w, X0, U, T, W_hidden, g, config)
    Y = forwardModelTrajectory(X0, U, W_hidden, g, w, config);
    Yvec = reshape(Y,1,[]);
    Tvec = reshape(T,1,[]);
    L = l2loss(Yvec, Tvec, 'DataFormat', 'CB');
    grad = dlgradient(L, w);
end
function [Utr, Ytr, Uva, Yva] = loadDataByConfig_min(config)
    switch lower(config.data.source)
        case 'twotankdata'
            load twotankdata.mat; u = u(:); y = y(:);
            w = config.data.twotank.warmup_samples; u = u(w+1:end); y = y(w+1:end);
            if isfield(config.data.twotank,'filter_cutoff') && config.data.twotank.filter_cutoff>0
                fc = config.data.twotank.filter_cutoff; Ts = config.data.twotank.sampling_time;
                a = 2*pi*fc*Ts / (1 + 2*pi*fc*Ts);
                uf=zeros(size(u)); yf=zeros(size(y)); uf(1)=u(1); yf(1)=y(1);
                for k=2:length(u); uf(k)=a*u(k)+(1-a)*uf(k-1); yf(k)=a*y(k)+(1-a)*yf(k-1); end
                u = uf; y = yf;
            end
        otherwise
            error('Unknown data source');
    end
    N = length(u); Ntr = floor(config.data.train_ratio * N);
    Utr = u(1:Ntr); Ytr = y(1:Ntr); Uva = u(Ntr+1:end); Yva = y(Ntr+1:end);
end
function [Utr,Ytr,Uva,Yva,stats] = normalizeData_min(method,Utr,Ytr,Uva,Yva)
    switch lower(method)
        case 'zscore'
            stats.u_mu = mean(Utr); stats.u_std = std(Utr)+eps;
            stats.y_mu = mean(Ytr); stats.y_std = std(Ytr)+eps;
            Utr = (Utr - stats.u_mu)/stats.u_std; Uva = (Uva - stats.u_mu)/stats.u_std;
            Ytr = (Ytr - stats.y_mu)/stats.y_std; Yva = (Yva - stats.y_mu)/stats.y_std;
        otherwise
            error('Unknown normalization');
    end
end
function [X0, Useq, Tseq] = createTrajectoryDataset(U, Y, config, N)
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)';
    maxLag = 0; if ~isempty(ulags(ulags>0)); maxLag = max(maxLag, max(ulags(ulags>0))); end
    if ~isempty(ylags); maxLag = max(maxLag, max(ylags)); end
    Ns = length(Y) - N - maxLag; if Ns<1; error('Not enough data'); end
    nu = numel(ulags); ny = numel(ylags);
    X0 = zeros(Ns, nu+ny); Useq = zeros(Ns, N); Tseq = zeros(Ns, N);
    for idx=1:Ns
        i = idx + maxLag; row = zeros(1,nu+ny);
        for j=1:nu; L=ulags(j); if L==0; row(j)=U(i); else row(j)=U(i+1-L); end; end
        for j=1:ny; L=ylags(j); row(nu+j)=Y(i+1-L); end
        X0(idx,:) = row; Useq(idx,:) = U(i+1:i+N)'; Tseq(idx,:) = Y(i+1:i+N)';
    end
end
function Yhat = recursivePredictFullSeries(U, Y, W_hidden, w_o, g, config)
    N = length(Y); Yhat=zeros(N,1); if N>=1; Yhat(1)=Y(1); end
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)'; nu=numel(ulags); ny=numel(ylags);
    for k=2:N
        uvals=zeros(nu,1); for j=1:nu; L=ulags(j); if L==0; uvals(j)=U(k); else idx=k-L; if idx>=1; uvals(j)=U(idx); else uvals(j)=0; end; end; end
        yvals=zeros(ny,1); for j=1:ny; L=ylags(j); idx=k-L; if idx>=1; yvals(j)=Yhat(idx); else yvals(j)=0; end; end
        x = [uvals(:)', yvals(:)']; for h=1:numel(W_hidden); x=[x, g(x*W_hidden{h})]; end
        Yhat(k)= x * w_o;
    end
end
function fit = fitPercent(y, yhat)
    fit = 100 * (1 - norm(y - yhat) / norm(y - mean(y)));
end