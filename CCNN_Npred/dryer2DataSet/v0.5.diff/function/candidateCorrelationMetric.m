function metric = candidateCorrelationMetric(w_h, X0, U, T, W_hidden, w_o, g, config)
    % Compute model N-step output without candidate
    Y_model = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config);
    R = T - Y_model; % residual (M x N)

    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)';
    nu = numel(ulags); 
    ny = numel(ylags);
    M = size(X0,1);
    N = size(U,2);

    % candidate feature layout per time step (must match forwardModelTrajectory):
    % [u_curr (nu), y_curr (ny), u_diff (nu), y_diff (ny), previous hidden acts...]

    v = dlarray(zeros(M,N));

    % Full y-history buffer: yhist(:,L) = y(t0-L), works for any lag combination
    maxLagY = max([ylags, 0]);
    if isempty(maxLagY) || isnan(maxLagY)
        maxLagY = 0;
    end
    if maxLagY>0
        yhist = zeros(M, maxLagY);
        for j = 1:ny
            yhist(:, ylags(j)) = X0(:, nu+j);
        end
    else
        yhist = zeros(M,1);
    end

    for t=1:N
        % u current and previous values for each regressor
        u_curr = zeros(M, nu);
        u_prev = zeros(M, nu);
        for j=1:nu
            L = ulags(j);
            if L==0
                idx = t;
            else
                idx = t - L;
            end
            if idx >= 1
                u_curr(:,j) = U(:, idx);
            else
                u_curr(:,j) = X0(:, j);
            end
            idx_prev = idx - 1;
            if idx_prev >= 1
                u_prev(:,j) = U(:, idx_prev);
            else
                u_prev(:,j) = X0(:, j);
            end
        end

        % y current and previous values for each regressor (from history buffer)
        y_curr = zeros(M, ny);
        y_prev = zeros(M, ny);
        for j=1:ny
            L = ylags(j);
            if maxLagY>0 && L>=1
                y_curr(:,j) = yhist(:, L);
                if L+1 <= maxLagY
                    y_prev(:,j) = yhist(:, L+1);
                else
                    y_prev(:,j) = yhist(:, L);
                end
            else
                y_curr(:,j) = 0;
                y_prev(:,j) = 0;
            end
        end

        u_diff = u_curr - u_prev; % (M x nu)
        y_diff = y_curr - y_prev; % (M x ny)

        % base features: [u_curr, y_curr, u_diff, y_diff]
        base_feat = [u_curr, y_curr, u_diff, y_diff]; % (M x (2*(nu+ny)))

        % compute hidden activations sequentially using expanded inputs
        hidden_acts = [];
        for h=1:numel(W_hidden)
            x_hidden = [base_feat, hidden_acts];
            x_hidden = dlarray(x_hidden);
            act = g(x_hidden * W_hidden{h});
            hidden_acts = [hidden_acts, act];
        end

        x_feat = [base_feat, hidden_acts];
        x_feat = dlarray(x_feat);
        v(:,t) = g(x_feat * w_h); % pass through g for candidate if desired

        % advance history with the current model prediction
        y_t = Y_model(:,t);
        if maxLagY>0
            yhist = [y_t, yhist(:, 1:maxLagY-1)];
        end
    end

    % flatten and center
    r_vec = reshape(R, [], 1);
    v_vec = reshape(v, [], 1);
    r_mean = mean(r_vec);
    v_mean = mean(v_vec);
    r_c = r_vec - r_mean;
    v_c = (v_vec - v_mean) + 0.1 * sign(v_vec - v_mean);

    cov_vr = sum(v_c .* r_c);
    denom = (sum(v_c.^2) + eps) .* (sum(r_c.^2) + eps);
    % correlation squared (not used directly here but kept for reference)
    corr2 = (cov_vr.^2) ./ denom; 

    metric = abs(cov_vr);
end
