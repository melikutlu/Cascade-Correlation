function metric = candidateCorrelationMetric(w_h, X0, U, T, W_hidden, w_o, g, config, candOrder)
    % compute current model N-step output without candidate
    Y_model = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config);
    R = T - Y_model; % residual (M x N)

    % compute candidate activation v (M x N) using the same recursive
    % y-feedback as forwardModelTrajectory so the regressors are consistent.
    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)';
    nu = numel(ulags); 
    ny = numel(ylags);
    M = size(X0,1);
    N = size(U,2);
    v = dlarray(zeros(M,N));

    % Full y-history buffer: yhist(:,L) = y(t0-L), works for any lag combination
    maxLagY = max(ylags);
    yhist = dlarray(zeros(M, maxLagY));
    for j = 1:ny
        yhist(:, ylags(j)) = X0(:, nu+j);
    end

    % track previous pre-activation per hidden and candidate for diff modes
    orders_hidden = getHiddenOrders(config, numel(W_hidden));
    if nargin < 9 || isempty(candOrder)
        candOrder = 1;
    end
    z_state_hidden = cell(numel(W_hidden),1);
    for h=1:numel(W_hidden)
        z_state_hidden{h} = struct('prev1', dlarray(zeros(M,1)), 'prev2', dlarray(zeros(M,1)));
    end
    z_state_cand = struct('prev1', dlarray(zeros(M,1)), 'prev2', dlarray(zeros(M,1)));

    for t=1:N
        % u part
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
        % y part: read directly from full history buffer
        yvals = zeros(M, ny);
        for j=1:ny
            yvals(:,j) = yhist(:, ylags(j));
        end
        x_t = dlarray([uvals, yvals]);
        
        for h=1:numel(W_hidden)
            z_h = x_t * W_hidden{h};
            [a_h, z_state_hidden{h}] = applyHiddenActivation(z_h, z_state_hidden{h}, g, config, orders_hidden(h));
            x_t = [x_t, a_h];
        end
        x_t = dlarray(x_t);
        z_c = x_t * w_h;
        [v(:,t), z_state_cand] = applyHiddenActivation(z_c, z_state_cand, g, config, candOrder);

        % advance history with the current model prediction (same as forwardModelTrajectory)
        y_t = Y_model(:,t);
        yhist = [y_t, yhist(:, 1:maxLagY-1)];
    end

    % flatten and center
    r_vec = reshape(R, [], 1);
    v_vec = reshape(v, [], 1);
    r_mean = mean(r_vec);
    v_mean = mean(v_vec);
    r_c = r_vec - r_mean;
    v_c = (v_vec - v_mean) + 0.1 * sign(v_vec - v_mean);
    %Fahlman makalede aday ünite aktivasyonunun (v) çok hızlı doyuma ulaşıp (-1 veya +1) gradyanın sıfırlanmasından bahseder. Bunu engellemek için v_c hesaplanırken ufak bir "offset" (0.1 gibi) eklenmesini önerir.

    cov_vr = sum(v_c .* r_c);
    denom = (sum(v_c.^2) + eps) .* (sum(r_c.^2) + eps);
    corr2 = (cov_vr.^2) ./ denom; % correlation squared (scalar)

    metric = abs(cov_vr);
    

end
