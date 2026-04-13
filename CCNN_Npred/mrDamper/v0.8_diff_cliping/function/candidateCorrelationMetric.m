function metric = candidateCorrelationMetric(w_h, X0, U, T, W_hidden, w_o, g, config)
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

    % Turev operatoru aktivasyonda hesaplanir; ilk adimda z_prev = 0 alinir.
    nWarmupSteps = 1;

    % Full y-history buffer: yhist(:,L) = y(t0-L), works for any lag combination
    % Bu surumde nWarmupSteps=1 oldugu icin t0 gecmisinden basla.
    maxLagY = max(ylags);
    yhist = dlarray(zeros(M, maxLagY));
    for j = 1:ny
        yhist(:, ylags(j)) = X0(:, nWarmupSteps, nu+j);
    end

    % hidden ve candidate icin bir onceki pre-activation durumlari
    z_prev_hidden = cell(numel(W_hidden),1);
    for h=1:numel(W_hidden)
        z_prev_hidden{h} = dlarray(zeros(M,1));
    end
    z_prev_cand = dlarray(zeros(M,1));

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
                    uvals(:,j) = X0(:, nWarmupSteps, j);
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
            a_h = applyHiddenActivation(z_h, z_prev_hidden{h}, g, config);
            z_prev_hidden{h} = z_h;
            x_t = [x_t, a_h];
        end
        x_t = dlarray(x_t);
        z_c = x_t * w_h;
        v(:,t) = applyHiddenActivation(z_c, z_prev_cand, g, config);
        z_prev_cand = z_c;

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
    v_c = (v_vec - v_mean) + 000.1 * sign(v_vec - v_mean);
    %Fahlman makalede aday ünite aktivasyonunun (v) çok hızlı doyuma ulaşıp (-1 veya +1) gradyanın sıfırlanmasından bahseder. Bunu engellemek için v_c hesaplanırken ufak bir "offset" (0.1 gibi) eklenmesini önerir.

    cov_vr = sum(v_c .* r_c);
    denom = (sum(v_c.^2) + eps) .* (sum(r_c.^2) + eps);
    corr2 = (cov_vr.^2) ./ denom; % correlation squared (scalar)

    metric = abs(cov_vr);
    
end
