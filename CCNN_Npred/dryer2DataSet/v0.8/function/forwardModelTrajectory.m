function Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config)
    % forward pass computing N-step outputs with current W_hidden and w_o (no candidate)
    % z_prev initialized from t=0 regressors (X0), NOT with zeros
    M = size(X0,1); N = size(U,2);
    Y = dlarray(zeros(M,N));

    % we'll keep a history of predicted y for each sample; initialize from X0
    % X0 columns are [u regressors, y regressors]
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)';
    nu = numel(ulags); ny = numel(ylags);

    % Full y-history buffer: yhist(:,L) = y(t0-L), works for any lag combination
    maxLagY = max(ylags);
    yhist = dlarray(zeros(M, maxLagY));
    for j = 1:ny
        yhist(:, ylags(j)) = X0(:, nu+j);
    end

    % Initialize z_prev for all hidden units using t=0 regressors (X0)
    % z_prev_h(t=0) = z_h(t=0) from X0 data
    % Later: a_h(1) = z_h(1) - z_h(t=0) (first difference)
    z_prev = cell(numel(W_hidden),1);
    
    x_t0 = dlarray(X0);  % ← t=0 regressors
    
    % Initialize z_prev from t=0 data with cascading activations
    for h=1:numel(W_hidden)
        z = x_t0 * W_hidden{h};  % ← z_h(t=0)
        a = applyHiddenActivation(z, dlarray(zeros(M,1)), g, config);  % ← activation at t=0
        z_prev{h} = z;  % ← z_prev_h(t=0) = z_h(t=0)
        x_t0 = [x_t0, a];  % ← Extend for next hidden unit with cascade activation
    end

    % PREDICTION PHASE: t=1 onwards (with proper z_prev values from t=0)
    for t=1:N
        % build current regressor x for each sample
        uvals = zeros(M, nu);
        for j=1:nu
            L = ulags(j);
            if L==0
                uvals(:,j) = U(:, t);
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

        x = dlarray([uvals, yvals]);
        for h=1:numel(W_hidden)
            z = x * W_hidden{h};
            a = applyHiddenActivation(z, z_prev{h}, g, config);
            z_prev{h} = z;
            x = [x, a];
        end
        y = x * w_o;
        Y(:,t) = y;

        % update history: shift right, insert latest prediction at lag-1
        yhist = [y, yhist(:, 1:maxLagY-1)];
    end
end
