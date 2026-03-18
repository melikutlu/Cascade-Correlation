function Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config)
    % forward pass computing N-step outputs with current W_hidden and w_o (no candidate)
    % Now includes warm-up phase to properly initialize z_prev values for cascading differences
    M = size(X0,1); N = size(U,2);
    Y = dlarray(zeros(M,N));

    % we'll keep a history of predicted y for each sample; initialize from X0
    % X0 columns are [u regressors, y regressors]
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)';
    nu = numel(ulags); ny = numel(ylags);

    % Calculate maxLag and warm-up steps needed for cascading differences
    maxLagY = max(ylags);
    maxLag = 0;
    if ~isempty(ulags(ulags>0)); maxLag = max(maxLag, max(ulags(ulags>0))); end
    if ~isempty(ylags); maxLag = max(maxLag, max(ylags)); end
    maxHidden = numel(W_hidden);
    warmupSteps = min(maxHidden + maxLag, N - 1);  % number of warm-up iterations
    
    % Full y-history buffer: yhist(:,L) = y(t0-L), works for any lag combination
    yhist = dlarray(zeros(M, maxLagY));
    for j = 1:ny
        yhist(:, ylags(j)) = X0(:, nu+j);
    end

    % Initialize z_prev for all hidden units
    z_prev = cell(numel(W_hidden),1);
    for h=1:numel(W_hidden)
        z_prev{h} = dlarray(zeros(M,1));
    end

    % WARM-UP PHASE: t=1 to warmupSteps (compute but don't store predictions, build z_prev)
    Y_warmup = dlarray(zeros(M, warmupSteps));
    for t=1:warmupSteps
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
        Y_warmup(:,t) = y;  % store for history update
        
        % update history: shift right, insert latest prediction at lag-1
        yhist = [y, yhist(:, 1:maxLagY-1)];
    end

    % Copy warm-up outputs to Y
    Y(:, 1:warmupSteps) = Y_warmup;

    % PREDICTION PHASE: t=warmupSteps+1 onwards (with proper z_prev values)
    for t=warmupSteps+1:N
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
