function Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config)
    % forward pass computing N-step outputs with current W_hidden and w_o (no candidate)
    M = size(X0,1); N = size(U,2);
    Y = dlarray(zeros(M,N));

    % we'll keep a history of predicted y for each sample; initialize from X0
    % X0 columns are [u regressors, y regressors]
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)';
    nu = numel(ulags); ny = numel(ylags);

    % Full y-history buffer: yhist(:,L) = y(t0-L), works for any lag combination
    maxLagY = max(ylags);
    yhist = zeros(M, maxLagY);
    for j = 1:ny
        yhist(:, ylags(j)) = X0(:, nu+j);
    end

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

        x = [uvals, yvals];
        for h=1:numel(W_hidden)
            x = [x, g(x * W_hidden{h})];
        end
        y = x * w_o;
        Y(:,t) = y;

        % update history: shift right, insert latest prediction at lag-1
        yhist = [y, yhist(:, 1:maxLagY-1)];
    end
end
