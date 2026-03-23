function Y = forwardModelTrajectory(X0, U, W_hidden, g, w_o, config)
    % forward pass computing N-step outputs with current W_hidden and w_o (no candidate)
    M = size(X0,1); N = size(U,2);
    Y = dlarray(zeros(M,N));

    % we'll keep a history of predicted y for each sample; initialize from X0
    % X0 columns are [u regressors, y regressors]
    ulags = config.regressors.u(:)'; ylags = config.regressors.y(:)';
    nu = numel(ulags); ny = numel(ylags);

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

        % compute diffs for u and y to create expanded hidden inputs
        u_curr = uvals;
        % previous u: shift by one sample where available (use X0 when not)
        u_prev = zeros(M, nu);
        for j=1:nu
            L = ulags(j);
            if L==0
                idx = t - 1;
            else
                idx = t - L - 1;
            end
            if idx >= 1
                u_prev(:,j) = U(:, idx);
            else
                u_prev(:,j) = X0(:, j);
            end
        end
        u_diff = u_curr - u_prev;

        y_curr = yvals;
        y_prev = zeros(M, ny);
        for j=1:ny
            L = ylags(j);
            if maxLagY>0 && (L+1) <= maxLagY
                y_prev(:,j) = yhist(:, L+1);
            else
                y_prev(:,j) = yhist(:, L);
            end
        end
        y_diff = y_curr - y_prev;

        % base for output remains [u_curr, y_curr]
        x_out = [u_curr, y_curr];

        % compute hidden activations using expanded inputs: [u_curr,y_curr,u_diff,y_diff, prev_hidden_acts]
        hidden_acts = [];
        for h=1:numel(W_hidden)
            x_hidden = [u_curr, y_curr, u_diff, y_diff, hidden_acts];
            x_hidden = dlarray(x_hidden);
            act = g(x_hidden * W_hidden{h});
            hidden_acts = [hidden_acts, act];
        end

        x = [x_out, hidden_acts];
        y = x * w_o;
        Y(:,t) = y;

        % update history: shift right, insert latest prediction at lag-1
        if maxLagY>0
            yhist = [y, yhist(:, 1:maxLagY-1)];
        end
    end
end
