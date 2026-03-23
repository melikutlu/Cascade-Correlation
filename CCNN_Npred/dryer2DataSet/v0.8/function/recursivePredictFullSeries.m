function Yhat = recursivePredictFullSeries(U, Y, W_hidden, w_o, g, config)
    N = length(Y); 
    Yhat = zeros(N,1); 
    if N>=1; Yhat(1)=Y(1); end
    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)'; 
    nu=numel(ulags); ny=numel(ylags);

    % work with gathered weights for speed during inference
    W_local = cellfun(@gather, W_hidden, 'UniformOutput', false);
    w_o_local = gather(w_o);

    % Calculate z_prev for all hidden units using t=0 regressors (initial values)
    % z_prev_h(t=0) = z_h(t=0) from initial regressors
    z_prev = cell(numel(W_local),1);
    
    % Build t=0 regressors: u and y values at initial time
    uvals_t0 = zeros(nu,1); 
    for j=1:nu
        L=ulags(j); 
        if L==0
            uvals_t0(j)=U(1);  % u(t=1)
        else 
            idx=1-L; 
            if idx>=1; uvals_t0(j)=U(idx); else uvals_t0(j)=0; end
        end
    end
    yvals_t0 = zeros(ny,1); 
    for j=1:ny
        L=ylags(j); 
        idx=1-L; 
        if idx>=1; yvals_t0(j)=Y(idx); else yvals_t0(j)=Y(1); end
    end
    x_t0 = [uvals_t0(:)', yvals_t0(:)'];
    
    % Initialize z_prev from t=0 regressors with cascading activations
    % a_h(1) = z_h(1) - z_h(0), so we need z_h(0) from t=0 data
    for h=1:numel(W_local)
        z = x_t0 * W_local{h};  % ← z_h(t=0)
        a = applyHiddenActivation(z, 0, g, config);  % ← activation needs prev (0 for initialization)
        z_prev{h} = z;  % ← z_prev_h = z_h(t=0)
        x_t0 = [x_t0, a];  % ← Extend for next hidden unit with cascade activation
    end

    % PREDICTION PHASE: k=1 onwards (with proper z_prev values from t=0)
    for k=1:N
        uvals=zeros(nu,1); 
        for j=1:nu
            L=ulags(j); 
            if L==0
                uvals(j)=U(k); 
            else 
                idx=k-L; 
                if idx>=1; uvals(j)=U(idx); else uvals(j)=0; end
            end
        end
        yvals=zeros(ny,1); 
        for j=1:ny
            L=ylags(j); 
            idx=k-L; 
            if idx>=1; yvals(j)=Yhat(idx); else yvals(j)=0; end
        end
        x = [uvals(:)', yvals(:)']; 
        for h=1:numel(W_local)
            z = x*W_local{h}; 
            a = applyHiddenActivation(z, z_prev{h}, g, config); 
            z_prev{h} = z; 
            x=[x, a];
        end
        Yhat(k)= x * w_o_local;
    end
end
