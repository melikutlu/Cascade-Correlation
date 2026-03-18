function Yhat = recursivePredictFullSeries(U, Y, W_hidden, w_o, g, config)
    N = length(Y); 
    Yhat = zeros(N,1); 
    if N>=1; Yhat(1)=Y(1); end
    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)'; 
    nu=numel(ulags); ny=numel(ylags);

    % Calculate maxLag and warm-up steps needed for cascading differences
    maxLag = 0;
    if ~isempty(ulags(ulags>0)); maxLag = max(maxLag, max(ulags(ulags>0))); end
    if ~isempty(ylags); maxLag = max(maxLag, max(ylags)); end
    maxHidden = numel(W_hidden);
    warmupSteps = min(maxHidden + maxLag, N - 2);  % number of warm-up iterations

    % work with gathered weights for speed during inference
    W_local = cellfun(@gather, W_hidden, 'UniformOutput', false);
    w_o_local = gather(w_o);

    % Initialize z_prev for all hidden units
    z_prev = cell(numel(W_local),1);
    for h=1:numel(W_local)
        z_prev{h} = 0;
    end

    % Pre-populate Yhat with ground truth during warm-up (t=1 to warmupSteps+1)
    for k=1:min(warmupSteps+1, N)
        Yhat(k) = Y(k);
    end

    % WARM-UP PHASE: build z_prev history without predictions (k=1 to warmupSteps)
    for k=1:warmupSteps
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
    end

    % PREDICTION PHASE: k=warmupSteps+2 onwards (with proper z_prev values)
    for k=warmupSteps+2:N
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
