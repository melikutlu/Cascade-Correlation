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

    % Calculate initial z_prev from k=1 regressors (first data point)
    z_prev = cell(numel(W_local),1);
    uvals_init = zeros(nu,1);
    for j=1:nu
        L = ulags(j);
        if L==0
            uvals_init(j) = U(1);
        else
            idx = 1 - L;
            if idx >= 1
                uvals_init(j) = U(idx);
            else
                uvals_init(j) = 0;
            end
        end
    end
    yvals_init = zeros(ny,1);
    for j=1:ny
        L = ylags(j);
        idx = 1 - L;
        if idx >= 1
            yvals_init(j) = Yhat(idx);
        else
            yvals_init(j) = 0;
        end
    end
    x_init = [uvals_init(:)', yvals_init(:)'];
    for h=1:numel(W_local)
        z_prev{h} = x_init * W_local{h};
    end

    for k=2:N
        uvals=zeros(nu,1); for j=1:nu; L=ulags(j); if L==0; uvals(j)=U(k); else idx=k-L; if idx>=1; uvals(j)=U(idx); else uvals(j)=0; end; end; end
        yvals=zeros(ny,1); for j=1:ny; L=ylags(j); idx=k-L; if idx>=1; yvals(j)=Yhat(idx); else yvals(j)=0; end; end
        x = [uvals(:)', yvals(:)']; for h=1:numel(W_local); z = x*W_local{h}; a = applyHiddenActivation(z, z_prev{h}, g, config); z_prev{h} = z; x=[x, a]; end
        Yhat(k)= x * w_o_local;
    end
end
