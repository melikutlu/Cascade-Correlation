function Yhat = recursivePredictFullSeries(U, Y, W_hidden, w_o, g, config)
    N = length(Y); 
    Yhat = zeros(N,1); 
    if N>=1; Yhat(1)=Y(1); end
    ulags = config.regressors.u(:)'; 
    ylags = config.regressors.y(:)'; 
    nu=numel(ulags); ny=numel(ylags);
    for k=2:N
        uvals=zeros(nu,1); for j=1:nu; L=ulags(j); if L==0; uvals(j)=U(k); else idx=k-L; if idx>=1; uvals(j)=U(idx); else uvals(j)=0; end; end; end
        yvals=zeros(ny,1); for j=1:ny; L=ylags(j); idx=k-L; if idx>=1; yvals(j)=Yhat(idx); else yvals(j)=0; end; end
        x = [uvals(:)', yvals(:)']; for h=1:numel(W_hidden); x=[x, g(x*W_hidden{h})]; end
        Yhat(k)= x * w_o;
    end
end
