function g = resolveActivation(name)
%RESOLVEACTIVATION Return activation function handle based on name.
% Supported: tanh (default), sigmoid, relu, leakyrelu, linear.
    if nargin < 1 || isempty(name)
        name = 'tanh';
    end

    switch lower(string(name))
        case {'tanh','hyperbolic'}
            g = @(x) tanh(x);
        case {'sigmoid','logistic'}
            g = @(x) 1 ./ (1 + exp(-x));
        case 'relu'
            g = @(x) max(x, 0);
        case 'leakyrelu'
            alpha = 0.01;
            g = @(x) max(alpha * x, x);
        case {'linear','identity','none','diff','diff-only','diff_tanh','diff-tanh'}
            % diff flavors are handled by applyHiddenActivation via order; keep nonlinearity identity
            g = @(x) x;
        otherwise
            warning('Unknown activation %s, falling back to tanh.', string(name));
            g = @(x) tanh(x);
    end
end
