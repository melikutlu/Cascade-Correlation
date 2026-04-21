function a = applyHiddenActivation(z, z_prev, g, config)
    % Apply activation with optional time-difference transform based on config.model.activation.
    if nargin < 2 || isempty(z_prev)
        z_prev = zeros(size(z), 'like', z);
    end
    % If g is not a function handle, resolve it from the provided value or from config
    if ~isa(g, 'function_handle')
        % try interpret g as a name/string; otherwise fall back to config
        modeStr = "";
        if ischar(g) || isstring(g)
            modeStr = lower(string(g));
        elseif isfield(config, 'model') && isfield(config.model, 'activation')
            modeStr = lower(string(config.model.activation));
        end
        switch modeStr
            case {"tanh","diff-tanh", "diff_tanh"}
                g = @(x) tanh(x);
            case {"diff-sigmoid", "diff_sigmoid"}
                g = @(x) 1 ./ (1 + exp(-x));
        
            otherwise
                g = @(x) x;
        end
    end

    % determine activation mode safely (from config if available)
    mode = "";
    if isfield(config, 'model') && isfield(config.model, 'activation') && ~isempty(config.model.activation)
        mode = lower(string(config.model.activation));
    end

    dzdk = diffByStepOperator(z, z_prev);

    switch mode
        case {"diff", "diff-only"}
            % Tamamen dlarray'de işlem - GPU destekli
            % z_prev'dan epsilon hesapla (dlarray operasyonu)
            epsilon = max(1e-2, 0.01 * max(abs(z_prev)));
            
            % Element-wise bölme (dlarray)
            a = dzdk ./ (abs(z_prev) + epsilon);
            
            % Gradient clipping (dlarray)
            a = max(min(a, 10), -10);
        case {"diff-sigmoid", "diff_sigmoid"}
            a = g(dzdk);
        case {"diff-tanh", "diff_tanh"}
            a = g(dzdk);
            a = max(min(a, 10), -10);
        otherwise
            a = g(z);
    end
end

function dzdk = diffByStepOperator(z, z_prev)
    dzdk = z - z_prev;
end
