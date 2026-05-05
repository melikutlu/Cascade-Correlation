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
        
            otherwise
                g = @(x) x;
        end
    end

    % determine activation mode safely (from config if available)
    mode = "";
    if isfield(config, 'model') && isfield(config.model, 'activation') && ~isempty(config.model.activation)
        mode = lower(string(config.model.activation));
    end

    clipLower = -10;
    clipUpper = 10;
    if isfield(config, 'model')
        if isfield(config.model, 'diff_clip_lower') && ~isempty(config.model.diff_clip_lower)
            clipLower = config.model.diff_clip_lower;
        end
        if isfield(config.model, 'diff_clip_upper') && ~isempty(config.model.diff_clip_upper)
            clipUpper = config.model.diff_clip_upper;
        end
    end
    if clipLower > clipUpper
        tmp = clipLower;
        clipLower = clipUpper;
        clipUpper = tmp;
    end

    clipEnabled = true;
    if isfield(config, 'model') && isfield(config.model, 'use_activation_clipping') && ~isempty(config.model.use_activation_clipping)
        clipEnabled = logical(config.model.use_activation_clipping);
    end

    dzdk = diffByStepOperator(z, z_prev);

    switch mode
        case {"diff", "diff-only"}
            % Tamamen dlarray'de işlem - GPU destekli
            % z_prev'dan epsilon hesapla (dlarray operasyonu)
            epsilon = max(1e-2, 0.01 * max(abs(z_prev)));
            
            % Element-wise bölme (dlarray)
            a = dzdk ./ (abs(z_prev) + epsilon);
            if clipEnabled
                a = max(min(a, clipUpper), clipLower);
            end
        case {"diff-tanh", "diff_tanh"}

            epsilon = max(1e-2, 0.01 * max(abs(z_prev)));
            
            % Element-wise bölme (dlarray)
            k = dzdk ./ (abs(z_prev) + epsilon);

            a = g(k);
            if clipEnabled
                a = max(min(a, clipUpper), clipLower);
            end
        otherwise
            a = g(z);
            if clipEnabled
                a = max(min(a, clipUpper), clipLower);
            end
    end
end

function dzdk = diffByStepOperator(z, z_prev)
    % Discrete derivative operator over full pre-activation:
    % d/dk z(k) ~= z(k) - z(k-1).
    dzdk = z - z_prev;
end
