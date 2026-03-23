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

    switch mode
        case {"diff", "diff-only"}
            a = z - z_prev;
        case {"diff-tanh", "diff_tanh"}
            a = g(z - z_prev);
        otherwise
            a = g(z);
    end
end
