function a = applyHiddenActivation(z, z_prev, g, config)
    % Apply activation with optional time-difference transform based on config.model.activation.
    if nargin < 2 || isempty(z_prev)
        z_prev = zeros(size(z), 'like', z);
    end
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
