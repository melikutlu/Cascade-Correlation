function diffOrder = getDiffOrder(config)
%GETDIFFORDER Determine differencing order from model activation mode.
    if isfield(config, 'model') && isfield(config.model, 'activation')
        mode = lower(string(config.model.activation));
        switch mode
            case {"diff", "diff-only", "diff-tanh", "diff_tanh"}
                diffOrder = 1;
            case {"diff2", "diff2-only", "diff2-tanh"}
                diffOrder = 2;
            otherwise
                diffOrder = 0;
        end
    else
        diffOrder = 0;
    end
end
