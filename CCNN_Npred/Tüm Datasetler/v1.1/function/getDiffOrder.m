function diffOrder = getDiffOrder(config)
%GETDIFFORDER Determine differencing order from model activation mode.
    if isfield(config, 'model') && isfield(config.model, 'activation')
        mode = lower(string(config.model.activation));
        switch mode
            case {"diff", "diff-only", "diff-tanh", "diff_tanh", ...
                "tustin", "tustin-only", "diff-tustin", "diff_tustin", "difftustin", "diff-tustinn", "diff_tustinn", "difftustinn", ...
                "tanh-tustin", "tanh_tustin", "tanh-diff-tustin", "tanh_diff_tustin", "tanh-difftustin", "tanhdifftustin", "tanh-difftustinn", ...
                "tanh-diff-tustinn", "tanh_diff_tustinn", "tanhdifftustinn", ...
                "sigmoid-tustin", "sigmoid_tustin", "sigmoid-diff-tustin", "sigmoid_diff_tustin", "sigmoid-difftustin", "sigmoiddifftustin", "sigmoid-difftustinn", ...
                "sigmoid-diff-tustinn", "sigmoid_diff_tustinn", "sigmoiddifftustinn"}
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
