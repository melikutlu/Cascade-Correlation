function [plotHandle, figHandle] = updateLossFigure(plotHandle, figHandle, mse_hist, figureName, plotTitle)
    if nargin < 4 || isempty(figureName)
        figureName = 'Train MSE vs Hidden Units';
    end
    if nargin < 5 || isempty(plotTitle)
        plotTitle = 'Loss Graph';
    end

    xVals = 0:numel(mse_hist)-1;
    if isempty(figHandle) || ~ishandle(figHandle)
        figHandle = figure('Name', figureName, 'Color','w');
    else
        figure(figHandle);
        set(figHandle, 'Name', figureName);
    end
    if isempty(plotHandle) || ~isvalid(plotHandle)
        clf(figHandle);
        plotHandle = plot(xVals, mse_hist, '-o', 'LineWidth', 1.4);
        grid on;
        xlabel('Hidden Units');
        ylabel('Train MSE');
        title(plotTitle);
    else
        set(plotHandle, 'XData', xVals, 'YData', mse_hist);
    end
    drawnow;
end
