function [plotHandle, figHandle] = updateLossFigure(plotHandle, figHandle, mse_hist)
    xVals = 0:numel(mse_hist)-1;
    if isempty(figHandle) || ~ishandle(figHandle)
        figHandle = figure('Name','Train MSE vs Hidden Units','Color','w');
    else
        figure(figHandle);
    end
    if isempty(plotHandle) || ~isvalid(plotHandle)
        clf(figHandle);
        plotHandle = plot(xVals, mse_hist, '-o', 'LineWidth', 1.4);
        grid on;
        xlabel('Hidden Units');
        ylabel('Train MSE');
        title('Loss Graph');
    else
        set(plotHandle, 'XData', xVals, 'YData', mse_hist);
    end
    drawnow;
end
