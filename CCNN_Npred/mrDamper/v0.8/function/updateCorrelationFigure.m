function [plotHandle, figHandle] = updateCorrelationFigure(plotHandle, figHandle, corrHist)
    % Update or create the cumulative candidate correlation plot without resetting prior hidden units.
    xVals = 1:numel(corrHist);
    if isempty(figHandle) || ~ishandle(figHandle)
        figHandle = figure('Name','Candidate Correlation','Color','w');
    else
        figure(figHandle);
    end
    if isempty(plotHandle) || ~isvalid(plotHandle)
        clf(figHandle);
        plotHandle = plot(xVals, corrHist, '-o', 'LineWidth', 1.5, ...
        'MarkerSize', 4, ...
        'MarkerEdgeColor', 'b', ...
        'MarkerFaceColor', 'b');
        grid on;
        xlabel('Candidate epochs (cumulative)');
        ylabel('Correlation metric');
        title('Residual correlation during candidate training');
    else
        set(plotHandle, 'XData', xVals, 'YData', corrHist);
    end
    drawnow;
end
