function reportSymbolMachineS26()
% function reportSymbolMachineS26
%
% Creates a report on the quality of the forecasts fed into the Symbol
% Machine. Should be used after the whole sequence has been parsed by
% the Symbol Machine, i.e., after symbolMachineS26.m has been called once
% for each symbol appearing in the sequence.
%
% Inputs/outputs: none.
%
% Colorado School of Mines EENG311 - Spring 2026 - Mike Wakin

global SYMBOLDATA

if ~strcmp(SYMBOLDATA.machineVersion,'S26')
    error('Incorrect Symbol Machine version; use symbolMachineS26.m.');
end

fprintf('\n');
fprintf('|--BEGIN SYMBOL MACHINE S26 REPORT \n| ');
if SYMBOLDATA.nextIndex-1 < SYMBOLDATA.sequenceLength
    fprintf('ERROR: Did not complete forecasting the whole sequence.\n| ')
    fprintf('Processed only %d out of %d symbols in %s.\n| ',SYMBOLDATA.nextIndex-1,SYMBOLDATA.sequenceLength,SYMBOLDATA.filename);
else
    fprintf('Processed %d out of %d symbols in %s.\n| ',SYMBOLDATA.nextIndex-1,SYMBOLDATA.sequenceLength,SYMBOLDATA.filename);
    fprintf('Total penalty: %.3f bits (%.4f bits per symbol).\n|',SYMBOLDATA.totalPenaltyInBits,SYMBOLDATA.totalPenaltyInBits/SYMBOLDATA.sequenceLength);
    pctileBoundaries = [0 1/256 1/128 1/64 1/32 1/16 1/8 1/4 1/2 1]*100;
    for pctileIndex = 1:length(pctileBoundaries)-1
        if pctileBoundaries(pctileIndex) == 0
            % Include probabilities that are exactly 0.
            a = sum((SYMBOLDATA.winnerProbabilities >= pctileBoundaries(pctileIndex)/100) & (SYMBOLDATA.winnerProbabilities <= pctileBoundaries(pctileIndex+1)/100));
            b = sum((SYMBOLDATA.loserProbabilities(:) >= pctileBoundaries(pctileIndex)/100) & (SYMBOLDATA.loserProbabilities(:) <= pctileBoundaries(pctileIndex+1)/100));
        else
            a = sum((SYMBOLDATA.winnerProbabilities > pctileBoundaries(pctileIndex)/100) & (SYMBOLDATA.winnerProbabilities <= pctileBoundaries(pctileIndex+1)/100));
            b = sum((SYMBOLDATA.loserProbabilities(:) > pctileBoundaries(pctileIndex)/100) & (SYMBOLDATA.loserProbabilities(:) <= pctileBoundaries(pctileIndex+1)/100));
        end
        if a+b > 0
            fprintf('%9d probabilities forecasted between %1.4f (%d bits) and %1.4f (%d bits); actual occurrence rate %1.4f.\n|',a+b,pctileBoundaries(pctileIndex)/100,-log2(pctileBoundaries(pctileIndex)/100),pctileBoundaries(pctileIndex+1)/100,-log2(pctileBoundaries(pctileIndex+1)/100),a/(a+b));
        else
            fprintf('%9d probabilities forecasted between %1.4f (%d bits) and %1.4f (%d bits).\n|',0,pctileBoundaries(pctileIndex)/100,-log2(pctileBoundaries(pctileIndex)/100),pctileBoundaries(pctileIndex+1)/100,-log2(pctileBoundaries(pctileIndex+1)/100));
        end
    end
    % Disabled this; it caused some confusion about what metric was being 
    % used. But students can uncomment this if they are interested.
    % fprintf(' "Most likely" prediction came true %.3f%% of the time.\n|',100*SYMBOLDATA.correctPredictions/SYMBOLDATA.sequenceLength);
end

if SYMBOLDATA.verbose
    figure; hold on;
    imagesc(SYMBOLDATA.forecastedProbabilities',[0 1]); colorbar;
    title('Forecast probabilities');
    xlabel('Index');
    ylabel('Symbol');
    plot(SYMBOLDATA.sequence,'wo','LineWidth',2); axis tight;
end

fprintf('--END SYMBOL MACHINE S26 REPORT\n\n')