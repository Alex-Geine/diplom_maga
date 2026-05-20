function llr = soft_demapper(rxSig, constellation, bitMap, noiseVar)
% OPTIMIZED_SOFT_DEMAPPER Вычисляет мягкие решения Max-Log LLR с нормировкой метрик
    N = length(rxSig);
    bps = size(bitMap, 2);
    llr = zeros(N, bps);

    for b = 1:bps
        idx0 = (bitMap(:, b) == 0);
        idx1 = (bitMap(:, b) == 1);
        
        for i = 1:N
            dists = abs(rxSig(i) - constellation).^2;
            minD0 = min(dists(idx0));
            minD1 = min(dists(idx1));
            
            % Точная Max-Log LLR метрика: деление разности на дисперсию шума.
            % Для многоуровневых созвездий (16/256QAM) это выравнивает масштаб 
            % между старшими и младшими битами после их перемешивания интерливером.
            llr(i, b) = (minD1 - minD0) / noiseVar;
        end
    end
end
