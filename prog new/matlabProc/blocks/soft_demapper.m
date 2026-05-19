function llr = soft_demapper(rxSig, constellation, bitMap, noiseVar)
    % Оптимизированный по скорости демодулятор
    N = length(rxSig);
    bps = size(bitMap, 2);
    llr = zeros(N, bps);

    for b = 1:bps
        idx0 = (bitMap(:, b) == 0);
        idx1 = (bitMap(:, b) == 1);
        
        % Векторизованное вычисление расстояний для всех точек сразу
        % Позволяет избежать глубокого вложенного цикла
        for i = 1:N
            dists = abs(rxSig(i) - constellation).^2;
            minD0 = min(dists(idx0));
            minD1 = min(dists(idx1));
            llr(i, b) = (minD1 - minD0) / noiseVar;
        end
    end
end