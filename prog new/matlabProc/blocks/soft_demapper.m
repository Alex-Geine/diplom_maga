function llr = soft_demapper(rxSig, constellation, bitMap, noiseVar, H)
% ОБНОВЛЕННЫЙ SOFT_DEMAPPER под каналы с замираниями и эквалайзером
% Если пятый аргумент H не передан, считаем канал идеальным (H = 1)
    if nargin < 5
        H = ones(size(rxSig));
    end

    N = length(rxSig);
    bps = size(bitMap, 2);
    llr = zeros(N, bps);

    for b = 1:bps
        idx0 = (bitMap(:, b) == 0);
        idx1 = (bitMap(:, b) == 1);
        
        for i = 1:N
            % Динамически масштабируем эталонное созвездие под текущий отсчет канала!
            % Это гарантирует точный расчет LLR при замираниях
            dists = abs(rxSig(i) - H(i) * constellation).^2;
            
            minD0 = min(dists(idx0));
            minD1 = min(dists(idx1));
            
            llr(i, b) = (minD1 - minD0) / noiseVar;
        end
    end
end
