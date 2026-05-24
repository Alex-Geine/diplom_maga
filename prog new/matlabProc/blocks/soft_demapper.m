function llr = soft_demapper(rxSig, constellation, bitMap, noiseVar, H)
    if nargin < 5
        H = 1;
    end
    N = length(rxSig);
    bps = size(bitMap, 2);
    llr = zeros(N, bps);
    
    % Если H передан как скаляр, расширяем до вектора нужной длины
    if isscalar(H)
        H = H * ones(N, 1);
    end
    % Если noiseVar – скаляр, превращаем его в вектор для единообразия
    if isscalar(noiseVar)
        noiseVar = noiseVar * ones(N, 1);
    elseif length(noiseVar) ~= N
        error('Длина вектора noiseVar должна совпадать с длиной rxSig');
    end
    
    for b = 1:bps
        idx0 = (bitMap(:, b) == 0);
        idx1 = (bitMap(:, b) == 1);
        for i = 1:N
            scaledConst = H(i) * constellation;
            dists = abs(rxSig(i) - scaledConst).^2;
            minD0 = min(dists(idx0));
            minD1 = min(dists(idx1));
            llr(i, b) = (minD1 - minD0) / noiseVar(i); % здесь noiseVar тоже должен быть вектором
        end
    end
end