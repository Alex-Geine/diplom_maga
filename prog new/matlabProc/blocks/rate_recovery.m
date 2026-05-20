function recoveredLLR = rate_recovery(rxLLR, originalLength)
% RATE_RECOVERY Восстановление скорости и объединение метрик LLR
% Вход: rxLLR          — вектор LLR от демодулятора (длиной targetLength)
%       originalLength — исходная длина вектора до рейт-матчера (длина txCodedBits)

    targetLength = length(rxLLR);
    recoveredLLR = zeros(originalLength, 1);
    
    if targetLength <= originalLength
        % --- СЛУЧАЙ 1: БЫЛО ВЫКАЛЫВАНИЕ (Puncture) ---
        % Возвращаем принятые LLR на свои места. 
        % Остаток вектора (выколотые биты) заполняется нейтральными нулями!
        recoveredLLR(1:targetLength) = rxLLR;
    else
        % --- СЛУЧАЙ 2: БЫЛО ПОВТОРЕНИЕ (Repetition) ---
        % Суммируем LLR повторенных бит для накопления энергии (MRC)
        numRepeats = floor(targetLength / originalLength);
        remBits = rem(targetLength, originalLength);
        
        % Суммируем полные блоки
        for i = 1:numRepeats
            idx_src = (i-1)*originalLength + 1 : i*originalLength;
            recoveredLLR = recoveredLLR + rxLLR(idx_src);
        end
        
        % Суммируем оставшийся хвостик
        if remBits > 0
            idx_src = numRepeats*originalLength + 1 : targetLength;
            recoveredLLR(1:remBits) = recoveredLLR(1:remBits) + rxLLR(idx_src);
        end
    end
end
