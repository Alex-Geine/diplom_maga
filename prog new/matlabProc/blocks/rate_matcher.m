function outputBits = rate_matcher(inputBits, targetLength)
% RATE_MATCHER Согласование скорости через циклический буфер
% Вход: inputBits    — вектор закодированных бит (0 или 1)
%       targetLength — требуемая длина выходного потока бит

    N = length(inputBits);
    outputBits = zeros(targetLength, 1);
    
    if targetLength <= N
        % --- СЛУЧАЙ 1: ВЫКАЛЫВАНИЕ (Puncture) или точное совпадение ---
        % Просто берем первые targetLength бит
        outputBits = inputBits(1:targetLength);
    else
        % --- СЛУЧАЙ 2: ПОВТОРЕНИЕ (Repetition) ---
        % Циклически повторяем буфер данных, пока не наберем targetLength
        numRepeats = floor(targetLength / N);
        remBits = rem(targetLength, N);
        
        % Векторизованное заполнение
        outputBits(1:numRepeats*N) = repmat(inputBits, numRepeats, 1);
        if remBits > 0
            outputBits(numRepeats*N + 1 : end) = inputBits(1:remBits);
        end
    end
end
