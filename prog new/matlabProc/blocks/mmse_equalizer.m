function [eqSig, mmseWeights] = mmse_equalizer(rxSig, H, noiseVar)
% MMSE_EQUALIZER Однополосный частотный эквалайзер MMSE (Вектор-столбец версия)
% Вход и выход гарантированно являются векторами-столбцами

    % Принудительно вытягиваем всё в столбцы
    rx = rxSig(:);
    H_col = H(:);

    % Формула MMSE: W = H* / (|H|^2 + noiseVar)
    H_conj = conj(H_col);
    H_mag2 = abs(H_col).^2;
    
    % Вычисляем веса (тоже столбец)
    mmseWeights = H_conj ./ (H_mag2 + noiseVar);
    
    % Поэлементное выравнивание сигнала
    eqSig = rx .* mmseWeights;
end
