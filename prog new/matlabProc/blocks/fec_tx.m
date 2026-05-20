function [txInterleavedBits, lenCodedOriginal, lenInterleavedOriginal] = fec_tx(txInfoBits, numRowsInterleaver, targetLength)
% FEC_TX Полный тракт помехоустойчивого кодирования на передатчике
% Вход:
%   txInfoBits         - Вектор информационных бит (0 или 1)
%   numRowsInterleaver - Глубина интерливера (число строк матрицы)
%   targetLength       - Целевой размер кадра в битах для рейт-матчера
% Выход:
%   txInterleavedBits      - Сформированный поток бит для маппера
%   lenCodedOriginal       - Исходная длина после кодера (нужна для деинтерливера)
%   lenInterleavedOriginal - Исходная длина после интерливера (нужна для рейт-рековери)

    % 1. Сверточное кодирование (Rate=1/2, K=7)
    txCodedBits = conv_encoder(txInfoBits);
    lenCodedOriginal = length(txCodedBits); 

    % 2. Блочное перемежение бит (Интерливер)
    txInterleavedBits_raw = interleaver(txCodedBits, numRowsInterleaver);
    lenInterleavedOriginal = length(txInterleavedBits_raw);

    % 3. Согласование скорости под физический канал (Рейт-Матчер)
    txInterleavedBits = rate_matcher(txInterleavedBits_raw, targetLength);
end
