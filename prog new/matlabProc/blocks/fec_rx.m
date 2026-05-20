function rxInfoBits = fec_rx(llrBitsStream, lenInterleavedOriginal, numRowsInterleaver, lenCodedOriginal, N_info)
% FEC_RX Полный тракт помехоустойчивого декодирования на приемнике
% Вход:
%   llrBitsStream          - Поток LLR-метрик от демаппера
%   lenInterleavedOriginal - Длина до рейт-матчера (из fec_tx)
%   numRowsInterleaver     - Глубина деинтерливера (из fec_tx)
%   lenCodedOriginal       - Длина до интерливера (из fec_tx)
%   N_info                 - Ожидаемое количество информационных бит
% Выход:
%   rxInfoBits             - Восстановленные информационные биты

    % 1. Восстановление скорости и накопление энергии LLR (Рейт-Рековери)
    llrAfterRecovery = rate_recovery(llrBitsStream, lenInterleavedOriginal);

    % 2. Обратное перемежение мягких метрик LLR (Деинтерливер)
    llrDeinterleaved = deinterleaver(llrAfterRecovery, numRowsInterleaver, lenCodedOriginal);

    % 3. Мягкое декодирование Витерби
    rxInfoBits = viterbi_soft_decoder(llrDeinterleaved, N_info);
end
