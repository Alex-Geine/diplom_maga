function test_fec_system()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ
    clear; clc; close all;
    
    % Подключаем вашу папку с блоками
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    fprintf('==================================================\n');
    fprintf('       ЗАПУСК СКВОЗНОГО ТЕСТА FEC БЛОКОВ          \n');
    fprintf('==================================================\n\n');
    
    % Параметры симуляции
    modType = '16QAM';
    bps = 4;
    N_info = 4000;          % Число инфо-бит
    targetLength = 12000;   % Длина в канале (в 3 раза больше N_info -> жесткое повторение бит)
    numRowsInterleaver = 40;
    EbNo_dB = 5;            % Специально берем очень низкий SNR (высокий шум)
    
    %% 2. ПЕРЕДАТЧИК (FEC TX)
    txInfoBits = randi([0 1], N_info, 1);
    
    % Вызов объединенного блока FEC на передаче
    [txMatchedBits, lenCodedOrig, lenInterleavedOrig] = fec_tx(txInfoBits, numRowsInterleaver, targetLength);
    
    % Модуляция
    numSymbols = targetLength / bps;
    txBitsMatrix = reshape(txMatchedBits, numSymbols, bps);
    [txSig, constellation, bitMap] = mapper(txBitsMatrix, modType);
    
    %% 3. ДИСКРЕТНЫЙ КАНАЛ С ШУМОМ (AWGN)
    R_eff = N_info / targetLength; % Эффективная кодовая скорость (R_eff = 4000/12000 = 1/3)
    EsNo_dB = EbNo_dB + 10*log10(bps) + 10*log10(R_eff);
    EsNo = 10^(EsNo_dB/10);
    noiseVar = 1 / EsNo; 
    
    noise = sqrt(noiseVar/2) * (randn(size(txSig)) + 1i*randn(size(txSig)));
    rxSig = txSig + noise;
    
    %% 4. ПРИЕМНИК (FEC RX)
    % Мягкая демодуляция
    llrMatrix = soft_demapper(rxSig, constellation, bitMap, noiseVar);
    llrBitsStream = llrMatrix(:); % Вытягиваем по столбцам
    
    % Вызов объединенного блока FEC на приеме
    rxInfoBits = fec_rx(llrBitsStream, lenInterleavedOrig, numRowsInterleaver, lenCodedOrig, N_info);
    
    %% 5. АНАЛИЗ РЕЗУЛЬТАТОВ И BER
    % Расчет BER без кодирования (в канале до декодирования FEC)
    % Для честного сравнения сделаем это на основе LLR, прошедших рейт-рековери и деинтерливер
    llrAfterRecovery = rate_recovery(llrBitsStream, lenInterleavedOrig);
    llrDeinterleaved = deinterleaver(llrAfterRecovery, numRowsInterleaver, lenCodedOrig);
    hardCodedBits = (llrDeinterleaved < 0);
    
    % Исходный кодовый вектор для сравнения получаем быстрым кодером
    txCodedBits_check = conv_encoder(txInfoBits);
    ber_uncoded = sum(txCodedBits_check ~= hardCodedBits) / length(txCodedBits_check);
    
    % Расчет BER после работы полной системы FEC RX
    ber_coded = sum(txInfoBits ~= rxInfoBits) / N_info;
    
    % Вывод результатов в консоль
    fprintf('Параметры теста:\n');
    fprintf('  Модуляция: %s\n', modType);
    fprintf('  Эффективная скорость кода (R_eff): %.3f\n', R_eff);
    fprintf('  SNR (Eb/No): %.1f dB\n', EbNo_dB);
    fprintf('--------------------------------------------------\n');
    fprintf('[РЕЗУЛЬТАТ] BER в канале (без FEC): %g\n', ber_uncoded);
    fprintf('[РЕЗУЛЬТАТ] BER на выходе системы (с FEC): %g\n', ber_coded);
    fprintf('--------------------------------------------------\n');
    
    if ber_coded < ber_uncoded
        fprintf('[УСПЕХ] Макро-блоки fec_tx и fec_rx работают корректно.\n');
        fprintf('        Помехоустойчивая система успешно исправила ошибки канала!\n');
    else
        fprintf('[ОШИБКА] FEC-декодирование не привело к улучшению качества связи.\n');
    end
    fprintf('==================================================\n');
end
