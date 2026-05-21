function test_fec_mmse_system()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ
    clear; clc; close all;
    
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    fprintf('==================================================\n');
    fprintf('   ЗАПУСК ТЕСТА FEC СИСТЕМЫ С MMSE ЭКВАЛАЙЗЕРОМ   \n');
    fprintf('==================================================\n\n');
    
    % Параметры симуляции
    modType = '16QAM';
    bps = 4;
    N_info = 4000;          
    targetLength = 16000;   % R_eff = 4000/16000 = 1/4
    numRowsInterleaver = 40;
    EbNo_dB = 8;            
    
    %% 2. ПЕРЕДАТЧИК (FEC TX + Модуляция)
    txInfoBits = randi([0 1], N_info, 1);
    [txMatchedBits, lenCodedOrig, lenInterleavedOrig] = fec_tx(txInfoBits, numRowsInterleaver, targetLength);
    
    numSymbols = targetLength / bps;
    txBitsMatrix = reshape(txMatchedBits, numSymbols, bps);
    [txSig, constellation, bitMap] = mapper(txBitsMatrix, modType);
    
    %% 3. КАНАЛ С ЗАМИРАНИЯМИ (FADING) И ШУМОМ
    R_eff = N_info / targetLength; 
    EsNo_dB = EbNo_dB + 10*log10(bps) + 10*log10(R_eff);
    EsNo = 10^(EsNo_dB/10);
    noiseVar = 1 / EsNo; 
    
    % Генерация комплексных коэффициентов канала Рэлея (замирания амплитуды и фазы)
    % Каждый символ умножается на свой случайный коэффициент H
    H = (randn(size(txSig)) + 1i*randn(size(txSig))) / sqrt(2);
    
    % Искажение сигнала каналом и добавление белого шума
    fadedSig = txSig .* H;
    noise = sqrt(noiseVar/2) * (randn(size(fadedSig)) + 1i*randn(size(fadedSig)));
    rxSig = fadedSig + noise;
    
    %% 4. ПРИЕМНИК (MMSE ЭКВАЛАЙЗЕР + FEC RX)
    % Вызов MMSE Эквалайзера
    [eqSig] = mmse_equalizer(rxSig, H, noiseVar);
    
    % Мягкая демодуляция
    % Передаем эквализованный сигнал, но с учетом эффективного профиля канала для точного расчета LLR
    % Эффективный профиль канала после эквалайзера равен H_eff = H .* mmseWeights
    %H_eff = H .* mmseWeights;
    H_eff = (abs(H(:)).^2) ./ (abs(H(:)).^2 + noiseVar);
    llrMatrix = soft_demapper(eqSig, constellation, bitMap, noiseVar);
    llrBitsStream = llrMatrix(:); 
    
    % Декодирование FEC
    rxInfoBits = fec_rx(llrBitsStream, lenInterleavedOrig, numRowsInterleaver, lenCodedOrig, N_info);
    
    %% 5. АНАЛИЗ РЕЗУЛЬТАТОВ И СТАТИСТИКА
    ber_coded = sum(txInfoBits ~= rxInfoBits) / N_info;
    
    % Для демонстрации: найдем BER без эквалайзера (если бы мы демаппировали сразу искаженный сигнал)
    llrMatrix_noEq = soft_demapper(rxSig, constellation, bitMap, noiseVar);
    llrBitsStream_noEq = llrMatrix_noEq(:);
    llrAfterRecovery_noEq = rate_recovery(llrBitsStream_noEq, lenInterleavedOrig);
    llrDeinterleaved_noEq = deinterleaver(llrAfterRecovery_noEq, numRowsInterleaver, lenCodedOrig);
    rxInfoBits_noEq = viterbi_soft_decoder(llrDeinterleaved_noEq, N_info);
    ber_noEq = sum(txInfoBits ~= rxInfoBits_noEq) / N_info;
    
    fprintf('Параметры трансивера:\n');
    fprintf('  Модуляция: %s | SNR (Eb/No): %.1f dB\n', modType, EbNo_dB);
    fprintf('  Профиль канала: Частотно-плоские замирания Рэлея\n');
    fprintf('--------------------------------------------------\n');
    fprintf('[АНАЛИЗ] BER БЕЗ эквалайзера (только FEC): %g\n', ber_noEq);
    fprintf('[АНАЛИЗ] BER С MMSE эквалайзером + FEC:     %g\n', ber_coded);
    fprintf('--------------------------------------------------\n');
    
    %% 6. ВИЗУАЛИЗАЦИЯ ДЛЯ ДИПЛОМА
    figure('Color', 'w');
    
    subplot(1,2,1);
    plot(real(rxSig), imag(rxSig), '.', 'Color', [0.8 0.3 0.3]);
    grid on; axis square;
    title('До эквалайзера (Замирания + Шум)');
    xlabel('I'); ylabel('Q');
    
    subplot(1,2,2);
    plot(real(eqSig), imag(eqSig), '.', 'Color', [0.3 0.6 0.3]);
    hold on;
    plot(real(constellation), imag(constellation), 'r+', 'LineWidth', 2);
    grid on; axis square;
    title('После MMSE эквалайзера');
    xlabel('I'); ylabel('Q');
    
    if ber_coded < ber_noEq
        fprintf('[УСПЕХ] MMSE эквалайзер успешно восстановил ортогональность созвездия.\n');
    end
end
