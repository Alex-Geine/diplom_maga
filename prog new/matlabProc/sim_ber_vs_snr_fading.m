function sim_ber_vs_snr_fading()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ И ПОДКЛЮЧЕНИЕ ПАПОК
    clear; clc; close all;
    
    % Подключаем вашу папку с блоками
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    % Диапазон Eb/No (в дБ) для симуляции в канале с замираниями
    EbNo_vec = 0:2:24; 
    
    % Список типов модуляций для анализа
    modTypes = {'QPSK', '16QAM', '64QAM', '256QAM'};
    colors = {'b', 'r', 'g', 'm'};
    
    % Параметры системы
    numRowsInterleaver = 40; 
    targetLength = 24000; % Жестко заданный размер физического кадра
    
    % Массивы для хранения результатов BER
    BER_coded_results = zeros(length(modTypes), length(EbNo_vec));
    BER_uncoded_results = zeros(length(modTypes), length(EbNo_vec));
    
    %% 2. ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ
    for m = 1:length(modTypes)
        modType = modTypes{m};
        
        % Настройка количества информационных бит
        switch modType
            case 'QPSK',   bps = 2; N_info = 10000;  
            case '16QAM',  bps = 4; N_info = 10000;
            case '64QAM',  bps = 6; N_info = 10020; 
            case '256QAM', bps = 8; N_info = 10000;
        end
        
        R_eff = N_info / targetLength;
        fprintf('Симуляция: %s в канале Рэлея (R_eff = %.3f)...\n', modType, R_eff);
        
        for s = 1:length(EbNo_vec)
            EbNo_dB = EbNo_vec(s);
            
            % 2.1 Генерация данных и ПЕРЕДАТЧИК (FEC TX)
            txInfoBits = randi([0 1], N_info, 1);
            [txMatchedBits, lenCodedOrig, lenInterleavedOrig] = ...
                fec_tx(txInfoBits, numRowsInterleaver, targetLength);
            
            % 2.2 Модуляция (Маппер)
            numSymbols = targetLength / bps;
            txBitsMatrix = reshape(txMatchedBits, numSymbols, bps);
            [txSig, constellation, bitMap] = mapper(txBitsMatrix, modType);
            
            % 2.3 КАНАЛ С ЗАМИРАНИЯМИ РЭЛЕЯ И AWGN ШУМОМ
            EsNo_dB = EbNo_dB + 10*log10(bps) + 10*log10(R_eff);
            EsNo = 10^(EsNo_dB/10);
            noiseVar = 1 / EsNo; 
            
            % Генерация комплексных коэффициентов замираний Рэлея
            H = (randn(size(txSig)) + 1i*randn(size(txSig))) / sqrt(2);
            fadedSig = txSig .* H;
            
            noise = sqrt(noiseVar/2) * (randn(size(fadedSig)) + 1i*randn(size(fadedSig)));
            rxSig = fadedSig + noise;
            
            % 2.4 МЯГКИЙ ПРИЕМНИК (MMSE ЭКВАЛАЙЗЕР + ДЕМАППЕР)
            % Выравниваем сигнал эквалайзером
            [eqSig, ~] = mmse_equalizer(rxSig, H, noiseVar);
            
            % Демаппируем БЕЗ передачи H (как подтвердил успешный тест)
            llrMatrix = soft_demapper(eqSig, constellation, bitMap, noiseVar);
            llrBitsStream = llrMatrix(:); 
            
            % 2.5 ДЕКОДЕР (FEC RX)
            rxInfoBits = fec_rx(llrBitsStream, lenInterleavedOrig, ...
                                numRowsInterleaver, lenCodedOrig, N_info);
            
            % 2.6 Расчет BER с кодированием (Полный тракт)
            numErrorsCoded = sum(txInfoBits ~= rxInfoBits);
            BER_coded_results(m, s) = numErrorsCoded / N_info;
            
            % 2.7 Расчет BER без кодирования (Только MMSE эквалайзер в канале)
            llrAfterRecovery = rate_recovery(llrBitsStream, lenInterleavedOrig);
            llrDeinterleaved = deinterleaver(llrAfterRecovery, numRowsInterleaver, lenCodedOrig);
            hardCodedBits = (llrDeinterleaved < 0);
            
            txCodedBits_check = conv_encoder(txInfoBits);
            numErrorsUncoded = sum(txCodedBits_check ~= hardCodedBits);
            BER_uncoded_results(m, s) = numErrorsUncoded / length(txCodedBits_check);
            
            % Критерий останова симуляции при достижении нулевого BER
            if numErrorsCoded == 0 && s > 5
                BER_coded_results(m, s:end) = 0;
                break;
            end
        end
    end

    %% 3. ПОСТРОЕНИЕ ГРАФИКОВ BER vs SNR
    figure('Color', 'w', 'Position', [150, 100, 950, 650]);
    
    plots_for_legend = [];
    legend_labels = {};
    
    for m = 1:length(modTypes)
        valid_idx_coded = BER_coded_results(m, :) > 0;
        valid_idx_uncoded = BER_uncoded_results(m, :) > 0;
        
        % Сплошная линия — Полный тракт (MMSE + FEC)
        p_coded = semilogy(EbNo_vec(valid_idx_coded), BER_coded_results(m, valid_idx_coded), ...
            [colors{m} '-'], 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 5);
        hold on;
        
        % Пунктирная линия — Только MMSE эквалайзер (Без FEC)
        semilogy(EbNo_vec(valid_idx_uncoded), BER_uncoded_results(m, valid_idx_uncoded), ... 
            [colors{m} '--'], 'LineWidth', 1.2);
        
        plots_for_legend = [plots_for_legend, p_coded]; %#ok<AGROW>
        legend_labels = [legend_labels, {sprintf('%s (MMSE + FEC)', modTypes{m})}]; %#ok<AGROW>
    end
    
    % Добавление служебных линий стиля в легенду
    dummy_solid = plot(NaN, NaN, 'k-', 'LineWidth', 2);
    dummy_dashed = plot(NaN, NaN, 'k--', 'LineWidth', 1.2);
    plots_for_legend = [plots_for_legend, dummy_solid, dummy_dashed];
    legend_labels = [legend_labels, {'С кодированием (FEC + MMSE)', 'Без кодирования (Только MMSE)'}];
    
    grid on;
    set(gca, 'YScale', 'log'); 
    ylim([1e-5 1]); 
    xlim([EbNo_vec(1) EbNo_vec(end)]);
    
    title('Кривые BER vs SNR в канале Рэлея с замираниями (Фикс. размер кадра = 24000 бит)');
    xlabel('E_b/N_0 (dB)');
    ylabel('Bit Error Rate (BER)');
    legend(plots_for_legend, legend_labels, 'Location', 'southwest');
    
    % Автоматическое сохранение результатов для таблиц диплома
    save('fading_simulation_results.mat', 'EbNo_vec', 'modTypes', 'BER_coded_results', 'BER_uncoded_results');
    fprintf('\nСимуляция успешно завершена! Данные сохранены в fading_simulation_results.mat\n');
end
