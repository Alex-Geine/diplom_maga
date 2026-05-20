function sim_ber_vs_snr_macro()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ И ПОДКЛЮЧЕНИЕ ПАПОК
    clear; clc; close all;
    
    % Подключаем вашу папку с блоками
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    % Диапазон Eb/No (в дБ) для симуляции
    EbNo_vec = -10:2:18; 
    
    % Список типов модуляций для анализа
    modTypes = {'QPSK', '16QAM', '64QAM', '256QAM'};
    colors = {'b', 'r', 'g', 'm'};
    
    % Жестко задаем размер физического кадра в эфире (в битах)
    targetLength = 24000; 
    numRowsInterleaver = 40; % Глубина интерливера
    
    % Массивы для хранения результатов BER
    BER_coded_results = zeros(length(modTypes), length(EbNo_vec));
    BER_uncoded_results = zeros(length(modTypes), length(EbNo_vec));
    
    %% 2. ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ
    for m = 1:length(modTypes)
        modType = modTypes{m};
        
        % Настройка количества ИНФОРМАЦИОННЫХ бит
        switch modType
            case 'QPSK',   bps = 2; N_info = 10000;  
            case '16QAM',  bps = 4; N_info = 10000;
            case '64QAM',  bps = 6; N_info = 10020; % Кратность для интерливера
            case '256QAM', bps = 8; N_info = 10000;
        end
        
        % Вычисляем эффективную скорость кода (Effective Code Rate)
        R_eff = N_info / targetLength;
        fprintf('Симуляция для модуляции: %s (R_eff = %.3f)...\n', modType, R_eff);
        
        for s = 1:length(EbNo_vec)
            EbNo_dB = EbNo_vec(s);
            fprintf('  %s | Eb/No: %d dB\n', modType, EbNo_dB);
            
            % 2.1 Генерация данных и ПЕРЕДАТЧИК (FEC TX)
            txInfoBits = randi([0 1], N_info, 1);
            [txMatchedBits, lenCodedOrig, lenInterleavedOrig] = ...
                fec_tx(txInfoBits, numRowsInterleaver, targetLength);
            
            % 2.2 Модуляция (Маппер)
            numSymbols = targetLength / bps;
            txBitsMatrix = reshape(txMatchedBits, numSymbols, bps);
            [txSig, constellation, bitMap] = mapper(txBitsMatrix, modType);
            
            % 2.3 РАДИОКАНАЛ С ШУМОМ (AWGN)
            EsNo_dB = EbNo_dB + 10*log10(bps) + 10*log10(R_eff);
            EsNo = 10^(EsNo_dB/10);
            noiseVar = 1 / EsNo; 
            noise = sqrt(noiseVar/2) * (randn(size(txSig)) + 1i*randn(size(txSig)));
            rxSig = txSig + noise;
            
            % 2.4 Мягкая демодуляция (Демаппер)
            llrMatrix = soft_demapper(rxSig, constellation, bitMap, noiseVar);
            llrBitsStream = llrMatrix(:); % Вытягиваем строго по столбцам
            
            % 2.5 ПРИЕМНИК (FEC RX)
            rxInfoBits = fec_rx(llrBitsStream, lenInterleavedOrig, ...
                                numRowsInterleaver, lenCodedOrig, N_info);
            
            % 2.6 Расчет BER с кодированием (после FEC RX)
            numErrorsCoded = sum(txInfoBits ~= rxInfoBits);
            BER_coded_results(m, s) = numErrorsCoded / N_info;
            
            % 2.7 Расчет BER без кодирования (в канале, для пунктирной линии)
            % Используем внутреннюю математику для получения "сырых" бит из LLR
            llrAfterRecovery = rate_recovery(llrBitsStream, lenInterleavedOrig);
            llrDeinterleaved = deinterleaver(llrAfterRecovery, numRowsInterleaver, lenCodedOrig);
            hardCodedBits = (llrDeinterleaved < 0);
            
            % Генерируем эталонный кодовый вектор для точного подсчета ошибок
            txCodedBits_check = conv_encoder(txInfoBits);
            numErrorsUncoded = sum(txCodedBits_check ~= hardCodedBits);
            BER_uncoded_results(m, s) = numErrorsUncoded / length(txCodedBits_check);
            
            % Быстрый выход из цикла при достижении нулевого BER
            if numErrorsCoded == 0 && s > 4
                BER_coded_results(m, s:end) = 0;
                break;
            end
        end
    end

    %% 3. ПОСТРОЕНИЕ ГРАФИКОВ BER vs SNR
    figure('Color', 'w', 'Position', [150, 150, 950, 650]);
    
    plots_for_legend = [];
    legend_labels = {};
    
    for m = 1:length(modTypes)
        valid_idx_coded = BER_coded_results(m, :) > 0;
        valid_idx_uncoded = BER_uncoded_results(m, :) > 0;
        
        % Сплошная линия — Макро-блоки FEC (С кодированием, интерливером и рейт-матчером)
        p_coded = semilogy(EbNo_vec(valid_idx_coded), BER_coded_results(m, valid_idx_coded), ...
            [colors{m} '-'], 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 5);
        hold on;
        
        % Пунктирная линия — Базовый радиоканал без FEC
        semilogy(EbNo_vec(valid_idx_uncoded), BER_uncoded_results(m, valid_idx_uncoded), ... 
            [colors{m} '--'], 'LineWidth', 1.2);
        
        plots_for_legend = [plots_for_legend, p_coded]; %#ok<AGROW>
        
        % Формируем подпись для легенды с указанием кодовой скорости
        switch modTypes{m}
            case 'QPSK',   N_i = 10000;
            case '16QAM',  N_i = 10000;
            case '64QAM',  N_i = 10020;
            case '256QAM', N_i = 10000;
        end
        legend_labels = [legend_labels, {sprintf('%s (R_{eff} = %.2f)', modTypes{m}, N_i/targetLength)}]; %#ok<AGROW>
    end
    
    % Добавление служебных линий стиля в легенду
    dummy_solid = plot(NaN, NaN, 'k-', 'LineWidth', 2);
    dummy_dashed = plot(NaN, NaN, 'k--', 'LineWidth', 1.2);
    plots_for_legend = [plots_for_legend, dummy_solid, dummy_dashed];
    legend_labels = [legend_labels, {'С кодированием (FEC Blocks)', 'Без кодирования (Канал)'}];
    
    grid on;
    set(gca, 'YScale', 'log'); 
    ylim([1e-5 1]); 
    xlim([EbNo_vec(1) EbNo_vec(end)]);
    
    title('Кривые BER vs SNR: Высокоуровневые макро-блоки fec\_tx и fec\_rx');
    xlabel('E_b/N_0 (dB)');
    ylabel('Bit Error Rate (BER)');
    legend(plots_for_legend, legend_labels, 'Location', 'southwest');
    
    fprintf('\nРасчет SNR кривых для макро-структуры завершен!\n');
end
