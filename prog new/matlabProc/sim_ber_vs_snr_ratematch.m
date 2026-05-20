function sim_ber_vs_snr_ratematch()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ И ПОДКЛЮЧЕНИЕ ПАПОК
    clear; clc; close all;
    
    % Подключаем вашу папку с блоками
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    % Диапазон Eb/No (в дБ) для симуляции
    EbNo_vec = -10:2:18; 
    
    % Список типов модуляций для анализа
    modTypes = {'QPSK', '16QAM', '64QAM', '256QAM'};
    colors = {'b', 'r', 'g', 'm'};
    
    % Параметры системы
    numRowsInterleaver = 40; % Глубина интерливера
    R_code = 1/2;           % Базовая скорость сверточного кодера
    
    % Жестко задаем размер физического кадра в эфире (в битах)
    % Все модуляции будут приводиться рейт-матчером именно к этому размеру!
    targetLength = 24000; 
    
    % Массивы для хранения результатов BER
    BER_coded_results = zeros(length(modTypes), length(EbNo_vec));
    BER_uncoded_results = zeros(length(modTypes), length(EbNo_vec));
    
    %% 2. ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ
    for m = 1:length(modTypes)
        modType = modTypes{m};
        
        % Настройка количества ИНФОРМАЦИОННЫХ бит
        % Сделаем длину N_info фиксированной, чтобы наглядно увидеть,
        % как рейт-матчер меняет эффективную скорость кода для разных модуляций.
        switch modType
            case 'QPSK',   bps = 2; N_info = 10000;  
            case '16QAM',  bps = 4; N_info = 10000;
            case '64QAM',  bps = 6; N_info = 10020; % Кратность для интерливера
            case '256QAM', bps = 8; N_info = 10000;
        end
        
        % Вычисляем эффективную скорость кода (Effective Code Rate) после рейт-матчера.
        % Она равна: (Инфо_биты) / (Биты_в_канале)
        R_eff = N_info / targetLength;
        
        fprintf('Симуляция для модуляции: %s (Эффективная скорость кода R_eff = %.3f)...\n', modType, R_eff);
        
        for s = 1:length(EbNo_vec)
            EbNo_dB = EbNo_vec(s);
            fprintf('  %s | Eb/No: %d dB\n', modType, EbNo_dB);
            
            % 2.1 Генерация данных и сверточное кодирование
            txInfoBits = randi([0 1], N_info, 1);
            txCodedBits = conv_encoder(txInfoBits);
            lenCodedOriginal = length(txCodedBits); 
            
            % 2.2 Блочное перемежение (Интерливер)
            txInterleavedBits = interleaver(txCodedBits, numRowsInterleaver);
            lenInterleavedOriginal = length(txInterleavedBits);
            
            % 2.3 СОГЛАСОВАНИЕ СКОРОСТИ (Рейт-Матчер)
            % Приводим поток к жесткому размеру targetLength
            txMatchedBits = rate_matcher(txInterleavedBits, targetLength);
            
            % Перегруппировка под формат маппера строго по столбцам (без транспонирования)
            numSymbols = targetLength / bps;
            txBitsMatrix = reshape(txMatchedBits, numSymbols, bps);
            
            % 2.4 Модуляция (Ваш внешний файл mapper.m)
            [txSig, constellation, bitMap] = mapper(txBitsMatrix, modType);
            
            % 2.5 Добавление шума AWGN с учетом эффективной скорости кодирования (R_eff)
            % ВАЖНО: чтобы график BER строился именно от энергии на ИНФОРМАЦИОННЫЙ бит (Eb),
            % мы обязаны использовать в формуле шума эффективную скорость R_eff, 
            % которая автоматически учитывает и выкалывание, и повторение бит!
            EsNo_dB = EbNo_dB + 10*log10(bps) + 10*log10(R_eff);
            EsNo = 10^(EsNo_dB/10);
            noiseVar = 1 / EsNo; 
            noise = sqrt(noiseVar/2) * (randn(size(txSig)) + 1i*randn(size(txSig)));
            rxSig = txSig + noise;
            
            % 2.6 Мягкая демодуляция (Ваш внешний файл soft_demapper.m)
            llrMatrix = soft_demapper(rxSig, constellation, bitMap, noiseVar);
            
            % Вытягиваем LLR в один непрерывный битовый поток по столбцам
            llrBitsStream = llrMatrix(:);
            
            % 2.7 ВОССТАНОВЛЕНИЕ СКОРОСТИ (Рейт-Рековери)
            % Возвращаем длину от targetLength обратно к размеру после интерливера.
            % Выколотые биты заполнятся нулями, повторенные — просуммируются.
            llrAfterRecovery = rate_recovery(llrBitsStream, lenInterleavedOriginal);
            
            % 2.8 Обратное перемежение LLR (Деинтерливер)
            llrDeinterleaved = deinterleaver(llrAfterRecovery, numRowsInterleaver, lenCodedOriginal);
            
            % 2.9 Декодирование Витерби на основе восстановленных LLR
            rxInfoBits = viterbi_soft_decoder(llrDeinterleaved, N_info);
            
            % 2.10 Расчет BER с кодированием (после Витерби)
            numErrorsCoded = sum(txInfoBits ~= rxInfoBits);
            BER_coded_results(m, s) = numErrorsCoded / N_info;
            
            % 2.11 Расчет BER без кодирования (в канале, на основе восстановленных LLR)
            hardCodedBits = (llrDeinterleaved < 0);
            numErrorsUncoded = sum(txCodedBits ~= hardCodedBits);
            BER_uncoded_results(m, s) = numErrorsUncoded / lenCodedOriginal;
            
            % Быстрый выход из цикла при нулевом BER
            if numErrorsCoded == 0 && s > 4
                BER_coded_results(m, s:end) = 0;
                break;
            end
        end
    end

    %% 3. ПОСТРОЕНИЕ СРАВНИТЕЛЬНЫХ ГРАФИКОВ BER vs SNR
    figure('Color', 'w', 'Position', [150, 100, 950, 650]);
    
    plots_for_legend = [];
    legend_labels = {};
    
    for m = 1:length(modTypes)
        valid_idx_coded = BER_coded_results(m, :) > 0;
        valid_idx_uncoded = BER_uncoded_results(m, :) > 0;
        
        % Сплошная линия — Полная цепочка (С кодированием, интерливером и рейт-матчером)
        p_coded = semilogy(EbNo_vec(valid_idx_coded), BER_coded_results(m, valid_idx_coded), ...
            [colors{m} '-'], 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 5);
        hold on;
        
        % Пунктирная линия — Канал (Без кодирования, но с учетом рейт-матчинга)
        semilogy(EbNo_vec(valid_idx_uncoded), BER_uncoded_results(m, valid_idx_uncoded), ... 
            [colors{m} '--'], 'LineWidth', 1.2);
        
        plots_for_legend = [plots_for_legend, p_coded]; %#ok<AGROW>
        
        % Считаем R_eff для красивой подписи легенды
        switch modTypes{m}
            case 'QPSK',   N_i = 10000;
            case '16QAM',  N_i = 10000;
            case '64QAM',  N_i = 10020;
            case '256QAM', N_i = 10000;
        end
        legend_labels = [legend_labels, {sprintf('%s (R_{eff} = %.2f)', modTypes{m}, N_i/targetLength)}]; %#ok<AGROW>
    end
    
    % Маркеры стилей для легенды
    dummy_solid = plot(NaN, NaN, 'k-', 'LineWidth', 2);
    dummy_dashed = plot(NaN, NaN, 'k--', 'LineWidth', 1.2);
    plots_for_legend = [plots_for_legend, dummy_solid, dummy_dashed];
    legend_labels = [legend_labels, {'С кодированием (Viterbi)', 'Без кодирования (Канал)'}];
    
    grid on;
    set(gca, 'YScale', 'log'); 
    ylim([1e-5 1]); 
    xlim([EbNo_vec(1) EbNo_vec(end)]);
    
    title('Кривые BER vs SNR с учетом Rate Matching (Фикс. размер кадра = 24000 бит)');
    xlabel('E_b/N_0 (dB)');
    ylabel('Bit Error Rate (BER)');
    legend(plots_for_legend, legend_labels, 'Location', 'southwest');
    
    fprintf('\nСимуляция трансивера с Rate Matching завершена успешно!\n');
end
