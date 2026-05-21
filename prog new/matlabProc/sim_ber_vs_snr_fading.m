function sim_ber_vs_snr_fading()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ И ПОДКЛЮЧЕНИЕ ПАПОК
    clear; clc; close all;
    
    % Подключаем вашу папку с блоками
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    % Диапазон Eb/No (в дБ) для симуляции в канале с замираниями и ICI
    % Спутниковые каналы с доплеровским размытием требуют более высокого SNR
    EbNo_vec = 0:2:24; 
    
    % Список типов модуляций для анализа
    modTypes = {'QPSK', '16QAM', '64QAM', '256QAM'};
    colors = {'b', 'r', 'g', 'm'};
    
    % Параметры OFDM и 5G NTN канала
    fft_size = 2048;          % Количество поднесущих в одном OFDM символе
    scs_khz = 15;           % Разнос поднесущих
    d_km = 600;             % Расстояние до спутника LEO
    fc_ghz = 2.0;           % Частота S-band
    shadowing_std_db = 3;   % СКО логнормального затенения
    profile_name = 'A';     % Профиль TDL-A из 3GPP TR 38.901
    numRowsInterleaver = 40;% Глубина интерливера
    
    % Жестко задаем размер физического кадра в эфире (в битах)
    targetLength = 24000; 
    
    % Массивы для хранения результатов BER
    BER_coded_results = zeros(length(modTypes), length(EbNo_vec));
    BER_uncoded_results = zeros(length(modTypes), length(EbNo_vec));
    
    %% 2. ИМИТАЦИЯ МАТРИЦЫ ИНТЕРФЕРЕНЦИИ ПОДНЕСУЩИХ (ICI)
    % Моделируем фиксированное частотное размытие из-за Доплера
    alpha_D = 0;
    epsilon = 8e3 / 3e8;
    I_matrix = ici_matrix_gen(fft_size, alpha_D, epsilon);
    %I_matrix = eye(fft_size); %* 0.97;
    %for idx = 1:fft_size-1
    %    I_matrix(idx, idx+1) = 0.03i;
    %    I_matrix(idx+1, idx) = 0.03i;
    %end
    
    %% 3. ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ ПО МОДУЛЯЦИЯМ
    for m = 1:length(modTypes)
        modType = modTypes{m};
        
        % Настройка количества информационных бит
        switch modType
            case 'QPSK',   bps = 2; N_info = 10000;  
            case '16QAM',  bps = 4; N_info = 10000;
            case '64QAM',  bps = 6; N_info = 10020; % Кратность для интерливера
            case '256QAM', bps = 8; N_info = 10000;
        end
        
        % Эффективная скорость кода с учетом рейт-матчера
        R_eff = N_info / targetLength;
        fprintf('Симуляция: %s в TDL-%s канале NTN (R_eff = %.3f)...\n', modType, profile_name, R_eff);
        
        % Вычисляем общее число OFDM-символов в одном кадре данных
        numSymbolsTotal = targetLength / bps;
        numOFDMSymbols = numSymbolsTotal / fft_size;
        
        for s = 1:length(EbNo_vec)
            EbNo_dB = EbNo_vec(s);
            fprintf('  %s | Eb/No: %d dB\n', modType, EbNo_dB);
            
            % 3.1 ПЕРЕДАТЧИК БИТОВОГО УРОВНЯ (FEC TX)
            txInfoBits = randi([0 1], N_info, 1);
            [txMatchedBits, lenCodedOrig, lenInterleavedOrig] = ...
                fec_tx(txInfoBits, numRowsInterleaver, targetLength);
            
            % 3.2 МОДУЛЯЦИЯ (Маппер)
            txSymbolsMatrix = reshape(txMatchedBits, numSymbolsTotal, bps);
            [txSigTotal, constellation, bitMap] = mapper(txSymbolsMatrix, modType);
            txSigTotal = txSigTotal(:); % Гарантируем вектор-столбец
            
            % Вычисление линейного SNR на символ (Es/No) с учетом избыточности кода
            EsNo_dB = EbNo_dB + 10*log10(bps) + 10*log10(R_eff);
            snr_lin = 10^(EsNo_dB/10);
            
            % Выделяем буфер под эквализованный сквозной поток символов кадра
            eqSigTotal = zeros(numSymbolsTotal, 1);
            
            % Справочный буфер для LLR канала (без FEC)
            % Нам понадобится рассчитать "чистый" шум канала для демаппера
            % На посимвольном уровне он плывет, усредним его для битового уровня
            mean_noiseVar = 0;
            
            %% 3.3 ПОСИМВОЛЬНЫЙ OFDM ЦИКЛ ОБРАБОТКИ (СИМВОЛЬНЫЙ УРОВЕНЬ)
            for b = 1:numOFDMSymbols
                % Вырезаем временное окно ровно ОДНОГО OFDM-символа
                idx_range = (b-1)*fft_size + 1 : b*fft_size;
                tx_ofdm_block = txSigTotal(idx_range);
                
                % Прогон через посимвольный спутниковый TDL канал
                [rx_ofdm_block, H_freq, N0] = channel_apply(tx_ofdm_block, profile_name, ...
                    fft_size, scs_khz, d_km, fc_ghz, shadowing_std_db, I_matrix, snr_lin);
                
                % Восстановление фазы, ICI и АРУ через посимвольный MMSE-эквалайзер
                eq_ofdm_block = mmse_equalizer(rx_ofdm_block, H_freq, N0);
                
                % Записываем восстановленный блок в общий приемный буфер кадра
                eqSigTotal(idx_range) = eq_ofdm_block;
                mean_noiseVar = mean_noiseVar + N0;
            end
            mean_noiseVar = mean_noiseVar / numOFDMSymbols;
            
            %% 3.4 МЯГКАЯ ДЕМОДУЛЯЦИЯ (Демаппер)
            % Демаппируем чистый эквализованный поток без передачи H (как доказал тест)
            % Используем среднее значение дисперсии шума в качестве опорного
            llrMatrix = soft_demapper(eqSigTotal, constellation, bitMap, mean_noiseVar);
            llrBitsStream = llrMatrix(:); % Вытягиваем строго по столбцам
            
            % 3.5 ПРИЕМНИК БИТОВОГО УРОВНЯ (FEC RX)
            rxInfoBits = fec_rx(llrBitsStream, lenInterleavedOrig, ...
                                numRowsInterleaver, lenCodedOrig, N_info);
            
            % 3.6 Расчет BER с кодированием (Полный тракт FEC + MMSE)
            numErrorsCoded = sum(txInfoBits ~= rxInfoBits);
            BER_coded_results(m, s) = numErrorsCoded / N_info;
            
            % 3.7 Расчет BER без кодирования (Только посимвольный MMSE в канале)
            llrAfterRecovery = rate_recovery(llrBitsStream, lenInterleavedOrig);
            llrDeinterleaved = deinterleaver(llrAfterRecovery, numRowsInterleaver, lenCodedOrig);
            hardCodedBits = (llrDeinterleaved < 0);
            
            txCodedBits_check = conv_encoder(txInfoBits);
            numErrorsUncoded = sum(txCodedBits_check ~= hardCodedBits);
            BER_uncoded_results(m, s) = numErrorsUncoded / length(txCodedBits_check);
            
            % Быстрый выход из SNR-цикла, если ошибок после FEC больше нет
            if numErrorsCoded == 0 && s > 5
                BER_coded_results(m, s:end) = 0;
                break;
            end
        end
    end

    %% 4. ПОСТРОЕНИЕ СРАВНИТЕЛЬНЫХ ГРАФИКОВ BER vs SNR ДЛЯ ДИПЛОМА
    figure('Color', 'w', 'Position', [150, 100, 950, 650]);
    
    plots_for_legend = [];
    legend_labels = {};
    
    for m = 1:length(modTypes)
        valid_idx_coded = BER_coded_results(m, :) > 0;
        valid_idx_uncoded = BER_uncoded_results(m, :) > 0;
        
        % Сплошная линия с маркером — Полный тракт (FEC + Посимвольный MMSE)
        p_coded = semilogy(EbNo_vec(valid_idx_coded), BER_coded_results(m, valid_idx_coded), ...
            [colors{m} '-'], 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 5);
        hold on;
        
        % Пунктирная линия — Только посимвольный MMSE (Без кодирования)
        semilogy(EbNo_vec(valid_idx_uncoded), BER_uncoded_results(m, valid_idx_uncoded), ... 
            [colors{m} '--'], 'LineWidth', 1.2);
        
        plots_for_legend = [plots_for_legend, p_coded]; %#ok<AGROW>
        legend_labels = [legend_labels, {sprintf('%s (FEC + MMSE)', modTypes{m})}]; %#ok<AGROW>
    end
    
    % Добавление служебных линий стиля в легенду
    dummy_solid = plot(NaN, NaN, 'k-', 'LineWidth', 2);
    dummy_dashed = plot(NaN, NaN, 'k--', 'LineWidth', 1.2);
    plots_for_legend = [plots_for_legend, dummy_solid, dummy_dashed];
    legend_labels = [legend_labels, {'Полный тракт (FEC + MMSE)', 'Без кодирования (Только MMSE)'}];
    
    grid on;
    set(gca, 'YScale', 'log'); 
    ylim([1e-5 1]); 
    xlim([EbNo_vec(1) EbNo_vec(end)]);
    
    title(sprintf('Кривые BER vs SNR в спутниковом канале 5G NTN (Профиль TDL-%s + ICI)', profile_name));
    xlabel('E_b/N_0 (dB)');
    ylabel('Bit Error Rate (BER)');
    legend(plots_for_legend, legend_labels, 'Location', 'southwest');
    
    % Сохраняем результаты в .mat файл для таблиц
    save('ntn_simulation_results.mat', 'EbNo_vec', 'modTypes', 'BER_coded_results', 'BER_uncoded_results');
    fprintf('\nСимуляция успешно завершена! Данные сохранены в ntn_simulation_results.mat\n');
end
