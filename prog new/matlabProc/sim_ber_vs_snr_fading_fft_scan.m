function sim_ber_vs_snr_fading_fft_scan()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ
    clear; clc; close all;
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    % Диапазон Eb/No (дБ)
    EbNo_vec = 0:2:24;
    
    % Типы модуляций
    modTypes = {'QPSK', '16QAM', '64QAM', '256QAM'};
    colors = {'b', 'r', 'g', 'm'};
    
    % Фиксированные параметры (кроме fft_size)
    scs_khz = 240;
    d_km = 600;
    fc_ghz = 2.0;
    shadowing_std_db = 3;
    profile_name = 'A';
    numRowsInterleaver = 40;
    numOFDMSymbolsPerFrame = 20;   % одинаково для всех fft_size
    
    % Параметры ICI (фиксированный доплер)
    alpha_D = 0;
    epsilon = 8e3 / 3e8;           % 8 км/с
    
    % Список размеров БПФ
    fft_size_list = [512, 1024, 2048];
    
    % Создаём фигуру с тремя подграфиками
    figure('Color', 'w', 'Position', [100, 100, 1400, 500]);
    
    for fft_idx = 1:length(fft_size_list)
        fft_size = fft_size_list(fft_idx);
        fprintf('\n========== Запуск симуляции для FFT = %d ==========\n', fft_size);
        
        % Количество комплексных символов в кадре
        numSymbolsTotal = fft_size * numOFDMSymbolsPerFrame;
        
        % Генерация матрицы ICI для текущего fft_size
        I_matrix = ici_matrix_gen(fft_size, alpha_D, epsilon);
        
        % Массивы для BER
        BER_coded = zeros(length(modTypes), length(EbNo_vec));
        BER_uncoded = zeros(length(modTypes), length(EbNo_vec));
        
        %% Основной цикл по модуляциям
        for m = 1:length(modTypes)
            modType = modTypes{m};
            switch modType
                case 'QPSK',   bps = 2;
                case '16QAM',  bps = 4;
                case '64QAM',  bps = 6;
                case '256QAM', bps = 8;
            end
            
            % Длина кадра в битах после рейт-матчинга
            targetLength = numSymbolsTotal * bps;
            desired_rate = 1/2;
            N_info = round(desired_rate * targetLength);
            R_eff = N_info / targetLength;
            fprintf('Модуляция: %s, R_eff = %.3f\n', modType, R_eff);
            
            % Внутренние переменные (как в исходном коде)
            numSymbolsTotal_local = targetLength / bps;
            numOFDMSymbols = numSymbolsTotal_local / fft_size;  % должно равняться numOFDMSymbolsPerFrame
            
            for s = 1:length(EbNo_vec)
                EbNo_dB = EbNo_vec(s);
                fprintf('  %s | Eb/No: %d dB\n', modType, EbNo_dB);
                
                % --- Передатчик битового уровня ---
                txInfoBits = randi([0 1], N_info, 1);
                [txMatchedBits, lenCodedOrig, lenInterleavedOrig, txCodedOriginal] = ...
                    fec_tx(txInfoBits, numRowsInterleaver, targetLength);
                
                % --- Модуляция (маппер) ---
                txSymbolsMatrix = reshape(txMatchedBits, numSymbolsTotal_local, bps);
                [txSigTotal, constellation, bitMap] = mapper(txSymbolsMatrix, modType);
                txSigTotal = txSigTotal(:);
                
                % --- SNR на символ (Es/No) ---
                EsNo_dB = EbNo_dB + 10*log10(bps) + 10*log10(R_eff);
                snr_lin = 10^(EsNo_dB/10);
                
                % Буферы для принятых символов и дисперсий шума
                eqSigTotal = zeros(numSymbolsTotal_local, 1);
                noiseVarTotal = zeros(numSymbolsTotal_local, 1);
                
                % --- Цикл по OFDM-символам ---
                for b = 1:numOFDMSymbolsPerFrame
                    idx_range = (b-1)*fft_size + 1 : b*fft_size;
                    tx_ofdm_block = txSigTotal(idx_range);
                    
                    % Канал
                    [rx_ofdm_block, H_freq, N0] = channel_apply(tx_ofdm_block, profile_name, ...
                        fft_size, scs_khz, d_km, fc_ghz, shadowing_std_db, I_matrix, snr_lin);
                    
                    % MMSE эквалайзер
                    [eq_ofdm_block, noiseVar_eq] = mmse_equalizer(rx_ofdm_block, H_freq, N0);
                    
                    eqSigTotal(idx_range) = eq_ofdm_block;
                    noiseVarTotal(idx_range) = noiseVar_eq;
                end
                
                % --- Мягкая демодуляция ---
                llrMatrix = soft_demapper(eqSigTotal, constellation, bitMap, noiseVarTotal);
                llrBitsStream = llrMatrix(:);
                
                % --- FEC декодер ---
                rxInfoBits = fec_rx(llrBitsStream, lenInterleavedOrig, ...
                    numRowsInterleaver, lenCodedOrig, N_info);
                BER_coded(m, s) = sum(txInfoBits ~= rxInfoBits) / N_info;
                
                % --- BER без кодирования ---
                llrAfterRecovery = rate_recovery(llrBitsStream, lenInterleavedOrig);
                llrDeinterleaved = deinterleaver(llrAfterRecovery, numRowsInterleaver, lenCodedOrig);
                hardCodedBits = (llrDeinterleaved < 0);
                BER_uncoded(m, s) = sum(txCodedOriginal ~= hardCodedBits) / lenCodedOrig;
                
                % Досрочный выход при нулевых ошибках (опционально)
                if BER_coded(m, s) == 0 && s > 5
                    BER_coded(m, s+1:end) = 0;
                    BER_uncoded(m, s+1:end) = 0;
                    break;
                end
            end
        end
        
        %% Построение подграфика для текущего fft_size
        subplot(1, 3, fft_idx);
        hold on;
        for m = 1:length(modTypes)
            valid_coded = BER_coded(m, :) > 0;
            valid_uncoded = BER_uncoded(m, :) > 0;
            
            semilogy(EbNo_vec(valid_coded), BER_coded(m, valid_coded), ...
                [colors{m} '-o'], 'LineWidth', 2, 'MarkerSize', 4);
            semilogy(EbNo_vec(valid_uncoded), BER_uncoded(m, valid_uncoded), ...
                [colors{m} '--'], 'LineWidth', 1.2);
        end
        grid on;
        set(gca, 'YScale', 'log', 'YLim', [1e-5 1], 'XLim', [EbNo_vec(1) EbNo_vec(end)]);
        xlabel('E_b/N_0 (dB)');
        ylabel('BER');
        title(sprintf('FFT size = %d', fft_size));
        legend(modTypes, 'Location', 'southwest');
        
        % Сохраняем результаты для каждого fft_size
        save(sprintf('ntn_fft_%d_results.mat', fft_size), ...
            'EbNo_vec', 'modTypes', 'BER_coded', 'BER_uncoded', 'fft_size');
    end
    
    sgtitle('Влияние размера БПФ на BER в канале 5G NTN (TDL-A, ICI = 8 км/с)');
    fprintf('\nСимуляция для всех FFT размеров завершена.\n');
end