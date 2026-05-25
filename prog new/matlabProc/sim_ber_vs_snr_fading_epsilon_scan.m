function sim_ber_vs_snr_fading_epsilon_scan()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ И ПОДКЛЮЧЕНИЕ ПАПОК
    clear; clc; close all;
    
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    % Диапазон Eb/No (дБ)
    EbNo_vec = 0:2:24;
    
    % Типы модуляций
    modTypes = {'QPSK', '16QAM', '64QAM', '256QAM'};
    colors = {'b', 'r', 'g', 'm'};
    
    % Параметры OFDM и канала (фиксированы)
    fft_size           = 2048;
    scs_khz            = 240;
    d_km               = 600;
    fc_ghz             = 2.0;
    shadowing_std_db   = 3;
    profile_name       = 'A';
    numRowsInterleaver = 40;
    numOFDMSymbolsPerFrame = 20;
    numSymbolsTotal = fft_size * numOFDMSymbolsPerFrame;
    
    % Значения epsilon для перебора
    epsilon_list = [0, 4e3/3e8, 8e3/3e8];   % 0, 4e3/3e8, 8e3/3e8
    epsilon_labels = {'0', '4 km/s', '8 km/s'}; % для подписей
    
    % Создаём фигуру с тремя подграфиками
    figure('Color', 'w', 'Position', [100, 100, 1200, 900]);
    
    for eps_idx = 1:length(epsilon_list)
        epsilon = epsilon_list(eps_idx);
        fprintf('\n========== Запуск симуляции для epsilon = %.2e ==========\n', epsilon);
        
        % Генерация матрицы ICI для текущего epsilon
        alpha_D = 0;
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
            
            targetLength = numSymbolsTotal * bps;
            desired_rate = 1/2;
            N_info = round(desired_rate * targetLength);
            R_eff = N_info / targetLength;
            fprintf('Симуляция: %s, R_eff = %.3f\n', modType, R_eff);
            
            % Переопределяем numSymbolsTotal и numOFDMSymbols (как в исходном коде)
            numSymbolsTotal_local = targetLength / bps;
            numOFDMSymbols = numSymbolsTotal_local / fft_size;
            
            for s = 1:length(EbNo_vec)
                EbNo_dB = EbNo_vec(s);
                fprintf('  %s | Eb/No: %d dB\n', modType, EbNo_dB);
                
                % --- Передатчик ---
                txInfoBits = randi([0 1], N_info, 1);
                [txMatchedBits, lenCodedOrig, lenInterleavedOrig, txCodedOriginal] = ...
                    fec_tx(txInfoBits, numRowsInterleaver, targetLength);
                
                % --- Модуляция ---
                txSymbolsMatrix = reshape(txMatchedBits, numSymbolsTotal_local, bps);
                [txSigTotal, constellation, bitMap] = mapper(txSymbolsMatrix, modType);
                txSigTotal = txSigTotal(:);
                
                % SNR на символ
                EsNo_dB = EbNo_dB + 10*log10(bps) + 10*log10(R_eff);
                snr_lin = 10^(EsNo_dB/10);
                
                % Буферы
                eqSigTotal = zeros(numSymbolsTotal_local, 1);
                noiseVarTotal = zeros(numSymbolsTotal_local, 1);
                
                for b = 1:numOFDMSymbolsPerFrame
                    idx_range = (b-1)*fft_size + 1 : b*fft_size;
                    tx_ofdm_block = txSigTotal(idx_range);
                    
                    % Канал
                    [rx_ofdm_block, H_freq, N0] = channel_apply(tx_ofdm_block, profile_name, ...
                        fft_size, scs_khz, d_km, fc_ghz, shadowing_std_db, I_matrix, snr_lin);
                    
                    % MMSE
                    [eq_ofdm_block, noiseVar_eq] = mmse_equalizer(rx_ofdm_block, H_freq, N0);
                    
                    eqSigTotal(idx_range) = eq_ofdm_block;
                    noiseVarTotal(idx_range) = noiseVar_eq;
                end
                
                % Мягкая демодуляция
                llrMatrix = soft_demapper(eqSigTotal, constellation, bitMap, noiseVarTotal);
                llrBitsStream = llrMatrix(:);
                
                % FEC декодер
                rxInfoBits = fec_rx(llrBitsStream, lenInterleavedOrig, ...
                    numRowsInterleaver, lenCodedOrig, N_info);
                BER_coded(m, s) = sum(txInfoBits ~= rxInfoBits) / N_info;
                
                % Uncoded BER
                llrAfterRecovery = rate_recovery(llrBitsStream, lenInterleavedOrig);
                llrDeinterleaved = deinterleaver(llrAfterRecovery, numRowsInterleaver, lenCodedOrig);
                hardCodedBits = (llrDeinterleaved < 0);
                BER_uncoded(m, s) = sum(txCodedOriginal ~= hardCodedBits) / lenCodedOrig;
                
                % Досрочный выход (опционально)
                if BER_coded(m, s) == 0 && s > 5
                    BER_coded(m, s+1:end) = 0;
                    BER_uncoded(m, s+1:end) = 0;
                    break;
                end
            end
        end
        
        %% Построение подграфика для текущего epsilon
        subplot(1, 3, eps_idx);
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
        title(sprintf('ε = %s (%.2e)', epsilon_labels{eps_idx}, epsilon));
        legend(modTypes, 'Location', 'southwest');
        
        % Сохраняем результаты для каждого epsilon
        save(sprintf('ntn_results_epsilon_%d.mat', eps_idx), ...
            'EbNo_vec', 'modTypes', 'BER_coded', 'BER_uncoded', 'epsilon');
    end
    
    sgtitle('Влияние доплеровского сдвига (ICI) на BER в канале 5G NTN (TDL-A)');
    fprintf('\nСимуляция для всех epsilon завершена. Графики построены.\n');
end