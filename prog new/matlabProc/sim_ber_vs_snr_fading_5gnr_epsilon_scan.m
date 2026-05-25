function sim_ber_vs_snr_fading_5gnr_epsilon_scan()
    clear; clc; close all;
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    % === 5G NR NTN параметры (фиксированные) ===
    band = 'L_S';           % 'L_S' или 'Ka'
    bw_mhz = 20;            % полоса в МГц
    scs_khz = 15;           % разнос поднесущих (кГц)
    d_km = 600;             % расстояние до спутника (км)
    fc_ghz = 2.0;           % несущая частота (ГГц)
    shadowing_std_db = 3;
    profile_name = 'A';
    numRowsInterleaver = 40;
    
    % Параметры OFDM (зависят от band, bw, scs)
    numRB = get_nr_rb(band, bw_mhz, scs_khz);
    active_subcarriers_per_symbol = numRB * 12;
    [fft_size, ~, offset] = get_ofdm_params(active_subcarriers_per_symbol);
    numOFDMSymbolsPerFrame = 20;
    numSymbolsTotal = active_subcarriers_per_symbol * numOFDMSymbolsPerFrame;
    
    % Диапазон Eb/No и типы модуляций
    EbNo_vec = -4:2:24;
    modTypes = {'QPSK', '16QAM', '64QAM', '256QAM'};
    colors = {'b','r','g','m'};
    
    % Значения epsilon для перебора
    epsilon_list = [0, 4e3/3e8, 8e3/3e8];
    epsilon_labels = {'ε = 0', 'ε = 4 км/с', 'ε = 8 км/с'};
    
    % Создаём фигуру с тремя подграфиками
    figure('Color', 'w', 'Position', [100, 100, 1400, 500]);
    
    for eps_idx = 1:length(epsilon_list)
        epsilon = epsilon_list(eps_idx);
        alpha_D = 0;
        I_matrix = ici_matrix_gen(fft_size, alpha_D, epsilon);
        
        fprintf('\n========== Симуляция для epsilon = %.2e ==========\n', epsilon);
        
        % Массивы для BER
        BER_coded = zeros(length(modTypes), length(EbNo_vec));
        BER_uncoded = zeros(length(modTypes), length(EbNo_vec));
        
        % Цикл по модуляциям
        for m = 1:length(modTypes)
            modType = modTypes{m};
            switch modType
                case 'QPSK', bps = 2;
                case '16QAM', bps = 4;
                case '64QAM', bps = 6;
                case '256QAM', bps = 8;
            end
            
            targetLength = numSymbolsTotal * bps;
            desired_rate = 1/2;
            N_info = round(desired_rate * targetLength);
            R_eff = N_info / targetLength;
            fprintf('  %s: скорость кода = %.3f\n', modType, R_eff);
            
            for s = 1:length(EbNo_vec)
                EbNo_dB = EbNo_vec(s);
                % fprintf('    Eb/No = %d dB\n', EbNo_dB);
                
                % --- Передатчик ---
                txInfoBits = randi([0 1], N_info, 1);
                [txMatchedBits, lenCodedOrig, lenInterleavedOrig, txCodedOriginal] = ...
                    fec_tx(txInfoBits, numRowsInterleaver, targetLength);
                
                % --- Модуляция ---
                txSymbolsMatrix = reshape(txMatchedBits, numSymbolsTotal, bps);
                [txSigActiveTotal, constellation, bitMap] = mapper(txSymbolsMatrix, modType);
                txSigActiveTotal = txSigActiveTotal(:);
                
                % SNR на символ
                EsNo_dB = EbNo_dB + 10*log10(bps) + 10*log10(R_eff);
                snr_lin = 10^(EsNo_dB/10);
                
                % Буферы
                eqSigTotal = zeros(numSymbolsTotal, 1);
                noiseVarTotal = zeros(numSymbolsTotal, 1);
                
                % --- OFDM обработка ---
                for b = 1:numOFDMSymbolsPerFrame
                    idx_active = (b-1)*active_subcarriers_per_symbol + 1 : b*active_subcarriers_per_symbol;
                    symbols_active = txSigActiveTotal(idx_active);
                    
                    tx_ofdm_block = ofdm_modulate(symbols_active, fft_size, offset);
                    
                    [rx_ofdm_block, H_freq_full, N0] = channel_apply(tx_ofdm_block, profile_name, ...
                        fft_size, scs_khz, d_km, fc_ghz, shadowing_std_db, I_matrix, snr_lin);
                    
                    [eq_full, noiseVar_eq_full] = mmse_equalizer(rx_ofdm_block, H_freq_full, N0);
                    
                    [eq_active, ~] = ofdm_demodulate(eq_full, H_freq_full, fft_size, offset, active_subcarriers_per_symbol);
                    
                    eqSigTotal(idx_active) = eq_active;
                    noiseVarTotal(idx_active) = noiseVar_eq_full(offset+1 : offset+active_subcarriers_per_symbol);
                end
                
                % Мягкая демодуляция
                llrMatrix = soft_demapper(eqSigTotal, constellation, bitMap, noiseVarTotal);
                llrBitsStream = llrMatrix(:);
                
                % FEC декодер
                rxInfoBits = fec_rx(llrBitsStream, lenInterleavedOrig, numRowsInterleaver, lenCodedOrig, N_info);
                BER_coded(m, s) = sum(txInfoBits ~= rxInfoBits) / N_info;
                
                % Uncoded BER
                llrAfterRecovery = rate_recovery(llrBitsStream, lenInterleavedOrig);
                llrDeinterleaved = deinterleaver(llrAfterRecovery, numRowsInterleaver, lenCodedOrig);
                hardCodedBits = (llrDeinterleaved < 0);
                BER_uncoded(m, s) = sum(txCodedOriginal ~= hardCodedBits) / lenCodedOrig;
                
                if BER_coded(m, s) == 0 && s > 5
                    BER_coded(m, s+1:end) = 0;
                    BER_uncoded(m, s+1:end) = 0;
                    break;
                end
            end
        end
        
        % --- Построение подграфика для текущего epsilon ---
        subplot(1, 3, eps_idx);
        hold on;
        for m = 1:length(modTypes)
            semilogy(EbNo_vec, BER_coded(m,:), [colors{m} '-o'], 'LineWidth', 2);
            semilogy(EbNo_vec, BER_uncoded(m,:), [colors{m} '--'], 'LineWidth', 1);
        end
        grid on;
        xlabel('E_b/N_0 (dB)');
        ylabel('BER');
        title(epsilon_labels{eps_idx});
        set(gca, 'YScale', 'log', 'YLim', [1e-5 1]);
        if eps_idx == 1
            legend('QPSK FEC','QPSK uncoded','16QAM FEC','16QAM uncoded',...
                   '64QAM FEC','64QAM uncoded','256QAM FEC','256QAM uncoded',...
                   'Location', 'southwest');
        end
    end
    
    sgtitle(sprintf('5G NR NTN (%s, BW=%d MHz, SCS=%d kHz, %d RB) - влияние доплеровского сдвига', ...
        band, bw_mhz, scs_khz, numRB));
    
    % Сохраняем итоговую фигуру и данные (опционально)
    saveas(gcf, 'ntn_5gnr_epsilon_scan.png');
    fprintf('\nСимуляция для всех epsilon завершена. График сохранён.\n');
end

function RB_count = get_nr_rb(band, bw_mhz, scs_khz)
    % Таблицы Resource Blocks из 3GPP TS 38.101-2 (для NTN)
    % band: 'L_S' или 'Ka'
    % bw_mhz: полоса в МГц
    % scs_khz: разнос поднесущих в кГц
    if strcmpi(band, 'L_S')
        % L/S диапазон (до 20 МГц, SCS 15 или 30 кГц)
        switch bw_mhz
            case 5
                if scs_khz == 15, RB_count = 25;
                elseif scs_khz == 30, RB_count = 11;
                else error('SCS %d кГц не поддерживается для L_S 5 МГц', scs_khz);
                end
            case 10
                if scs_khz == 15, RB_count = 52;
                elseif scs_khz == 30, RB_count = 24;
                else error('SCS %d кГц не поддерживается для L_S 10 МГц', scs_khz);
                end
            case 15
                if scs_khz == 15, RB_count = 79;
                elseif scs_khz == 30, RB_count = 38;
                else error('SCS %d кГц не поддерживается для L_S 15 МГц', scs_khz);
                end
            case 20
                if scs_khz == 15, RB_count = 106;
                elseif scs_khz == 30, RB_count = 51;
                else error('SCS %d кГц не поддерживается для L_S 20 МГц', scs_khz);
                end
            otherwise
                error('Неподдерживаемая полоса %d МГц для L_S', bw_mhz);
        end
    elseif strcmpi(band, 'Ka')
        % Ka диапазон (большие полосы, SCS 60 или 120 кГц)
        switch bw_mhz
            case 50
                if scs_khz == 60, RB_count = 66;
                elseif scs_khz == 120, RB_count = 32;
                else error('SCS %d кГц не поддерживается для Ka 50 МГц', scs_khz);
                end
            case 100
                if scs_khz == 60, RB_count = 132;
                elseif scs_khz == 120, RB_count = 66;
                else error('SCS %d кГц не поддерживается для Ka 100 МГц', scs_khz);
                end
            case 200
                if scs_khz == 60, RB_count = 264;
                elseif scs_khz == 120, RB_count = 132;
                else error('SCS %d кГц не поддерживается для Ka 200 МГц', scs_khz);
                end
            case 400
                if scs_khz == 120, RB_count = 264;
                else error('SCS %d кГц не поддерживается для Ka 400 МГц (только 120 кГц)', scs_khz);
                end
            otherwise
                error('Неподдерживаемая полоса %d МГц для Ka', bw_mhz);
        end
    else
        error('Неизвестный диапазон. Используйте ''L_S'' или ''Ka''');
    end
end

function [fft_size, active_subcarriers, offset] = get_ofdm_params(num_active_subcarriers)
    % Подбирает минимальный размер БПФ (степень 2) и смещение для центрирования
    % num_active_subcarriers - количество активных поднесущих (RB*12)
    fft_size = 64;
    while fft_size < num_active_subcarriers
        fft_size = fft_size * 2;
    end
    % Размещаем активные поднесущие симметрично, чтобы DC-поднесущая (индекс fft_size/2+1) попала в защиту
    offset = floor((fft_size - num_active_subcarriers) / 2);
    active_subcarriers = num_active_subcarriers; % для ясности
end

function tx_ofdm_block = ofdm_modulate(symbols_active, fft_size, offset)
    % symbols_active - вектор комплексных символов для активных поднесущих
    % Возвращает временной OFDM-символ (длины fft_size)
    tx_freq = zeros(fft_size, 1);
    tx_freq(offset+1 : offset+length(symbols_active)) = symbols_active;
    % IFFT с нормировкой (чтобы мощность во временной области сохранялась)
    tx_ofdm_block = tx_freq;
end

function [rx_active, H_active] = ofdm_demodulate(rx_time, H_freq_full, fft_size, offset, active_count)
    % rx_time - принятый временной сигнал (длины fft_size)
    % H_freq_full - частотная характеристика канала (все поднесущие)
    % Возвращает активные поднесущие после FFT и соответствующий H_freq
    rx_freq = rx_time; %fftshift(fft(rx_time)) / sqrt(fft_size);
    rx_active = rx_freq(offset+1 : offset+active_count);
    H_active = H_freq_full(offset+1 : offset+active_count);
end