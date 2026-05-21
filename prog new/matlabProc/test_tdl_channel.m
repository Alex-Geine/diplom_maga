function test_tdl_channel()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ
    clear; clc; close all;
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    fprintf('==================================================\n');
    fprintf('   ПОСИМВОЛЬНЫЙ ТЕСТ TDL КАНАЛА И MMSE ЭКВАЛАЙЗЕРА \n');
    fprintf('==================================================\n\n');
    
    % Константы OFDM и канала
    fft_size = 64;
    scs_khz = 15;
    d_km = 600;
    fc_ghz = 2.0;
    shadowing_std_db = 3;
    
    snr_db = 22; 
    snr_lin = 10^(snr_db/10);
    
    modType = '16QAM';
    bps = 4;
    num_ofdm_blocks = 32; % Симулируем 32 OFDM-символа
    N_symbols = fft_size * num_ofdm_blocks;
    
    %% 2. ИМИТАЦИЯ МАТРИЦЫ ИНТЕРФЕРЕНЦИИ (ICI)
    I_test = eye(fft_size) * 0.97;
    for idx = 1:fft_size-1
        I_test(idx, idx+1) = 0.03i;
        I_test(idx+1, idx) = 0.03i;
    end
    
    %% 3. ГЕНЕРАЦИЯ ДАННЫХ И МОДУЛЯЦИЯ
    tx_bits = randi([0 1], N_symbols, bps);
    [tx_signal_all, constellation, ~] = mapper(tx_bits, modType);
    tx_signal_all = tx_signal_all(:);
    
    % Буферы для сбора сквозных результатов
    rx_signal_all = zeros(N_symbols, 1);
    eq_signal_all = zeros(N_symbols, 1);
    
    %% 4. ПОСИМВОЛЬНЫЙ ЦИКЛ ОБРАБОТКИ (OFDM СТАНДАРТ)
    for b = 1:num_ofdm_blocks
        % 4.1 Вырезаем временное окно одного OFDM символа
        idx_range = (b-1)*fft_size + 1 : b*fft_size;
        tx_block = tx_signal_all(idx_range);
        
        % 4.2 Прогон через посимвольный спутниковый канал
        [rx_block, H_freq, N0] = channel_apply(tx_block, 'A', fft_size, ...
            scs_khz, d_km, fc_ghz, shadowing_std_db, I_test, snr_lin);
        
        % 4.3 Прогон через посимвольный MMSE эквалайзер
        eq_block = mmse_equalizer(rx_block, H_freq, N0);
        
        % 4.4 Сохраняем в общие массивы для анализа
        rx_signal_all(idx_range) = rx_block;
        eq_signal_all(idx_range) = eq_block;
    end
    
    %% 5. ВЫВОД КЛЮЧЕВОЙ СТАТИСТИКИ
    fprintf('Энергетический баланс системы:\n');
    fprintf('  RMS амплитуды до канала:     %g\n', rms(tx_signal_all));
    fprintf('  RMS амплитуды в канале:      %g (Затухание 5G NTN)\n', rms(rx_signal_all));
    fprintf('  RMS амплитуды после MMSE:    %g (АРУ восстановило сигнал)\n', rms(eq_signal_all));
    
    %% 6. ОТРИСОВКА ГРАФИКОВ ДЛЯ ДИПЛОМНОЙ РАБОТЫ
    figure('Color', 'w');
    max_val = max(abs([real(tx_signal_all); imag(tx_signal_all)])) * 1.3;
    
    subplot(1,2,1);
    plot(real(rx_signal_all), imag(rx_signal_all), 'm.', 'MarkerSize', 6);
    grid on; axis square;
    xlim([-max_val, max_val]); ylim([-max_val, max_val]);
    title('OFDM поднесущие в канале (Затухание)');
    xlabel('I'); ylabel('Q');
    
    subplot(1,2,2);
    plot(real(eq_signal_all), imag(eq_signal_all), 'g.', 'MarkerSize', 6);
    hold on;
    plot(real(constellation), imag(constellation), 'r+', 'LineWidth', 2, 'MarkerSize', 10);
    grid on; axis square;
    xlim([-max_val, max_val]); ylim([-max_val, max_val]);
    title('OFDM поднесущие после MMSE эквалайзера');
    xlabel('I'); ylabel('Q');
    
    fprintf('\n[УСПЕХ] Посимвольный тест успешно завершен без ошибок размерностей!\n');
    fprintf('==================================================\n');
end
