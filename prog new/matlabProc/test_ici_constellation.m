function test_ici_constellation()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ
    clear; clc; close all;
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    % Параметры OFDM и ICI матрицы
    fft_size = 64;
    alpha_D = 0.15;     % Нормированный Доплер (15% от SCS)
    epsilon = 1e-4;     % Уход частоты дискретизации
    
    snr_db = 22;        % Уровень шума в канале
    snr_lin = 10^(snr_db/10);
    
    modType = '16QAM';
    bps = 4;
    num_blocks = 20;    % Генерируем 20 OFDM символов
    N_symbols = fft_size * num_blocks;
    
    %% 2. ГЕНЕРАЦИЯ СИГНАЛА И МАТРИЦЫ ICI
    I_matrix = ici_matrix_gen(fft_size, alpha_D, epsilon);
    
    tx_bits = randi([0 1], N_symbols, bps);
    [tx_sig, constellation, ~] = mapper(tx_bits, modType);
    tx_sig = tx_sig(:);
    
    %% 3. ПОБЛОЧНОЕ НАЛОЖЕНИЕ ICI И ШУМА
    txMatrix = reshape(tx_sig, fft_size, num_blocks);
    rxMatrix_pure_ici = zeros(size(txMatrix));
    rxMatrix_with_noise = zeros(size(txMatrix));
    
    for b = 1:num_blocks
        X_block = txMatrix(:, b).'; % Строка [1 x fft_size]
        
        % Вариант А: Чистая интерференция ICI (без шума)
        Y_pure = X_block * I_matrix;
        rxMatrix_pure_ici(:, b) = Y_pure.';
        
        % Вариант Б: Интерференция ICI + AWGN Шум
        P_signal = mean(abs(Y_pure).^2);
        N0 = P_signal / snr_lin;
        noise = sqrt(N0/2) * (randn(size(Y_pure)) + 1i*randn(size(Y_pure)));
        rxMatrix_with_noise(:, b) = (Y_pure + noise).';
    end
    
    rx_pure_ici = rxMatrix_pure_ici(:);
    rx_with_noise = rxMatrix_with_noise(:);
    
    %% 4. ОТРИСОВКА ГРАФИКОВ ДЛЯ ДИПЛОМА
    figure('Color', 'w');
    max_val = max(abs([real(tx_sig); imag(tx_sig)])) * 1.3;
    
    % Левая панель: только ICI (характерное "закручивание" точек по кругу)
    subplot(1,2,1);
    plot(real(rx_pure_ici), imag(rx_pure_ici), 'm.', 'MarkerSize', 6);
    hold on;
    plot(real(constellation), imag(constellation), 'r+', 'LineWidth', 2, 'MarkerSize', 10);
    grid on; axis square;
    xlim([-max_val, max_val]); ylim([-max_val, max_val]);
    title({sprintf('Влияние ICI матрицы (Без шума)'), sprintf('\\alpha_D = %.2f, \\epsilon = %g', alpha_D, epsilon)});
    xlabel('In-Phase (I)'); ylabel('Quadrature (Q)');
    
    % Правая панель: ICI + Шум (размытие созвездия в "облака")
    subplot(1,2,2);
    plot(real(rx_with_noise), imag(rx_with_noise), 'g.', 'MarkerSize', 6);
    hold on;
    plot(real(constellation), imag(constellation), 'r+', 'LineWidth', 2, 'MarkerSize', 10);
    grid on; axis square;
    xlim([-max_val, max_val]); ylim([-max_val, max_val]);
    title({sprintf('Эффект ICI + AWGN Шум (%.1f дБ)', snr_db)});
    xlabel('In-Phase (I)'); ylabel('Quadrature (Q)');
    
    fprintf('Тест созвездий успешно завершен. Посмотрите на графики эффекта ICI.\n');
end
