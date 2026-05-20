function modulo_ber()
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ
    clear; clc; close all;
    
    % Диапазон Eb/No (в дБ) для симуляции
    EbNo_vec = 0:2:24; 
    
    % Список типов модуляций для анализа
    modTypes = {'QPSK', '16QAM', '64QAM', '256QAM'};
    
    % Цвета и маркеры для графиков
    styles = { 'b-o', 'r-s', 'g-^', 'm-d' };
    
    % Массив для хранения итоговых BER
    BER_results = zeros(length(modTypes), length(EbNo_vec));
    
    %% 2. ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ
    for m = 1:length(modTypes)
        modType = modTypes{m};
        
        switch modType
            case 'QPSK',   bps = 2; N = 20000;  % Больше символов для низких BER
            case '16QAM',  bps = 4; N = 20000;
            case '64QAM',  bps = 6; N = 15000;
            case '256QAM', bps = 8; N = 10000;
        end
        
        fprintf('Симуляция для модуляции: %s...\n', modType);
        
        for s = 1:length(EbNo_vec)
            EbNo_dB = EbNo_vec(s);
            
            % Генерация случайных бит
            txBits = randi([0 1], N, bps);
            
            % Модуляция
            [txSig, constellation, bitMap] = mapper(txBits, modType);
            
            % Расчет шума
            EsNo_dB = EbNo_dB + 10*log10(bps);
            EsNo = 10^(EsNo_dB/10);
            noiseVar = 1 / EsNo; 
            noise = sqrt(noiseVar/2) * (randn(size(txSig)) + 1i*randn(size(txSig)));
            rxSig = txSig + noise;
            
            % Мягкая демодуляция
            llr = soft_demapper(rxSig, constellation, bitMap, noiseVar);
            
            % Жесткое решение на основе LLR
            hardBits = (llr < 0); 
            
            % Расчет BER
            numErrors = sum(txBits(:) ~= hardBits(:));
            BER_results(m, s) = numErrors / (N * bps);
            
            % Прерываем симуляцию для данной модуляции, если ошибок вообще нет (BER=0)
            % Это экономит время на высоких SNR и красиво обрывает график вниз
            if numErrors == 0 && s > 3
                BER_results(m, s:end) = 0;
                break;
            end
        end
    end

    %% 3. ПОСТРОЕНИЕ ГРАФИКОВ BER vs SNR
    figure('Color', 'w', 'Position', [100, 100, 800, 500]);
    
    for m = 1:length(modTypes)
        % Используем semilogy для логарифмической шкалы по оси Y
        semilogy(EbNo_vec, BER_results(m, :), styles{m}, 'LineWidth', 1.5, 'MarkerSize', 6);
        hold on;
    end
    
    grid on;
    set(gca, 'YScale', 'log'); % Убеждаемся, что шкала логарифмическая
    ylim([1e-5 1]);            % Ограничение снизу для наглядности (до 0.001%)
    xlim([EbNo_vec(1) EbNo_vec(end)]);
    
    title('Кривые BER vs SNR для различных типов модуляции');
    xlabel('E_b/N_0 (dB)');
    ylabel('Bit Error Rate (BER)');
    legend(modTypes, 'Location', 'southwest');
    
    fprintf('\nСимуляция успешно завершена!\n');
end

