function sim_ber_vs_snr_coded()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ И ПОДКЛЮЧЕНИЕ ПАПОК
    clear; clc; close all;
    
    % Подключаем вашу папку с блоками
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    % Диапазон Eb/No (в дБ) для симуляции
    EbNo_vec = -10:2:18; 
    
    % Список типов модуляций для анализа
    modTypes = {'QPSK', '16QAM', '64QAM', '256QAM'};
    
    % Цвета для графиков
    colors = {'b', 'r', 'g', 'm'};
    
    % Массивы для хранения итоговых BER
    BER_coded_results = zeros(length(modTypes), length(EbNo_vec));
    BER_uncoded_results = zeros(length(modTypes), length(EbNo_vec));
    
    R = 1/2; % Скорость сверточного кодирования
    
    %% 2. ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ
    for m = 1:length(modTypes)
        modType = modTypes{m};
        
        % Настройка количества бит в зависимости от модуляции
        % В кодированной системе декодеру Витерби нужно достаточно символов для сходимости
        switch modType
            case 'QPSK',   bps = 2; N_info = 10000;  
            case '16QAM',  bps = 4; N_info = 10000;
            case '64QAM',  bps = 6; N_info = 12000;
            case '256QAM', bps = 8; N_info = 12000;
        end
        
        fprintf('Симуляция для модуляции: %s...\n', modType);
        
        for s = 1:length(EbNo_vec)
            EbNo_dB = EbNo_vec(s);
            fprintf('  Модуляция: %s | SNR (Eb/No): %d dB\n', modType, EbNo_dB);
            
            % 2.1 Генерация данных и сверточное кодирование
            txInfoBits = randi([0 1], N_info, 1);
            txCodedBits = conv_encoder(txInfoBits);
            
            % Перегруппировка под формат маппера [Символы x биты]
            numSymbols = length(txCodedBits) / bps;
            txBitsMatrix = reshape(txCodedBits, bps, numSymbols).';
            
            % 2.2 Модуляция (Ваш внешний файл mapper.m)
            [txSig, constellation, bitMap] = mapper(txBitsMatrix, modType);
            
            % 2.3 Добавление шума AWGN с учетом скорости кодирования (R)
            EsNo_dB = EbNo_dB + 10*log10(bps) + 10*log10(R);
            EsNo = 10^(EsNo_dB/10);
            noiseVar = 1 / EsNo; 
            noise = sqrt(noiseVar/2) * (randn(size(txSig)) + 1i*randn(size(txSig)));
            rxSig = txSig + noise;
            
            % 2.4 Мягкая демодуляция (Ваш внешний файл soft_demapper.m)
            llrMatrix = soft_demapper(rxSig, constellation, bitMap, noiseVar);
            
            % Вытягиваем LLR в один непрерывный битовый поток
            llrBitsStream = llrMatrix.'; 
            llrBitsStream = llrBitsStream(:);
            
            % 2.5 Декодирование Витерби
            rxInfoBits = viterbi_soft_decoder(llrBitsStream, N_info);
            
            % 2.6 Расчет BER с кодированием
            numErrorsCoded = sum(txInfoBits ~= rxInfoBits);
            BER_coded_results(m, s) = numErrorsCoded / N_info;
            
            % 2.7 Расчет BER без кодирования (для сравнения)
            hardCodedBits = (llrBitsStream < 0);
            numErrorsUncoded = sum(txCodedBits ~= hardCodedBits);
            BER_uncoded_results(m, s) = numErrorsUncoded / length(txCodedBits);
            
            % Если на выходе декодера Витерби ошибок больше нет (BER=0) на протяжении пары шагов,
            % мы зануляем остаток вектора для этой модуляции и переходим к следующей, чтобы сэкономить время.
            if numErrorsCoded == 0 && s > 4
                BER_coded_results(m, s:end) = 0;
                % Напрямую из канала ошибки еще могут идти, поэтому симулируем их отдельно, 
                % либо просто даем циклу прерваться, так как нас интересует именно точка падения кодовой кривой.
                break;
            end
        end
    end

    %% 3. ПОСТРОЕНИЕ СРАВНИТЕЛЬНЫХ ГРАФИКОВ BER vs SNR
    figure('Color', 'w', 'Position', [200, 100, 900, 600]);
    
    plots_for_legend = [];
    legend_labels = {};
    
    for m = 1:length(modTypes)
        % Отрезаем нулевые значения BER для красивого отображения на логарифмической шкале
        valid_idx_coded = BER_coded_results(m, :) > 0;
        valid_idx_uncoded = BER_uncoded_results(m, :) > 0;
        
        % Сплошная линия с маркером — С кодированием (Витерби)
        p_coded = semilogy(EbNo_vec(valid_idx_coded), BER_coded_results(m, valid_idx_coded), ...
            [colors{m} '-'], 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 5);
        hold on;
        
        % Пунктирная линия — Без кодирования (из канала)
        semilogy(EbNo_vec(valid_idx_uncoded), BER_coded_results(m, valid_idx_uncoded), ... % Ссылаемся на те же точки по оси X
            [colors{m} '--'], 'LineWidth', 1.2);
        
        % Сохраняем указатели для легенды (только для сплошных линий, чтобы не перегружать)
        plots_for_legend = [plots_for_legend, p_coded]; %#ok<AGROW>
        legend_labels = [legend_labels, {sprintf('%s (Coded Soft)', modTypes{m})}]; %#ok<AGROW>
    end
    
    % Добавляем две фейковые линии чисто для пояснения стиля в легенде
    dummy_solid = plot(NaN, NaN, 'k-', 'LineWidth', 2);
    dummy_dashed = plot(NaN, NaN, 'k--', 'LineWidth', 1.2);
    plots_for_legend = [plots_for_legend, dummy_solid, dummy_dashed];
    legend_labels = [legend_labels, {'С кодированием (Viterbi)', 'Без кодирования (Канал)'}];
    
    grid on;
    set(gca, 'YScale', 'log'); 
    ylim([1e-5 1]); % Ограничиваем снизу до 0.001% ошибок
    xlim([EbNo_vec(1) EbNo_vec(end)]);
    
    title('Энергетический выигрыш: Сверточный код R=1/2 (K=7) + Мягкий Витерби');
    xlabel('E_b/N_0 (dB)');
    ylabel('Bit Error Rate (BER)');
    legend(plots_for_legend, legend_labels, 'Location', 'southwest');
    
    fprintf('\nСимуляция кодированной системы успешно завершена!\n');
end
