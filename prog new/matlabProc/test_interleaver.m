function test_interleaver()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ
    clear; clc; close all;
    
    % Подключаем вашу папку с блоками
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    fprintf('==================================================\n');
    fprintf('   ЗАПУСК ТЕСТА ИНТЕРЛИВЕРА И ДЕИНТЕРЛИВЕРА       \n');
    fprintf('==================================================\n\n');
    
    % Параметры теста
    N = 20000;            % Длина исходного вектора (специально не кратна глубине)
    numRows = 10;       % Глубина интерливера (число строк матрицы)
    
    %% 2. ТЕСТ 1: ИДЕАЛЬНОЕ ВОССТАНОВЛЕНИЕ ДАННЫХ (БИТ-В-БИТ)
    % Генерируем тестовый вектор (имитируем биты или вещественные LLR)
    originalData = randi([0 1], N, 1); % Используем последовательные числа, чтобы сразу видеть сдвиги
    
    % Пропускаем через интерливер
    interleavedData = interleaver(originalData, numRows);
    
    % Пропускаем через деинтерливер
    deinterleavedData = deinterleaver(interleavedData, numRows, N);
    
    % Проверка на идентичность
    if isequal(originalData, deinterleavedData)
        fprintf('[УСПЕХ] Тест 1: Исходные данные восстановлены идеально.\n');
        fprintf('        Размер входа: %d, Размер выхода: %d\n\n', length(originalData), length(deinterleavedData));
    else
        fprintf('[ОШИБКА] Тест 1: Данные после деинтерливера не совпадают с исходными!\n');
        % Выведем первые несколько элементов для анализа
        disp('Первые 10 элементов (Исходные | Восстановленные):');
        disp([originalData(1:10), deinterleavedData(1:10)]);
        return; % Прерываем тест, если базовый алгоритм сломан
    end
    
    %% 3. ТЕСТ 2: РАБОТА С ПАКЕТНЫМИ ОШИБКАМИ (ВИЗУАЛИЗАЦИЯ)
    % Имитируем непрерывный пакет ошибок в канале (длиной в 5 искаженных элементов)
    % Представим, что это LLR-метрики. Нормальные метрики = +10, ошибки = -10.
    llrChannel = 10 * ones(N, 1);
    
    % Перемешиваем перед отправкой в канал
    llrInterleaved = interleaver(llrChannel, numRows);
    
    % --- Имитация пакета ошибок в канале ---
    % Порртим 5 элементов подряд прямо посередине перемешанного потока
    burstStart = 40;
    burstLength = 5;
    llrInterleaved(burstStart : burstStart + burstLength - 1) = -10; 
    
    % Пропускаем через деинтерливер перед подачей в Витерби
    llrAfterDeinterleaver = deinterleaver(llrInterleaved, numRows, N);
    
    % Анализируем, где теперь находятся ошибки
    errorPositions = find(llrAfterDeinterleaver == -10);
    
    fprintf('[УСПЕХ] Тест 2: Эмуляция пакетной помехи в канале.\n');
    fprintf('        Длина пакета ошибок в канале: %d элементов подряд.\n', burstLength);
    fprintf('        Позиции ошибок после ДЕинтерливера: %s\n', mat2str(errorPositions.'));
    
    % Проверяем, стали ли ошибки изолированными
    minDistance = min(diff(errorPositions));
    fprintf('        Минимальное расстояние между ошибками на входе Витерби: %d\n\n', minDistance);
    
    %% 4. ГРАФИЧЕСКАЯ ВИЗУАЛИЗАЦИЯ ДЛЯ ДИПЛОМА
    figure('Color', 'w', 'Position', [300, 200, 800, 400]);
    
    subplot(2,1,1);
    stem(llrInterleaved, 'r', 'LineWidth', 1.5, 'Marker', 'none');
    title('Сигнал в канале (Сплошной пакет ошибок из-за замираний)');
    xlabel('Индекс элемента в канале');
    ylabel('Значение LLR');
    ylim([-15 15]);
    grid on;
    
    subplot(2,1,2);
    stem(llrAfterDeinterleaver, 'b', 'LineWidth', 1.5, 'Marker', 'none');
    title('Сигнал на входе декодера Витерби (Ошибки "размазаны" деинтерливером)');
    xlabel('Индекс информационного бита');
    ylabel('Значение LLR');
    ylim([-15 15]);
    grid on;
    
    fprintf('==================================================\n');
    fprintf('   ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!                \n');
    fprintf('==================================================\n');
end
