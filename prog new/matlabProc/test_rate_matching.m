function test_rate_matching()
    %% 1. ИСХОДНЫЕ НАСТРОЙКИ
    clear; clc; close all;
    
    % Подключаем вашу папку с блоками
    addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');
    
    fprintf('==================================================\n');
    fprintf('     ЗАПУСК ТЕСТА RATE MATCHER & RECOVERY         \n');
    fprintf('==================================================\n\n');
    
    % Базовые тестовые данные
    originalLength = 10;
    txCodedBits = [1; 0; 1; 1; 0; 0; 1; 0; 1; 1]; % Имитируем выход кодера
    
    %% 2. ТЕСТ 1: РЕЖИМ ВЫКАЛЫВАНИЯ (PUNCTURE)
    targetLength_punc = 7; % Требуется урезать поток
    
    % Выполняем рейт-матчинг (сокращение)
    matchedBits_punc = rate_matcher(txCodedBits, targetLength_punc);
    
    % Имитируем идеальный канал (0 -> +5 LLR, 1 -> -5 LLR)
    rxLLR_punc = 5 * (1 - 2 * matchedBits_punc); 
    
    % Восстановление скорости
    recoveredLLR_punc = rate_recovery(rxLLR_punc, originalLength);
    
    % Жесткое решение по восстановленным LLR (исключая нули)
    % Нулевой LLR означает "не знаю", принудительно проверим первые 7
    hardBits_punc = (recoveredLLR_punc(1:targetLength_punc) < 0);
    
    % Проверка
    if length(matchedBits_punc) == targetLength_punc && ...
       all(recoveredLLR_punc(targetLength_punc+1:end) == 0) && ...
       all(txCodedBits(1:targetLength_punc) == hardBits_punc)
   
        fprintf('[УСПЕХ] Тест 1: Режим Выкалывания (Puncture).\n');
        fprintf('        Вход: %d бит -> Канал: %d бит -> Восстановлено: %d LLR\n', ...
                originalLength, targetLength_punc, length(recoveredLLR_punc));
        fprintf('        Выколотые хвосты успешно заполнены нейтральными нулями.\n\n');
    else
        fprintf('[ОШИБКА] Тест 1: Режим выкалывания отработал некорректно!\n');
        return;
    end
    
    %% 3. ТЕСТ 2: РЕЖИМ ПОВТОРЕНИЯ (REPETITION)
%% 3. ТЕСТ 2: РЕЖИМ ПОВТОРЕНИЯ (REPETITION)
    targetLength_rep = 23; % Требуется сильно раздуть поток (2 полных круга + 3 бита)
    
    % Выполняем рейт-матчинг (повторение)
    matchedBits_rep = rate_matcher(txCodedBits, targetLength_rep);
    
    % Имитируем идеальный BPSK-канал, где каждый бит превращается в LLR.
    % Переводим биты {0, 1} в амплитуды LLR: 0 -> +2, 1 -> -2.
    % ТЕПЕРЬ matchedBits_rep ИСПОЛЬЗУЮТСЯ НАПРЯМУЮ!
    rxLLR_rep = 2 * (1 - 2 * matchedBits_rep);
    
    % Восстановление скорости (мягкое сложение LLR)
    recoveredLLR_rep = rate_recovery(rxLLR_rep, originalLength);
    
    % Рассчитываем ожидаемый идеальный LLR вручную для проверки:
    % Исходный вектор: [1; 0; 1; 1; 0; 0; 1; 0; 1; 1] -> в LLR это [-2; +2; -2; -2; +2; +2; -2; +2; -2; -2]
    % Первые 3 бита повторились 3 раза, остальные — 2 раза.
    expected_LLR = [-6; +6; -6; -4; +4; +4; -4; +4; -4; -4];
    
    if isequal(recoveredLLR_rep, expected_LLR)
        fprintf('[УСПЕХ] Тест 2: Режим Повторения (Repetition).\n');
        fprintf('        Вход: %d бит -> Канал: %d бит -> Мягкое объединение (MRC) выполнено.\n', ...
                originalLength, targetLength_rep);
        fprintf('        Метрики LLR накоплены со 100%% точностью на основе переданных бит.\n\n');
    else
        fprintf('[ОШИБКА] Тест 2: Энергетическое суммирование LLR не совпало с ожидаемым!\n');
        disp('Полученный LLR | Ожидаемый LLR:');
        disp([recoveredLLR_rep, expected_LLR]);
        return;
    end
   
    fprintf('==================================================\n');
    fprintf('      ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!                 \n');
    fprintf('==================================================\n');
end
