function decodedBits = viterbi_soft_decoder(llrBits, numInfoBits)
% OPTIMIZED_VITERBI_SOFT_DECODER Ускоренный декодер Витерби (Rate=1/2, K=7)
    
    % Инициализация структуры кода
    g1 = [1 1 1 1 0 0 1]; g2 = [1 0 1 1 0 1 1];
    numStates = 64;
    
    % Предрасчет переходов (выполняется строго 1 раз за всю сессию MATLAB)
    persistent trellis_init next_states state_outputs prev_states input_bits;
    if isempty(trellis_init)
        next_states = zeros(numStates, 2);
        state_outputs = zeros(numStates, 2, 2);
        
        % Матрицы для быстрого поиска «предков» (размер 64 x 2)
        % Для каждого состояния s_next храним два возможных предыдущих состояния
        prev_states = zeros(numStates, 2);
        % И соответствующие им входные биты, которые привели в s_next
        input_bits = zeros(numStates, 2);
        
        for s = 0:numStates-1
            reg = bitget(s, 6:-1:1);
            for inputBit = 0:1
                full_reg = [inputBit, reg];
                out1 = mod(sum(full_reg .* g1), 2);
                out2 = mod(sum(full_reg .* g2), 2);
                next_s = sum(full_reg(1:6) .* (2.^(5:-1:0))) + 1;
                
                next_states(s+1, inputBit+1) = next_s;
                state_outputs(s+1, inputBit+1, :) = [1 - 2*out1, 1 - 2*out2];
            end
        end
        
        % Заполняем обратную таблицу переходов для векторизации
        for s = 1:numStates
            % Находим, из каких состояний 'p' при входе 'b' мы попадаем в 's'
            [p0, b0] = find(next_states == s);
            prev_states(s, :) = p0.';
            input_bits(s, :) = b0.' - 1; % Переводим обратно в биты 0 и 1
        end
        trellis_init = true;
    end
    
    % Инициализация метрик путей
    path_metrics = Inf(numStates, 1);
    path_metrics(1) = 0; % Стартуем из нулевого состояния
    
    % Матрица решений для обратного хода (true — пришли по 2-й ветви, false — по 1-й)
    decision_history = false(numStates, numInfoBits);
    
    % Нормировка и масштабирование LLR
    max_llr = max(abs(llrBits));
    if max_llr > 0
        llr = (llrBits / max_llr) * 4; 
    else
        llr = llrBits;
    end
    
    % Группируем LLR по парам [2 x numInfoBits]
    llr_pairs = reshape(llr, 2, numInfoBits);
    
    %% ОСНОВНОЙ ЦИКЛ ПО ВРЕМЕНИ
    for t = 1:numInfoBits
        llr1 = llr_pairs(1, t);
        llr2 = llr_pairs(2, t);
        
        % Векторизованный расчет ВСЕХ метрик ветвей (размер 64 x 2)
        % Столбец 1 — для входного бита 0, Столбец 2 — для входного бита 1
        bm0 = (state_outputs(:, 1, 1) - llr1).^2 + (state_outputs(:, 1, 2) - llr2).^2;
        bm1 = (state_outputs(:, 2, 1) - llr1).^2 + (state_outputs(:, 2, 2) - llr2).^2;
        
        % Матрица полных метрик для всех возможных путей (64 x 2)
        % Каждая строка s соответствует целевому состоянию. 
        % Столбец 1 — путь через первого предка, Столбец 2 — через второго предка.
        p_prev1 = prev_states(:, 1);
        p_prev2 = prev_states(:, 2);
        
        % Ветвь, по которой шли, определяется сохраненным входным битом
        bit_from_1 = input_bits(:, 1);
        bit_from_2 = input_bits(:, 2);
        
        % Считаем метрику ветви для первого предка
        bm_p1 = bm0(p_prev1) .* (bit_from_1 == 0) + bm1(p_prev1) .* (bit_from_1 == 1);
        % Считаем метрику ветви для второго предка
        bm_p2 = bm0(p_prev2) .* (bit_from_2 == 0) + bm1(p_prev2) .* (bit_from_2 == 1);
        
        % Суммируем старую метрику пути и метрику новой ветви
        candidates = [path_metrics(p_prev1) + bm_p1, path_metrics(p_prev2) + bm_p2];
        
        % Метод ACS (Add-Compare-Select): выбираем минимум для каждого из 64 состояний
        [path_metrics, decs] = min(candidates, [], 2);
        
        % Сохраняем историю выбора (1 или 2)
        decision_history(:, t) = (decs == 2);
        
        % Защита от переполнения (динамическое выравнивание шкал)
        path_metrics = path_metrics - min(path_metrics);
    end
    
    %% ОБРАТНЫЙ ХОД (Traceback) — выполняется 1 раз в конце
    decodedBits = zeros(numInfoBits, 1);
    
    % Находим финальное состояние с наименьшей метрикой
    [~, curr_state] = min(path_metrics); 
    
    for t = numInfoBits:-1:1
        % К какому предку ведет выживший путь (1-й или 2-й столбец)
        dec_idx = decision_history(curr_state, t) + 1; 
        
        % Достаем информационный бит, который совершил данный переход
        decodedBits(t) = input_bits(curr_state, dec_idx);
        
        % Переходим к предыдущему состоянию
        curr_state = prev_states(curr_state, dec_idx);
    end
    
    decodedBits = decodedBits(:);
end
