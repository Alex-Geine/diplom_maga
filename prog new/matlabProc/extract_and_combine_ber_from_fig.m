function extract_and_combine_ber_from_fig()
    % Скрипт для извлечения данных из fig-файла с тремя подграфиками
    % и построения объединённого графика BER (только coded)
    
    %% 1. Выбор источника данных
    answer = questdlg('Использовать текущую открытую фигуру или выбрать fig-файл?', ...
        'Источник данных', 'Текущая фигура', 'Выбрать файл', 'Текущая фигура');
    if strcmp(answer, 'Выбрать файл')
        [fname, pathname] = uigetfile('*.fig', 'Выберите fig-файл с тремя подграфиками');
        if isequal(fname,0)
            disp('Файл не выбран. Работа прекращена.');
            return;
        end
        fig = openfig(fullfile(pathname, fname));
    else
        fig = gcf;
    end
    
    %% 2. Поиск всех осей (subplot)
    ax_all = findobj(fig, 'Type', 'Axes');
    if length(ax_all) < 3
        error('Найдено менее 3 осей. Убедитесь, что фигура содержит три подграфика.');
    end
    
    % Сортируем оси по горизонтальной позиции (слева направо)
    pos = get(ax_all, 'Position');
    if iscell(pos)
        pos = cell2mat(pos);
    end
    [~, idx] = sort(pos(:,1));
    ax = ax_all(idx);
    ax = ax(1:3);   % берём первые три (если их больше)
    
    %% 3. Параметры
    epsilon_labels = {'0', '4 км/с', '8 км/с'};
    epsilon_values = [0, 4e3/3e8, 8e3/3e8];
    mod_names = {'QPSK', '16QAM', '64QAM', '256QAM'};
    mod_colors = {'b', 'r', 'g', 'm'};
    % Соответствие цветов RGB (для сравнения)
    color_rgb = containers.Map();
    color_rgb('b') = [0 0 1];
    color_rgb('r') = [1 0 0];
    color_rgb('g') = [0 1 0];
    color_rgb('m') = [1 0 1];
    
    % Структура для хранения извлечённых данных
    data = {}; % каждый элемент: struct('epsilon', val, 'mod_idx', idx, 'x', [], 'y', [])
    
    for k = 1:3
        axes(ax(k));
        title_str = get(get(ax(k), 'Title'), 'String');
        fprintf('\n=== Ось %d: заголовок "%s" ===\n', k, title_str);
        % Запрашиваем, какой epsilon соответствует этой оси
        eps_choice = input(sprintf('Укажите номер epsilon для этой оси (1=%s, 2=%s, 3=%s): ', ...
            epsilon_labels{1}, epsilon_labels{2}, epsilon_labels{3}));
        if isempty(eps_choice) || eps_choice < 1 || eps_choice > 3
            eps_choice = k; % по умолчанию порядковый номер
        end
        epsilon = epsilon_values(eps_choice);
        
        % Находим все линии на данной оси
        lines = findobj(ax(k), 'Type', 'Line');
        % Отбираем только сплошные (coded) – предполагаем, что uncoded имеют стиль '--'
        solid = [];
        for i = 1:length(lines)
            if strcmp(get(lines(i), 'LineStyle'), '-')
                solid = [solid; lines(i)];
            end
        end
        if length(solid) < 4
            warning('На оси %d найдено %d сплошных линий, ожидалось 4. Будут взяты все сплошные.', k, length(solid));
        end
        
        % Для каждой сплошной линии определяем модуляцию по цвету
        for i = 1:length(solid)
            line_color = get(solid(i), 'Color'); % [R G B]
            % Ищем соответствие с ожидаемыми цветами
            found = false;
            mod_idx = [];
            for m = 1:4
                exp_rgb = color_rgb(mod_colors{m});
                if isequal(line_color, exp_rgb)
                    mod_idx = m;
                    found = true;
                    break;
                end
            end
            if ~found
                fprintf('Неизвестный цвет линии: [%f %f %f]\n', line_color(1), line_color(2), line_color(3));
                mod_idx = input('Введите номер модуляции (1-QPSK,2-16QAM,3-64QAM,4-256QAM): ');
            end
            xdata = get(solid(i), 'XData');
            ydata = get(solid(i), 'YData');
            % Убираем точки, где BER = 0 (досрочный выход)
            valid = (ydata > 0);
            data{end+1} = struct('epsilon', epsilon, 'mod_idx', mod_idx, ...
                'x', xdata(valid), 'y', ydata(valid));
        end
    end
    
    %% 4. Построение объединённого графика
    figure('Color', 'w', 'Position', [100, 100, 900, 600]);
    hold on;
    
    % Стили линий: для epsilon=0 – пунктир, иначе сплошной
    for eps_val = epsilon_values
        if eps_val == 0
            style = '--';
        else
            style = '-';
        end
        % Для каждой модуляции найдём соответствующую кривую
        for mod_idx = 1:4
            match = [];
            for d = 1:length(data)
                if abs(data{d}.epsilon - eps_val) < 1e-12 && data{d}.mod_idx == mod_idx
                    match = data{d};
                    break;
                end
            end
            if isempty(match)
                fprintf('Предупреждение: нет данных для %s, epsilon=%.2e\n', mod_names{mod_idx}, eps_val);
                continue;
            end
            semilogy(match.x, match.y, [mod_colors{mod_idx} style], ...
                'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 4, ...
                'DisplayName', sprintf('%s (%s)', mod_names{mod_idx}, get_eps_label(eps_val, epsilon_labels, epsilon_values)));
        end
    end
    
    grid on;
    xlabel('E_b/N_0 (dB)');
    ylabel('Bit Error Rate (BER)');
    title('Влияние доплеровского сдвига (только FEC+MMSE)');
    set(gca, 'YScale', 'log', 'YLim', [1e-5 1]);
    legend show;
    legend('Location', 'southwest');
    
    % Сохраняем результат
    saveas(gcf, 'combined_ber_from_fig.png');
    fprintf('\nГрафик сохранён в файл combined_ber_from_fig.png\n');
end

function label = get_eps_label(eps_val, labels, values)
    for i = 1:length(values)
        if abs(eps_val - values(i)) < 1e-12
            label = labels{i};
            return;
        end
    end
    label = num2str(eps_val);
end