function plot_combined_ber_from_files()
    %% 1. Загрузка данных из трёх файлов
    % Укажите корректные имена файлов или пути
    files = {
        %'epsilon_scan/ntn_results_epsilon_1.mat',   % epsilon = 0
        %'epsilon_scan/ntn_results_epsilon_2.mat',   % epsilon = 4e3/3e8
        %'epsilon_scan/ntn_results_epsilon_3.mat'    % epsilon = 8e3/3e8

        'ntn_fft_512_results.mat',   % epsilon = 0
        'ntn_fft_1024_results.mat',   % epsilon = 4e3/3e8
        'ntn_fft_2048_results.mat'    % epsilon = 8e3/3e8

    };
    
    % Метки для легенды
    epsilon_labels = {'0', '4 км/с', '8 км/с'};
    line_styles = {'--', '-', '-'};   % пунктир для epsilon=0, сплошные для остальных
    
    % Цвета модуляций (как в исходном коде)
    colors = {'b', 'r', 'g', 'm'};
    
    % Загружаем первый файл, чтобы получить EbNo_vec и modTypes
    temp = load(files{1});
    EbNo_vec = temp.EbNo_vec;
    modTypes = temp.modTypes;
    
    % Структура для хранения BER_coded: [eps_idx, mod_idx]
    % Будем хранить в ячейке, т.к. размеры могут различаться (досрочный выход)
    BER_coded_all = cell(length(files), length(modTypes));
    
    for f = 1:length(files)
        data = load(files{f});
        % Проверяем, что переменная называется BER_coded (как в предыдущем скрипте)
        BER_coded = data.BER_coded;   % матрица [len(modTypes) x len(EbNo_vec)]
        
        for m = 1:length(modTypes)
            % Сохраняем весь вектор BER для данной модуляции
            BER_coded_all{f, m} = BER_coded(m, :);
        end
        fprintf('Загружен файл %s: epsilon = %.2e\n', files{f}, data.fft_size);%data.epsilon);
    end
    
    %% 2. Построение сводного графика
    figure('Color', 'w', 'Position', [100, 100, 900, 600]);
    hold on;
    
    for eps_idx = 1:length(files)
        for m = 1:length(modTypes)
            ber_vec = BER_coded_all{eps_idx, m};
            % Находим ненулевые значения (кривая могла оборваться из-за досрочного выхода)
            valid = (ber_vec > 0);
            if any(valid)
                % Строим с соответствующей цветом и стилем
                semilogy(EbNo_vec(valid), ber_vec(valid), ...
                    [colors{m} line_styles{eps_idx}], 'LineWidth', 2, ...
                    'Marker', 'o', 'MarkerSize', 4, ...
                    'DisplayName', sprintf('%s (%s)', modTypes{m}, epsilon_labels{eps_idx}));
            else
                warning('Модуляция %s, epsilon %s: все BER равны нулю или не заданы', ...
                    modTypes{m}, epsilon_labels{eps_idx});
            end
        end
    end
    
    grid on;
    xlabel('E_b/N_0 (dB)');
    ylabel('Bit Error Rate (BER)');
    title('Сравнение влияния доплеровского сдвига (только FEC + MMSE)');
    set(gca, 'YScale', 'log', 'YLim', [1e-5 1], 'XLim', [EbNo_vec(1) EbNo_vec(end)]);
    legend show;
    legend('Location', 'southwest');
    
    %% 3. Сохранение итогового графика (опционально)
    saveas(gcf, 'combined_ber_epsilon_comparison.png');
    fprintf('График сохранён в combined_ber_epsilon_comparison.png\n');
end