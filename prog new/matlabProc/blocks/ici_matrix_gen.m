function I = ici_matrix_gen(fft_size, alpha_D, epsilon)
% ICI_MATRIX_GEN Генерирует матрицу межподнесущей интерференции (ICI) для 5G NTN
% Работает на базовом MATLAB без использования Signal Processing Toolbox

    % Создаем сетку индексов поднесущих: n - строки, k - столбцы
    [K, N] = meshgrid(0:fft_size-1, 0:fft_size-1);
    
    % Вычисляем вспомогательную переменную z для каждого элемента матрицы
    Z = (1 + epsilon) * N + alpha_D - K;
    
    %% БЕЗОПАСНЫЙ РАСЧЕТ SINC (ОБХОД ДЕЛЕНИЯ НА НОЛЬ)
    % Вычисляем числитель: sin(pi * z)
    numerator = sin(pi * Z);
    
    % Вычисляем знаменатель: pi * z
    denominator = pi * Z;
    
    % Создаем маску для поиска элементов, где знаменатель равен или очень близок к нулю
    zero_indices = (abs(denominator) < 1e-12);
    
    % Выполняем поэлементное деление матриц
    S = numerator ./ denominator;
    
    % Согласно первому замечательному пределу, в точках z = 0 значение sinc равно 1
    S(zero_indices) = 1.0;
    
    %% РАСЧЕТ ИТОГОВОЙ МАТРИЦЫ ICI
    % Вычисляем комплексную фазовую экспоненту
    E = exp(1i * pi * Z);
    
    % Формируем итоговую матрицу
    I = E .* S;
    
    % Нормировка энергии по строкам для сохранения SNR в канале
    %row_energies = sqrt(sum(abs(I).^2, 2));
    %I = I ./ row_energies;
end
