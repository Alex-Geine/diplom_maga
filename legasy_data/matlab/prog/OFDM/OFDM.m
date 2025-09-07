% Параметры OFDM
N = 64; % Количество поднесущих
cp_len = 16; % Длина циклического префикса
num_symbols = 10; % Количество OFDM символов
f = 1000; % Несущая частота (Гц)
Fs = 10000; % Частота дискретизации (Гц)
T = 1/Fs; % Период дискретизации
t = 0:T:(N + cp_len) * num_symbols * T - T; % Временной вектор
SNR_dB = 6; % Заданное значение SNR в дБ

% Генерация случайных данных (BPSK модуляция)
data = randi([0 1], 1, N * num_symbols); % Биты 0 и 1
bpsk_symbols = 2 * data - 1; % BPSK: 0 -> -1, 1 -> +1

% Построение графика начальных данных (BPSK)
figure;
stem(data(1:50), 'LineWidth', 2); % Покажем первые 50 бит для наглядности
title('Начальные данные (BPSK)');
xlabel('Биты');
ylabel('Значение');
ylim([-0.5 1.5]);
grid on;

% Преобразование данных в матрицу поднесущих
bpsk_symbols = reshape(bpsk_symbols, N, num_symbols);

% Обратное быстрое преобразование Фурье (IFFT)
ofdm_symbols = ifft(bpsk_symbols);

% Добавление циклического префикса
ofdm_symbols_with_cp = [ofdm_symbols(end-cp_len+1:end, :); ofdm_symbols];

% Преобразование матрицы в один временной ряд
ofdm_signal = ofdm_symbols_with_cp(:).';

% Перенос на несущую частоту
carrier = exp(1j * 2 * pi * f * t); % Комплексная экспонента
ofdm_signal_modulated = ofdm_signal .* carrier; % Перенос на несущую частоту

% Добавление шума
signal_power = mean(abs(ofdm_signal_modulated).^2); % Мощность сигнала
SNR_linear = 10^(SNR_dB / 10); % Преобразование SNR из дБ в линейный масштаб
noise_power = signal_power / SNR_linear; % Мощность шума
noise = sqrt(noise_power/2) * (randn(size(ofdm_signal_modulated)) + 1j * randn(size(ofdm_signal_modulated))); % Комплексный шум
ofdm_signal_noisy = ofdm_signal_modulated + noise; % Сигнал с шумом

% Построение графика модулированного сигнала с шумом
figure;
plot(t, real(ofdm_signal_noisy), 'b');
xlabel('Время');
ylabel('Амплитуда');
title('OFDM сигнал с шумом (SNR = 6 дБ)');
grid on;

% Построение спектра модулированного сигнала с шумом
L = length(ofdm_signal_noisy); % Длина сигнала
f_axis = Fs * (-L/2:L/2-1) / L; % Вектор частот
spectrum_noisy = fftshift(abs(fft(ofdm_signal_noisy))); % Спектр сигнала

figure;
plot(f_axis, spectrum_noisy);
xlabel('Частота (Гц)');
ylabel('Амплитуда');
title('Спектр OFDM сигнала с шумом (SNR = 6 дБ)');
grid on;

% OFDM приемник
% Обратный перенос на приемнике
ofdm_signal_demodulated = ofdm_signal_noisy .* conj(carrier); % Обратный перенос

% Разделение сигнала на OFDM символы с циклическим префиксом
ofdm_symbols_with_cp_rx = reshape(ofdm_signal_demodulated, N + cp_len, num_symbols);

% Удаление циклического префикса
ofdm_symbols_rx = ofdm_symbols_with_cp_rx(cp_len+1:end, :);

% Прямое быстрое преобразование Фурье (FFT)
received_symbols = fft(ofdm_symbols_rx);

% BPSK демодуляция
received_bits = real(received_symbols) > 0; % Демодуляция: положительные значения -> 1, отрицательные -> 0

% Преобразование матрицы в один битовый поток
received_data = received_bits(:).';

% Построение графика принятых данных (BPSK)
figure;
stem(received_data(1:50), 'LineWidth', 2); % Покажем первые 50 бит для наглядности
title('Принятые данные (BPSK)');
xlabel('Биты');
ylabel('Значение');
ylim([-0.5 1.5]);
grid on;

% Проверка ошибок
error_rate = sum(data ~= received_data) / length(data);
fprintf('Частота ошибок: %.4f\n', error_rate);
