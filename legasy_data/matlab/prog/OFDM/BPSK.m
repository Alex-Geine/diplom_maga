% Параметры
f = 10; % Частота несущей (Гц)
Fs = 100; % Частота дискретизации (Гц)
T = 1/Fs; % Период дискретизации
t = 0:T:1-T; % Временной вектор (1 секунда)
bits = [0 1 0 1 1 0 1 0]; % Бинарные данные (пример)

% BPSK модуляция
bpsk_symbols = 2 * bits - 1; % Преобразование битов в BPSK символы: 0 -> -1, 1 -> +1

% Создание несущей частоты
carrier_I = cos(2 * pi * f * t); % In-phase компонента (I)
carrier_Q = sin(2 * pi * f * t); % Quadrature компонента (Q)

% Модуляция BPSK
modulated_I = zeros(1, length(t));
modulated_Q = zeros(1, length(t)); % Q компонента будет нулевой для BPSK
for i = 1:length(bits)
    start_idx = (i-1) * length(t)/length(bits) + 1;
    end_idx = i * length(t)/length(bits);
    modulated_I(start_idx:end_idx) = bpsk_symbols(i) * carrier_I(start_idx:end_idx);
    modulated_Q(start_idx:end_idx) = 0; % Q компонента всегда 0 для BPSK
end

% Построение графиков
figure;

% График бинарных данных
subplot(4, 1, 1);
stem(bits, 'LineWidth', 2);
title('Бинарные данные');
xlabel('Биты');
ylabel('Значение');
ylim([-0.5 1.5]);
grid on;

% График несущей частоты (I компонента)
subplot(4, 1, 2);
plot(t, carrier_I, 'b');
title('Несущая частота (I компонента)');
xlabel('Время (с)');
ylabel('Амплитуда');
grid on;

% График модулированного сигнала (I компонента)
subplot(4, 1, 3);
plot(t, modulated_I, 'r');
title('BPSK модулированный сигнал (I компонента)');
xlabel('Время (с)');
ylabel('Амплитуда');
grid on;

% График модулированного сигнала (Q компонента)
subplot(4, 1, 4);
plot(t, modulated_Q, 'g');
title('BPSK модулированный сигнал (Q компонента)');
xlabel('Время (с)');
ylabel('Амплитуда');
grid on;
