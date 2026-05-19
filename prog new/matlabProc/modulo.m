function re = modulo()
addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');

%% 1. ИСХОДНЫЕ НАСТРОЙКИ И ЗАПУСК
clear; clc; close all;

modType = '256QAM'; % Доступно: 'QPSK', '16QAM', '64QAM', '256QAM'
N = 10000;         % Количество символов
EbNo_dB = 10;      % Отношение сигнал/шум на бит

% Генерация случайных бит
switch modType
    case 'QPSK',   bps = 2;
    case '16QAM',  bps = 4;
    case '64QAM',  bps = 6;
    case '256QAM', bps = 8;
end
txBits = randi([0 1], N, bps);

%% 2. РАБОТА ТРАНСИВЕРА
% Модуляция и получение эталонного созвездия
[txSig, constellation, bitMap] = mapper(txBits, modType);

% Расчет дисперсии шума и добавление помех
EsNo_dB = EbNo_dB + 10*log10(bps);
EsNo = 10^(EsNo_dB/10);
noiseVar = 1 / EsNo; 
noise = sqrt(noiseVar/2) * (randn(size(txSig)) + 1i*randn(size(txSig)));
rxSig = txSig + noise;

%% 3. ВИЗУАЛИЗАЦИЯ СОЗВЕЗДИЯ
figure('Color', 'w');
plot(real(rxSig), imag(rxSig), '.', 'Color', [0.7 0.7 0.7], 'MarkerSize', 6);
hold on;
plot(real(constellation), imag(constellation), 'r+', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
title(sprintf('Сигнальное созвездие %s (Eb/No = %.1f dB)', modType, EbNo_dB));
xlabel('In-Phase (I)');
ylabel('Quadrature (Q)');
legend('Зашумленный сигнал', 'Идеальные точки', 'Location', 'best');
axis square;

%% 4. МЯГКАЯ ДЕМОДУЛЯЦИЯ И ПРОВЕРКА
% Получение мягких решений (LLR)
llr = soft_demapper(rxSig, constellation, bitMap, noiseVar);

% Жесткое решение на основе LLR (положительный LLR -> 0, отрицательный LLR -> 1)
hardBits = (llr < 0); 

% Расчет BER
ber = sum(txBits(:) ~= hardBits(:)) / (N * bps);
fprintf('Модуляция: %s\n', modType);
fprintf('SNR (Eb/No): %.1f dB\n', EbNo_dB);
fprintf('Текущий BER (Hard Decision): %g\n', ber);
end