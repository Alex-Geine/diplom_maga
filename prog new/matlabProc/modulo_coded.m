function modulo_coded()
%% 1. ИСХОДНЫЕ НАСТРОЙКИ И ПОДКЛЮЧЕНИЕ ПАПОК
clear; clc; close all;

% Подключаем вашу папку с блоками
addpath('C:\Users\a.blagodatin\Desktop\temp\diploma\blocks\');

modType = '16QAM'; % Доступно: 'QPSK', '16QAM', '64QAM', '256QAM'
N_info = 5000;     % Количество ИНФОРМАЦИОННЫХ бит
EbNo_dB = 6;       % Отношение Eb/No

% Настройка глубины перемежения (количество строк матрицы интерливера)
% Для диплома: чем больше число, тем сильнее защита от длинных пакетов ошибок.
numRowsInterleaver = 40; 

% Определение количества бит на символ для маппера
switch modType
    case 'QPSK',   bps = 2;
    case '16QAM',  bps = 4;
    case '64QAM',  bps = 6;
    case '256QAM', bps = 8;
end

%% 2. ГЕНЕРАЦИЯ ДАННЫХ, КОДИРОВАНИЕ И ПЕРЕМЕЖЕНИЕ
% Генерация случайных информационных бит (вектор-столбец)
txInfoBits = randi([0 1], N_info, 1);

% Помехоустойчивое кодирование (скорость R=1/2, длина потока удваивается)
txCodedBits = conv_encoder(txInfoBits);
lenCodedOriginal = length(txCodedBits); % Запоминаем исходную кодовую длину

% Блочное перемежение бит (защита от пакетных ошибок)
txInterleavedBits = interleaver(txCodedBits, numRowsInterleaver);

% Перегруппировка ПЕРЕМЕШАННЫХ бит в матрицу под формат маппера
% Строка матрицы — один символ созвездия, столбцы — биты этого символа
numSymbols = length(txInterleavedBits) / bps;
txBitsMatrix = reshape(txInterleavedBits, bps, numSymbols).';

%% 3. МОДУЛЯЦИЯ И ДИСКРЕТНЫЙ КАНАЛ (AWGN)
% Вызов вашего внешнего маппера
[txSig, constellation, bitMap] = mapper(txBitsMatrix, modType);

% Расчет шума с учетом избыточности кодирования (R=1/2)
R = 1/2; 
EsNo_dB = EbNo_dB + 10*log10(bps) + 10*log10(R);
EsNo = 10^(EsNo_dB/10);
noiseVar = 1 / EsNo; 

% Генерация и добавление комплексного белого шума
noise = sqrt(noiseVar/2) * (randn(size(txSig)) + 1i*randn(size(txSig)));
rxSig = txSig + noise;

%% 4. ВИЗУАЛИЗАЦИЯ ЗАШУМЛЕННОГО СОЗВЕЗДИЯ
figure('Color', 'w');
plot(real(rxSig), imag(rxSig), '.', 'Color', [0.7 0.7 0.7], 'MarkerSize', 6);
hold on;
plot(real(constellation), imag(constellation), 'r+', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
title(sprintf('Созвездие %s с кодированием и интерливером (Eb/No = %.1f dB)', modType, EbNo_dB));
xlabel('In-Phase (I)');
ylabel('Quadrature (Q)');
legend('Принятый зашумленный сигнал', 'Идеальные точки', 'Location', 'best');
axis square;

%% 5. МЯГКАЯ ДЕМОДУЛЯЦИЯ, ДЕИНТЕРЛИВЕР И ВИТЕРБИ ДЕКОДИРОВАНИЕ
% Вызов вашего внешнего софт-демаппера (получаем матрицу LLR)
llrMatrix = soft_demapper(rxSig, constellation, bitMap, noiseVar);

% Вытягиваем LLR обратно в один непрерывный битовый поток (вектор-столбец)
% Транспонируем, чтобы биты считывались последовательно, символ за символом
llrBitsStream = llrMatrix.'; 
llrBitsStream = llrBitsStream(:);

% Обратное перемежение (деинтерливер) потока мягких LLR-метрик.
% Передаем сохраненную длину lenCodedOriginal, чтобы отрезать дополняющие нули интерливера.
llrDeinterleaved = deinterleaver(llrBitsStream, numRowsInterleaver, lenCodedOriginal);

% Мягкое декодирование Витерби на основе восстановленных LLR
rxInfoBits = viterbi_soft_decoder(llrDeinterleaved, N_info);

%% 6. СРАВНИТЕЛЬНЫЙ АНАЛИЗ ОШИБОК (BER)
% 6.1 Расчет BER С кодированием
ber_coded = sum(txInfoBits ~= rxInfoBits) / N_info;

% 6.2 Расчет BER БЕЗ кодирования (для наглядности)
% Чтобы оценить качество самого радиоканала, делаем жесткое решение по 
% деинтерливированным LLR и сравниваем их с исходными кодированными битами
hardCodedBits = (llrDeinterleaved < 0);
ber_uncoded = sum(txCodedBits ~= hardCodedBits) / length(txCodedBits);

% Вывод результатов в консоль
fprintf('\n========================================\n');
fprintf('Тип модуляции: %s\n', modType);
fprintf('Отношение Eb/No: %.1f dB\n', EbNo_dB);
fprintf('Глубина интерливера: %d строк\n', numRowsInterleaver);
fprintf('----------------------------------------\n');
fprintf('BER без кодирования (в канале): %g\n', ber_uncoded);
fprintf('BER с мягким декодером Витерби: %g\n', ber_coded);
fprintf('========================================\n');
end
