function [rxBlock, H_freq, N0] = channel_apply(txBlock, profile_name, fft_size, scs_khz, d_km, fc_ghz, shadowing_std_db, I, snr_lin)
% CHANNEL_APPLY Посимвольный спутниковый 5G NTN канал (Вход/Выход — вектор длины fft_size)
% Вход:
%   txBlock          - Вектор комплексных частотных символов одного OFDM-блока (размер fft_size x 1)
%   profile_name     - Строка 'A', 'B' или 'C' (профиль 3GPP TR 38.901)
%   fft_size         - Количество поднесущих (размер FFT)
%   scs_khz          - Разнос поднесущих в кГц
%   d_km             - Дистанция до спутника LEO в км
%   fc_ghz           - Несущая частота в ГГц
%   shadowing_std_db - СКО логнормального затенения в дБ
%   I                - Матрица межподнесущей интерференции (ICI) [fft_size x fft_size]
%   snr_lin          - Линейное значение SNR (Es/No)

    %% 1. РАСЧЕТ СИСТЕМНЫХ КОНСТАНТ И ПОТЕРЬ (FSPL)
    fs = fft_size * (scs_khz * 1e3);
    Ts = 1 / fs;
    d_m = d_km * 1e3;
    fc = fc_ghz * 1e9;
    lambda_c = 3e8 / fc;
    
    % Расчет Free Space Path Loss (FSPL) + Гауссовское затенение (Shadowing) без тулбоксов
    Pd_db = 10 * log10((4 * pi * d_m / lambda_c)^2);
    Ps_db = shadowing_std_db * randn(); 
    path_loss_factor = sqrt(10^((-(Pd_db + Ps_db)) / 10.0));

    %% 2. ЗАГРУЗКА МАССИВОВ ЗАДЕРЖЕК 3GPP СЛУЧАЙНЫХ ЛУЧЕЙ (РЭЛЕЙ)
    if strcmp(profile_name, 'A')
        delays_ns = [0, 30, 70, 90, 110, 190, 410, 530, 750, 1070, 1090, 1290];
        powers_db = [-13.4, -0.0, -9.2, -7.5, -4.7, -11.0, -4.7, -16.6, -11.8, -13.4, -19.4, -24.8];
    elseif strcmp(profile_name, 'B')
        delays_ns = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200];
        powers_db = [-7.8, -6.2, -7.2, -8.6, -7.5, -10.0, -8.7, -11.0, -11.2, -12.8, -13.4, -14.5, -15.2, -16.9, -17.2, -18.0, -19.4, -20.7, -21.3, -26.4];
    elseif strcmp(profile_name, 'C')
        delays_ns = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200];
        powers_db = [-3.5, -6.8, -8.2, -9.2, -10.2, -11.1, -12.2, -13.2, -14.1, -15.1, -17.6, -18.7, -19.7, -20.7, -21.7, -22.7, -23.7, -24.7, -25.7, -26.7];
    else
        error('Неизвестный профиль TDL канала');
    end
    
    delays_samples = round(delays_ns * 1e-9 / Ts);
    path_gains_lin = 10.^(powers_db / 10.0);
    ir_len = max(delays_samples) + 1;

    %% 3. ГЕНЕРАЦИЯ ИМПУЛЬСНОЙ ХАРАКТЕРИСТИКИ И FFT
    h = zeros(ir_len, 1) + 1i*zeros(ir_len, 1);
    for idx = 1:length(delays_samples)
        delay_idx = delays_samples(idx) + 1;
        amp = sqrt(path_gains_lin(idx)) * path_loss_factor;
        h(delay_idx) = h(delay_idx) + amp * exp(1i * 2 * pi * rand());
    end
    
    % Получение частотной характеристики (ЧХ) одного OFDM-символа
    h_full = zeros(fft_size, 1);
    h_full(1:length(h)) = h;
    H_freq = fft(h_full) / sqrt(fft_size);
    
    % В соответствии с вашим Python-шаблоном:
    %H_freq = ones(fft_size, 1) + 1i*zeros(fft_size, 1);

    %% 4. МАТРИЧНОЕ НАЛОЖЕНИЕ ICI И АДДИТИВНОГО ШУМА
    X_block = txBlock(:).'; % Вектор-строка для корректного перемножения [1 x fft_size]
    
    % Искажение ЧХ канала и применение Доплеровской матрицы интерференции поднесущих
    Y_block_ideal = X_block .* H_freq.';
    Y_block = Y_block_ideal * I; 
    
    % Расчет мощности и комплексного Гауссовского шума
    P_signal = mean(abs(Y_block).^2);
    N0 = P_signal / snr_lin;
    noise = sqrt(N0/2) * (randn(size(Y_block)) + 1i*randn(size(Y_block)));
    
    % Формируем выходной вектор-столбец
    rxBlock = (Y_block + noise).';
end
