function encodedBits = conv_encoder(txBits)
% OPTIMIZED_CONV_ENCODER Ускоренный сверточный кодер (Rate=1/2, K=7)
    bits = txBits(:).'; 
    
    % Полиномы в двоичном виде
    g1 = [1 1 1 1 0 0 1]; 
    g2 = [1 0 1 1 0 1 1];
    
    % Вместо цикла используем быструю встроенную фильтрацию по модулю 2
    out1 = mod(filter(g1, 1, bits), 2);
    out2 = mod(filter(g2, 1, bits), 2);
    
    % Быстрое чередование (мультиплексирование) бит без циклов
    encodedBits = zeros(1, 2*length(bits));
    encodedBits(1:2:end) = out1;
    encodedBits(2:2:end) = out2;
    
    encodedBits = encodedBits(:);
end
