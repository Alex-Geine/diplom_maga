function [txSig, constellation, bitMap] = mapper(txBits, modType)
    switch modType
        case 'QPSK',   M = 4;   bps = 2;
        case '16QAM',  M = 16;  bps = 4;
        case '64QAM',  M = 64;  bps = 6;
        case '256QAM', M = 256; bps = 8;
    end

    k = sqrt(M);
    if M == 4
        [X, Y] = meshgrid([-1 1], [-1 1]);
    else
        pam = (-(k-1):2:(k-1));
        [X, Y] = meshgrid(pam, -pam);
    end
    constellation = X(:) + 1i*Y(:);

    gray1D = [0; 1];
    for i = 2:log2(k)
        gray1D = [gray1D; flipud(gray1D) + 2^(i-1)];
    end

    bitMap = zeros(M, bps);
    for idx = 1:M
        [r, c] = find(X == real(constellation(idx)) & Y == imag(constellation(idx)), 1);
        grayI = gray1D(c);
        grayQ = gray1D(r);
        bitStr = [dec2bin(grayI, bps/2), dec2bin(grayQ, bps/2)];
        bitMap(idx, :) = bitStr - '0';
    end

    constellation = constellation / sqrt(mean(abs(constellation).^2));

    [~, txIndices] = ismember(txBits, bitMap, 'rows');
    txSig = constellation(txIndices).';
end