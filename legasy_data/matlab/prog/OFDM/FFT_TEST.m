bps = 1;    % Bits per symbol
M = 2^bps;  % 16QAM
nFFT = 128; % Number of FFT bins

  txsymbols = randi([0 M-1],nFFT,1);
txgrid = qammod(txsymbols,M,UnitAveragePower=true);
txout = ifft(txgrid,nFFT);
stem(1:nFFT,real(txout))
