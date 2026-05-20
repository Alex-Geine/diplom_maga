function deinterleavedLLR = deinterleaver(llrBits, numRows, originalLength)
    N = length(llrBits);
    numCols = N / numRows;
    
    idxMatrix = reshape(1:N, numCols, numRows).';
    deinterleavedLLR = zeros(N, 1);
    deinterleavedLLR(idxMatrix(:)) = llrBits;

    if N > originalLength
        deinterleavedLLR = deinterleavedLLR(1:originalLength);
    end
end
