function interleavedBits = interleaver(codedBits, numRows)
    N = length(codedBits);
    numCols = ceil(N / numRows);
    totalLen = numRows * numCols;
    
    if totalLen > N
        codedBits(totalLen) = 0; 
    end
    
    idxMatrix = reshape(1:totalLen, numCols, numRows).';

    interleavedBits = codedBits(idxMatrix(:));
end

