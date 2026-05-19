function interleavedBits = interleaver(codedBits, numRows)
% OPTIMIZED_INTERLEAVER Максимально быстрый матричный перемежитель
    N = length(codedBits);
    numCols = ceil(N / numRows);
    totalLen = numRows * numCols;
    
    % Быстрое дополнение нулями без выделения новой памяти через конкатенацию
    if totalLen > N
        % Используем встроенное расширение массива, это быстрее
        codedBits(totalLen) = 0; 
    end
    
    % Предрасчет индексов (выполняется мгновенно через арифметику указателей)
    % Запись построчно, чтение по столбцам эквивалентно перестановке индексов:
    idx = reshape(1:totalLen, numCols, numRows).';
    
    % Перемежение в одну строчку без транспонирования данных
    interleavedBits = codedBits(idx(:));
end
