function deinterleavedLLR = deinterleaver(llrBits, numRows, originalLength)
% MY_DEINTERLEAVER Обратный блочный перемежитель для потока LLR
% Вход: llrBits        — вектор LLR от демодулятора
%       numRows        — то же количество строк, что использовалось в интерливере
%       originalLength — исходная длина вектора бит до интерливера (длина txCodedBits)

    N = length(llrBits);
    numCols = N / numRows;
    
    % Восстанавливаем матрицу: так как интерливер считывал по столбцам,
    % мы заполняем матрицу по столбцам с помощью reshape
    matrix = reshape(llrBits, numRows, numCols);
    
    % Считываем построчно (транспонируем и вытягиваем)
    deinterleavedLLR = matrix.';
    deinterleavedLLR = deinterleavedLLR(:);
    
    % Отрезаем дополняющие биты (нули), которые были добавлены интерливером
    deinterleavedLLR = deinterleavedLLR(1:originalLength);
end