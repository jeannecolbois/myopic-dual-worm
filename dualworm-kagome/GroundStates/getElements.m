function elems = getElements(indices,matrix)
    % use linear indexing
    % assuming indices = [[i1,j1];[i2,j2];...]
    s = size(matrix,1);
    nr = size(indices,1);
    lindices = zeros(nr,1);
    for i = 1:nr
        lindices(i) = indices(i,1) + s*(indices(i,2)-1);
    end
    elems = matrix(lindices);
    
end