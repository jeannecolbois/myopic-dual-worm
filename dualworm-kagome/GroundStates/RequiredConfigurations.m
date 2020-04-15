function [DiffMFUConfigs, TotMFUConfigs] = RequiredConfigurations(unit,foldername)
% [configurations] = RequiredConfigurations(unit, foldername)
% For a unit cell "unit" and a list of ground state configurations "foldername"
% this function returns the configurations on the unit cell which are
% expected to be in the ground state of a proper MFU hamiltonian on this unit cell
%
% The field 'sites' in unit, contains an array of coordinates of all the kagome sites included in the unit.
% The field 'basis' in unit, defines a superimposed square lattice grid, the center of each MFU may be imagined
%    on the vertices of this grid. The tensor network will also take this shape


% 1 - Load the folder family by family
listing = dir(foldername + "Family*"); % list all states

% 2 - For each family, find all the unit cell configurations which are allowed
% (given the basis)
nmaxfam = length(listing);
MFUConfigs = {};
for i = 1:nmaxfam
    familyfolder = listing(i).folder+"/"+listing(i).name+"/";
    familylist = dir(familyfolder+"matrix*.txt");
    family = {};
    for j = 1:length(familylist)
        family{j} = importdata(familyfolder+familylist(j).name);
    end
    MFUConfigs{i} = getMFUConfigurations(unit, family); 
end

% sort by number of MFU configurations
[id1, ~] = cellfun( @size, MFUConfigs );
[~,id] = sort(id1);
MFUConfigs = MFUConfigs(id);
smax = max(id1);

DiffMFUConfigs = {};
count = 0;
for i = 1:numel(MFUConfigs)
    include = true;% by default, the configuration will be included
    for j = i+1:numel(MFUConfigs)
       % if the size is different, then check intersection, otherwise
       % compare
       if size(MFUConfigs{i},1) < size(MFUConfigs{j},1)
           compare = intersect(MFUConfigs{i},MFUConfigs{j},'rows');
           if size(compare) == size(MFUConfigs{i})
               compare = compare == MFUConfigs{i};
           else
               compare = [false];
           end
            
           if all(compare(:)) % then i is included in j and can be neglected
               include = false;
           end
       else % if they have the same size, then we can make a row-by-row comparison
           same = ~(MFUConfigs{i}-MFUConfigs{j}); % This works because they have been sorted before
           if all(same) % then i is the same as j and can be neglected
               include = false;
           end
       end   
    end
    if include
        count = count+1;
        DiffMFUConfigs{count} = MFUConfigs{i};
    end
end

TotMFUConfigs = DiffMFUConfigs{1};
for i = 1:numel(DiffMFUConfigs)
    TotMFUConfigs = [TotMFUConfigs; DiffMFUConfigs{i}];
    TotMFUConfigs = unique(TotMFUConfigs, 'rows');
end
