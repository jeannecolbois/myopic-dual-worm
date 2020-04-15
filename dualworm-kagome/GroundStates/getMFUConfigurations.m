function MFUConfigs = getMFUConfigurations(unit, family)
% [configurations] = RequiredConfigurations(unit, foldername)
% For a unit cell "unit" and a family of ground state configuraion
% this function returns the configurations on the unit cell which are
% expected to be in the ground state of a proper MFU hamiltonian on this unit cell
%
% The field 'sites' in unit, contains an array of coordinates of all the kagome sites included in the unit.
% The field 'basis' in unit, defines a superimposed square lattice grid, the center of each MFU may be imagined
%    on the vertices of this grid. The tensor network will also take this shape
%
% Output: MFUConfigs: columns are configurations of the unit on the whole
% family

MFUConfigs = [];
for n = 1:numel(family)
    state = family{n};
    configs = getMFUConfigurationsFromState(unit, state);
    MFUConfigs = [MFUConfigs; configs];
    MFUConfigs = unique(MFUConfigs,'rows','sorted'); % remove duplicates and sort rows
end

end

function configs = getMFUConfigurationsFromState(unit,state)
    sites = unit.sites; basis = unit.basis;
    sites = sites+ones(size(sites)); % indices 0->1 and on on
    [x, y] = size(state);
    
    nx = fix(x/max(basis(:,1)));
    ny = fix(y/max(basis(:,2)));
    
    configs = {};
    count = 0;
    for ax = 0:nx
        for ay = 0:ny
            trans = sites + ax*basis(1,:) + ay*basis(2,:);
            if all(trans(:)>0) && all(trans(:,1)<= x) && all(trans(:,2)<=y)
                count = count + 1;
                configs{count} = getElements(trans,state);
            end
        end
    end
    configs = cell2mat(configs)'; % turn such that the rows are the configurations
    configs = [configs; - configs]; % add all the flipped states
    configs = unique(configs,'rows'); % remove duplicates
end

