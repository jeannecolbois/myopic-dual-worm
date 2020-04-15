%function [configurations] = RequiredConfigurations(unit, foldername)
% [configurations] = RequiredConfigurations(unit, foldername)
% For a unit cell "unit" and a list of ground state configurations "foldername"
% this function returns the configurations on the unit cell which are
% expected to be in the ground state of a proper MFU hamiltonian on this unit cell

% 1 - Load the folder family by family
listing = dir("./IntermediateFamilyStates/Results/Family*"); % list all states

% 2 - For each family, find all the unit cell configurations which are allowed
% (given the basis)

% 3 - Check that the families are different


%end
