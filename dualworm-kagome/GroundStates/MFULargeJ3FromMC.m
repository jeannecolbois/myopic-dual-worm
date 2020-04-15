clear all;
close all;
% 
% sites=[1   0;
%        0   0;
%        0   1;
%        -1  0;
%        0  -1;
%        0   2;
%        1   2;
%        2   1;
%        2   0;
%        2   2;
%        2   3;
%        3   2];
% basis=[2 0;
%        0 2];
%    
% unit.sites=sites; unit.basis=basis;

unit.sites = [[0,0];[1,0];[2,0];[0,1];[0,2];[2,1];[1,2];[2,2]];
unit.basis = [[2,0];[0,2]];

foldername = "./J1J3Families3/Results/";

[MFUFamConfigs, MFUConfigs] = RequiredConfigurations(unit,foldername);

save("WrongJ1J3MFU.mat", 'unit','foldername','MFUFamConfigs','MFUConfigs');