clear all;
close all;
 
sites=[1   0;
    0   0;
    0   1;
    -1  0;
    0  -1;
    0   2;
    1   2;
    2   1;
    2   0;
    2   2;
    2   3;
    3   2];

basis=[2 0;
    0 2];

% 9 stars   
% sites=[[-1 0];[-1 2];[-1 4];
% [0 -1];[0 0];[0 1];[0 2]; [0 3];[0 4];[0 5];[0 6];
% [1 0];[1 2];[1 4];[1 6];
% [2 -1];[2 0];[2 1];[2 2];[2 3];[2 4];[2 5];[2 6];[2 7];
% [3 0];[3 2];[3 4];[3 6];
% [4 -1];[4 0];[4 1];[4 2];[4 3];[4 4];[4 5];[4 6];[4 7];
% [5 2];[5 4];[5 6];
% [6 -1];[6 0];[6 1];[6 2];[6 3];[6 4];[6 5];[6 6];[6 7];
% [7 2];[7 4];[7 6]
% ];
% basis = [4 0;0 4];

unit.sites=sites; unit.basis=basis;


foldername = "./LargeJ3FamiliesR/Results/";

[MFUFamConfigs, MFUConfigs] = RequiredConfigurations(unit,foldername);

save("LargeJ3MFUfromMCR.mat", 'unit','foldername','MFUFamConfigs','MFUConfigs');