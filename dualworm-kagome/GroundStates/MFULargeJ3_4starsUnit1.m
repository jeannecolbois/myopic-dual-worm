clear all;
close all;
 
sites=[[-1 0];[-1 2];
    [0 -1];[0 0];[0 1];[0 2]; [0 3];[0 4];
    [1 0];[1 2];[1 4];
    [2 -1];[2 0];[2 1];[2 2];[2 3];[2 4];[2 5];
    [3 0];[3 2];[3 4];
    [4 -1];[4 0];[4 1];[4 2];[4 3];[4 4];[4 5];
    [5 2];[5 4]];

basis=[4 0;
    0 4];
unit.sites=sites; unit.basis=basis;


foldername = "./LargeJ3Families/Results/";

[MFUFamConfigs, MFUConfigs] = RequiredConfigurations(unit,foldername);

save("LargeJ3MFUfromMC_4stars2.mat", 'unit','foldername','MFUFamConfigs','MFUConfigs');