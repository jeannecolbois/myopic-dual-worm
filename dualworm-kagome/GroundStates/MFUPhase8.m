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


unit.sites=sites; unit.basis=basis;


foldername = "./Phase8Families_Clusters/Results/";

[MFUFamConfigs, MFUConfigs] = RequiredConfigurations(unit,foldername);

MFUFamConfigsClusters = MFUFamConfigs;
MFUConfigsClusters = MFUConfigs;

save("Phase8MFUfromMC_Clusters.mat", 'unit','foldername','MFUFamConfigs','MFUConfigs');

foldername = "./Phase8Families/Results/";

[MFUFamConfigs, MFUConfigs] = RequiredConfigurations(unit,foldername);

save("Phase8MFUfromMC.mat", 'unit','foldername','MFUFamConfigs','MFUConfigs');