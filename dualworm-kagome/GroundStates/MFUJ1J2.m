
unit.sites = [[0,0];[1,0];[2,0];[0,1];[0,2];[2,1];[1,2];[2,2]];
unit.basis = [[2,0];[0,2]];

foldername = "./J1J2Families/Results/";

[MFUFamConfigs, MFUConfigs] = RequiredConfigurations(unit,foldername);

save("J1J2MFU.mat", 'unit','foldername','MFUFamConfigs','MFUConfigs');