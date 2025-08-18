%% RAVEN simulation: Example for creating shoebox room model

% Author: las@akustik.rwth-aachen.de
% date:     2019/04/10
%
% <ITA-Toolbox>
% This file is part of the application Raven for the ITA-Toolbox. All rights reserved.
% You can find the license for this m-file in the application folder.
% </ITA-Toolbox>


%% project settings

myLength=90;
myWidth=12;
myHeight=6;
projectName = [ 'myShoeboxRoom' num2str(myLength) 'x' num2str(myWidth) 'x' num2str(myHeight) ];

%% create project and set input data
rpf = itaRavenProject('C:\ITASoftware\Raven\RavenInput\Classroom\Classroom.rpf');   % modify path if not installed in default directory
rpf.copyProjectToNewRPFFile(['C:\ITASoftware\Raven\RavenInput\' projectName '.rpf' ]);
rpf.setProjectName(projectName);
rpf.setModelToShoebox(myLength,myWidth,myHeight);

% set values of six surfaces:
% 10% absorption and 10% scattering for floor and ceiling
% Identical material with 5% absorption and 20% scattering for walls

% surface order:
% (1) floor, (2) ceiling,
% (3) larger wall (length x height; left, view from origin)
% (4) smaller wall (width x height; front)
% (5) larger wall (length x height; right)
% (6) smaller wall (width x height; back)
%

myAbsorpGround = 0.01 * ones(1,10);
myScatterGround = 0.1 * ones(1,10);

myAbsorpWall = 0.07 * ones(1,10);
myScatterWall = 0 * ones(1,10);
myAbsorpWall(1) = 1;
myAbsorpWall(2) = 1;
myScatterWall(3) = 0.05898381;
myScatterWall(4) = 0.2288077;
myScatterWall(5) = 0.63353846;
myScatterWall(6) = 0.78131131;
myScatterWall(7) = 0.906805;
myScatterWall(8) = 0.94444149;
myAbsorpWall(9) = 1;
myAbsorpWall(10) = 1;

myAbsorp1 = 1 * ones(1,10);
myScatter1 = 0 * ones(1,10);

rpf.setMaterial(rpf.getRoomMaterialNames{1},myAbsorpGround,myScatterGround);
rpf.setMaterial(rpf.getRoomMaterialNames{2},myAbsorp1,myScatter1);
rpf.setMaterial(rpf.getRoomMaterialNames{3},myAbsorpWall,myScatterWall);
rpf.setMaterial(rpf.getRoomMaterialNames{4},myAbsorp1,myScatter1);
rpf.setMaterial(rpf.getRoomMaterialNames{5},myAbsorpWall,myScatterWall);
rpf.setMaterial(rpf.getRoomMaterialNames{6},myAbsorp1,myScatter1);



rpf.setSourcePositions([20, 1, -6]);
rpf.setSourceViewVectors([ 1     0     0]);
rpf.setSourceUpVectors([ 0     1    0]);
rpf.setSourceDirectivity('Omnidirectional');


rpf.setReceiverPositions([70, 2, -6]);
rpf.setReceiverUpVectors([0 1 0]);
rpf.setReceiverViewVectors([1 0 0]);
% rpf.setReceiverHRTF([ ravenBasePath 'RavenDatabase\HRTF\ITA-Kunstkopf_HRIR_AP11_Pressure_Equalized_3x3_256.daff']);


% uncomment to see plot of room and absorption coefficient
% rpf.plotMaterialsAbsorption;
rpf.plotModel;

%% set simulation parameters
rpf.setGenerateRIR(1);
rpf.setGenerateBRIR(1);
rpf.setSimulationTypeRT(1);
rpf.setSimulationTypeIS(1);
rpf.setNumParticles(5000000);
rpf.setFilterLength(900);
rpf.setTimeSlotLength(2); % ms

rpf.setISOrder_PS(0);
rpf.disableAirAbsorption

%% run simulation
rpf.run;



%%

hist = rpf.getHistogram_itaResult;
hist.ptd

%%
histogram = hist.time;

disp(['max histogram ' num2str(max(histogram))])
size(histogram)
histogram = [hist.timeVector histogram];
histogram = [[0, rpf.freqVectorOct]; histogram];
size(histogram)
writematrix(histogram, 'out/raven_streetcanyon.csv');


%%
