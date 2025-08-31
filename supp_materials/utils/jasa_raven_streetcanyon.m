%% RAVEN simulation: Street canyon scene with triangular-profile walls (VI.B)

% Based on the script "Example for creating shoebox room model"
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
projectName = [ 'street_canyon' num2str(myLength) 'x' num2str(myWidth) 'x' num2str(myHeight) ];

%% create project and set input data (based on shoebox model)
rpf = itaRavenProject('C:\ITASoftware\Raven\RavenInput\Classroom\Classroom.rpf');   % modify path if not installed in default directory
rpf.copyProjectToNewRPFFile(['C:\ITASoftware\Raven\RavenInput\' projectName '.rpf' ]);
rpf.setProjectName(projectName);
rpf.setModelToShoebox(myLength,myWidth,myHeight);

% material properties of ground surface
myAbsorpGround = 0.01 * ones(1,10);
myScatterGround = 0.1 * ones(1,10);

% material properties of the triangular profile walls
myAbsorpWall = 0.07 * ones(1,10);
myScatterWall = 0 * ones(1,10);
myAbsorpWall(1) = 1;
myAbsorpWall(2) = 1;
myScatterWall(3) = 0.18646731;
myScatterWall(4) = 0.4809733;
myScatterWall(5) = 0.82650344;
myScatterWall(6) = 0.90453428;
myScatterWall(7) = 0.95312613;
myScatterWall(8) = 0.96773037;
myAbsorpWall(9) = 1;
myAbsorpWall(10) = 1;

% material properties of the virtual surfaces
% (perf. absorptive to emulate open regions)
myAbsorp1 = 1 * ones(1,10);
myScatter1 = 0 * ones(1,10);

% assign material properties to respective walls
rpf.setMaterial(rpf.getRoomMaterialNames{1},myAbsorpGround,myScatterGround);
rpf.setMaterial(rpf.getRoomMaterialNames{2},myAbsorp1,myScatter1);
rpf.setMaterial(rpf.getRoomMaterialNames{3},myAbsorpWall,myScatterWall);
rpf.setMaterial(rpf.getRoomMaterialNames{4},myAbsorp1,myScatter1);
rpf.setMaterial(rpf.getRoomMaterialNames{5},myAbsorpWall,myScatterWall);
rpf.setMaterial(rpf.getRoomMaterialNames{6},myAbsorp1,myScatter1);

% set source position
rpf.setSourcePositions([20, 1, -6]);
rpf.setSourceViewVectors([ 1     0     0]);
rpf.setSourceUpVectors([ 0     1    0]);
rpf.setSourceDirectivity('Omnidirectional');

% set receiver position
rpf.setReceiverPositions([21, 2, -6]);
rpf.setReceiverUpVectors([0 1 0]);
rpf.setReceiverViewVectors([1 0 0]);


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

rpf.setISOrder_PS(0); % direct sound
rpf.disableAirAbsorption

%% run simulation
rpf.run;

%%
% get and plot ETC (called "histogram" in the RAVEN context)
hist = rpf.getHistogram_itaResult;
hist.ptd

%% export ETC ("histogram) data
histogram = hist.time;

disp(['max histogram ' num2str(max(histogram))])
size(histogram)
histogram = [hist.timeVector histogram];
histogram = [[0, rpf.freqVectorOct]; histogram];
size(histogram)
writematrix(histogram, '../resources/user/raven_streetcanyon_retro.csv');

