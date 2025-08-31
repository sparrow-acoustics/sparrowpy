%% RAVEN simulation: Single triangular-profile façade scene (VI.A)

% Based on the script "Example for creating a room model specifying points" 
% and planes 
% Author: eac@akustik.rwth-aachen.de
% date:     2020/01/07
%
% <ITA-Toolbox>
% This file is part of the application Raven for the ITA-Toolbox. All rights reserved.
% You can find the license for this m-file in the application folder.
% </ITA-Toolbox>

%% project settings
projectName = 'facade';

% set room dimensions (facade is 9x19)
% ray-tracing simulation -> model must be a closed room
h1=-19; h2 = -30;
w1=9; w2 =15;
l = 30;

% Example shape: Fan shape
points=[[0, 0, 0]; [0, 0, h1]; [w1, 0, h1];[w1, 0, 0]; ...
    [w2, l, 0]; [w2, l, h2];[0, l, h2]; [0, l, 0]];
faces={...
    [1 4 3 2 1];...
    [2 8 7 6 5];...
    [2 1 2 7 8];...
    [2 3 4 5 6];...
    [2 2 3 6 7]; ...
    [2 1 8 5 4]...
    };


%% Create project and set input data (based on shoebox model)
rpf = itaRavenProject('C:\ITASoftware\Raven\RavenInput\Classroom\Classroom.rpf');   % modify path if not installed in default directory
rpf.copyProjectToNewRPFFile(['C:\ITASoftware\Raven\RavenInput\' projectName '.rpf' ]);
rpf.setProjectName(projectName);

% set façade properties
% scattering coefficient [0.05898381, 0.2288077 , 0.63353846, 0.78131131, 0.906805, 0.94444149]
myAbsorpWall = 0.07 * ones(1,10);
myScatterWall = 1 * ones(1,10);
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

% set maximum absorption in remaining surfaces to emulate open geometries
myAbsorp = 1 * ones(1,10);
myScatter = 0 * ones(1,10);

% set model geometry and S-R positions
materials={'retro_ahe';'absoptive_ahe';};
rpf.setModelToFaces(points,faces,materials)
rpf.setSourcePositions([1.5 10, -19.5])
rpf.setSourceDirectivity('Omnidirectional')
rpf.setReceiverPositions([1.5 11, -20.5])
nM=6;

% assign material properties to different model walls
for iMat=2:nM
    % virtual walls (perf. absorptive)
    rpf.setMaterial(rpf.getRoomMaterialNames{iMat},myAbsorp,myScatter);
end
% façade
rpf.setMaterial(rpf.getRoomMaterialNames{1},myAbsorpWall,myScatterWall);


%% Check: Plane normals should point to the inner side of the room
rpf.model.plotModel([], [1 2 3], 0,1)
axis equal
%% set simulation parameters
rpf.setGenerateRIR(1);
rpf.setGenerateBRIR(1);
rpf.setSimulationTypeRT(1);
rpf.setSimulationTypeIS(1);
rpf.setNumParticles(5000000);
rpf.setFilterLength(400);
rpf.setTimeSlotLength(1); % ms
rpf.setISOrder_PS(0);
rpf.disableAirAbsorption

%% run simulation
rpf.run

% plot model
rpf.plotModel;

%% 
% get and plot see ETC (called "histogram" in the RAVEN context)
hist = rpf.getHistogram_itaResult;
hist.ptd

%% export ETC ("histogram") data
histogram = hist.time;
disp(['max histogram ' num2str(max(histogram))])
size(histogram)
histogram = [hist.timeVector histogram];
histogram = [[0, rpf.freqVectorOct]; histogram];
size(histogram)
% write freq-wise ETC in file
writematrix(histogram, '../resources/user/raven_facade.csv');

