%% RAVEN simulation: Example for creating a room model specifying points 
% and planes 

% Author: eac@akustik.rwth-aachen.de
% date:     2020/01/07
%
% <ITA-Toolbox>
% This file is part of the application Raven for the ITA-Toolbox. All rights reserved.
% You can find the license for this m-file in the application folder.
% </ITA-Toolbox>

%% project settings
projectName = 'my_Fan_shape';

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
%     flip([4 3 2 1 1]);...
%     flip([8 7 6 5 2]);...
%     flip([1 2 7 8 2]);...
%     flip([3 4 5 6 2]);...
%     flip([2 3 6 7 2]); ...
% 
%     flip([1 8 5 4 2])...
    };

% for i=1:length(faces)
%     faces{i}(2:end)=fliplr(faces{i}(2:end));
% end

%% Create project and set input data
rpf = itaRavenProject('C:\ITASoftware\Raven\RavenInput\Classroom\Classroom.rpf');   % modify path if not installed in default directory
rpf.copyProjectToNewRPFFile(['C:\ITASoftware\Raven\RavenInput\' projectName '.rpf' ]);
rpf.setProjectName(projectName);

% source and receiver
hs=1.2; %source height
hr=1.2; %receiver height

myAbsorpWall = 0 * ones(1,10);
myScatterWall = 1 * ones(1,10);
myScatterWall(3) = 0.18646731;
myScatterWall(4) = 0.4809733;
myScatterWall(5) = 0.82650344;
myScatterWall(6) = 0.90453428;
myScatterWall(7) = 0.95312613;
myScatterWall(8) = 0.96773037;
myAbsorp = 1 * ones(1,10);
myScatter = 0 * ones(1,10);

% rpf.setMaterial('absoptive_ahe',myAbsorp,myScatter)
% 
% rpf.setMaterial('retro_ahe',myAbsorpWall,myScatterWall)
materials={'retro_ahe';'absoptive_ahe';};
rpf.setModelToFaces(points,faces,materials)
rpf.setSourcePositions([1.5 10, -19.5])
rpf.setSourceDirectivity('Omnidirectional')
rpf.setReceiverPositions([1.5 11, -20.5])
nM=6;

% Coefficients
for iMat=2:nM
    rpf.setMaterial(rpf.getRoomMaterialNames{iMat},myAbsorp,myScatter);
end
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
rpf.setFilterLength(200);
rpf.setTimeSlotLength(2); % ms
rpf.setISOrder_PS(0);
rpf.disableAirAbsorption

%% run simulation
rpf.run

%%

facade = itaCoordinates();
facade.cart =  [ [0, 0, 0]; [0, 0, h1]; [w1, 0, h1];[w1, 0, 0]];

source = itaCoordinates();
source.cart = [1.5 10, -19.5];
receiver = itaCoordinates();
receiver.cart = [1.5 11, -20.5];

diff_s = (source-facade);
diff_r = receiver-facade;
distance = diff_s.r+diff_r.r;
delays = distance/343;
direct = source-receiver;
direct.r/343
delays*1000
rpf.plotModel;

%% plot results

hist = rpf.getHistogram_itaResult;
hist.ptd
%%
histogram = hist.time;
size(histogram)
histogram = [hist.timeVector histogram];
histogram = [[0, rpf.freqVectorOct]; histogram];
size(histogram)
writematrix(histogram, 'raven_facade.csv');

