clear rpf
rpf=itaRavenProject('diffuse_room.rpf');

%% setting up the scene (materials are loaded already)
rpf.setSourceNames('S');
rpf.setSourcePositions([2,2,-2]);
rpf.setSourceDirectivity('C:\\ITASoftware\\Raven\\RavenDatabase\\HRTF\\Omnidirectional.daff');

rpf.setReceiverNames('R');
rpf.setReceiverPositions([2,3,-2]);
rpf.setReceiverHRTF('C:\\ITASoftware\\Raven\\RavenDatabase\\HRTF\\Receiver_IR_2ch_omni_ds31_30x30.daff');


rpf.setTemperature(20)
rpf.setHumidity(50)
rpf.setPressure(101325)
%% setting up the simulation

rpf.setEnergyLoss(80)
rpf.setFilterLength(1200);
rpf.setGenerateRIR(1);
rpf.setExportHistogram(1);
rpf.setExportFilter(1);
rpf.setISOrder_PS(0);

RT30 = [];
curves = {};
runtimes = [];
step_size =[];
resolution = [];

for step = [50 100 500 1000 5000 10000]
for nParticles = [100 500 1000 5000 10000 50000]

rpf.setNumParticles(nParticles)
rpf.setTimeSlotLength(1000/step)
%% run
tic
rpf.run();
runtimes = [runtimes toc];
%% check
%mono_ir_ita = rpf.getMonauralImpulseResponseItaAudio();
h = rpf.getHistogram_itaResult();
curves{end+1} = h.time(:,6);
RT30 = [RT30 rpf.getT30()];
resolution = [resolution nParticles];
step_size = [step_size 1/step];

end

end

save("..\\examples\\out\\diffuse_room_raven.mat", ...
    'curves',...
    "RT30",...
    "runtimes",...
    "resolution","step_size")