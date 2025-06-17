clear rpf
rpf=itaRavenProject('diffuse_room.rpf');

%% setting up the scene (materials are loaded already)
rpf.setSourceNames('S');
rpf.setSourcePositions([2,2,-2]);
rpf.setSourceDirectivity('Omnidirectional.daff');

rpf.setReceiverNames('R');
rpf.setReceiverPositions([2,3,-2]);
rpf.setReceiverHRTF('Receiver_IR_2ch_omni_ds31_30x30.daff');


rpf.setTemperature(20)
%% setting up the simulation
rpf.setNumParticles(20000)
rpf.setEnergyLoss(60)
rpf.setFilterLength(1200);
rpf.setGenerateRIR(1);
rpf.setExportHistogram(1);
rpf.setExportFilter(1);
rpf.setISOrder_PS(0)


%% run
rpf.run();

%% check
mono_ir_ita = rpf.getMonauralImpulseResponseItaAudio();
hist = rpf.getHistogram_itaResult();
hist.plot_time_dB;
T60 = rpf.getT60()
