clear rpf
rpf=itaRavenProject('scene9.rpf');

%% setting up the scene (materials are loaded already)
rpf.setSourceNames('S');
rpf.setSourcePositions([.119,2.880,1.203]);
rpf.setSourceDirectivity('Omnidirectional.daff');

rpf.setReceiverNames('R');
rpf.setReceiverPositions([.439,-.147,1.230]);
rpf.setReceiverHRTF('Receiver_IR_2ch_omni_ds31_30x30.daff');


rpf.setTemperature(19.5)
rpf.setHumidity(41.7)
rpf.setPressure(101325)
%% setting up the simulation

rpf.setEnergyLoss(80)
rpf.setFilterLength(2000);
rpf.setExportHistogram(1);
rpf.setExportFilter(1);
rpf.setISOrder_PS(0);

RT30 = [];
curves = {};
runtime = [];
step_size =[];
resolution = [];

material_list = convertCharsToStrings(rpf.getRoomMaterialNames());

scene_data=struct();
scene_data.f = rpf.freqVector3rd;
scene_data.T = rpf.getTemperature();
scene_data.H = rpf.getHumidity();
scene_data.P = rpf.getPressure();
scene_data.air_att = determineAirAbsorptionParameter(rpf.getTemperature(), ...
                                                    rpf.getPressure(), ...
                                                    rpf.getHumidity());
scene_data.sound_speed = rpf.getSoundSpeed();

scene_data.source.position = rpf.getSourcePosition(0);
%scene_data.source.up = rpf.getSourceUpVectors(0);
%scene_data.source.view = rpf.getSourceViewVectors(0);

scene_data.receiver.position = rpf.getReceiverPosition(0);
%scene_data.receiver.up = rpf.getReceiverUpVectors(0);
%scene_data.receiver.view = rpf.getReceiverViewVectors(0);

for material = material_list
    [a,s] = rpf.getMaterial(char(material));
    scene_data.materials.(material).absorption = a;
    scene_data.materials.(material).scattering = s;
end


sr = 500;

rpf.setNumParticles(nParticles)
rpf.setTimeSlotLength(1000/sr)
%% run
tic
rpf.run();
runtime = [runtime toc];
%% check
h = rpf.getHistogram_itaResult();
curves{end+1} = h.time;
RT30 = [RT30 rpf.getT30(0,1)];
resolution = [resolution nParticles];
step_size = [step_size 1/step];

scene_data.etc_step = rpf.timeSlotLength / 1000;
scene_data.etc_duration = rpf.filterLength / 1000;

save("..\\examples\\out\\seminar_room_raven.mat", ...
    'curves',...
    "RT30",...
    "runtime",...
    "resolution","step_size")

material_data_json = jsonencode(scene_data, PrettyPrint=true);
fid = fopen('..\\examples\\resources\\seminar_scene.json','w');
fprintf(fid,'%s',material_data_json);
fclose(fid);