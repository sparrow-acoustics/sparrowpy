function run_simu(sceneID)
    if nargin==0
        sceneID='ihtapark';
    end
    
    sr=500;
    switch sceneID
        case 'ihtapark'
            receiverID=5;
            sourceID=1;
            duration = 1200;
            src=[];
            rec=[];
            nParticles = 2000000;
    
        case 'seminar'
            receiverID=5;
            sourceID=1;
            duration = 1200;
            src=[];
            rec=[];
            nParticles = 50000;
    end
    
    
    clear rpf
    rpf=itaRavenProject(strcat(sceneID).rpf');
    
    receiverID=5;
    sourceID=1;
    
    %% setting up the simulation
    
    rpf.setEnergyLoss(80)
    rpf.setFilterLength(duration);
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
    
    if isempty(src) && sourceID~=0
        positions = rpf.getSourcePosition();
        scene_data.source.position = positions(sourceID,:);
        %scene_data.source.up = rpf.getSourceUpVectors(0);
        %scene_data.source.view = rpf.getSourceViewVectors(0);
    elseif ~isempty(src) && length(src)==3
        rpf.setSourcePositions(src)
        positions = rpf.getSourcePosition();
        scene_data.source.position = positions(1,:);
    else
        error(strcat(['source not defined for scene ' sceneID]))
    end
    
    if isempty(rec) && receiverID~=0
        positions = rpf.getReceiverPosition();
        scene_data.receiver.position = positions(receiverID,:);
        %scene_data.receiver.up = rpf.getReceiverUpVectors(0);
        %scene_data.receiver.view = rpf.getReceiverViewVectors(0);
    elseif ~isempty(rec) && length(rec)==3
        rpf.setReceiverPositions(rec)
        positions = rpf.getReceiverPosition();
        scene_data.receiver.position = positions(1,:);
    else
        error(strcat(['receiver not defined for scene ' sceneID]))
    end
    
    for material = material_list
        [a,s] = rpf.getMaterial(char(material));
        scene_data.materials.(material).absorption = a;
        scene_data.materials.(material).scattering = s;
    end
    
    rpf.setNumParticles(nParticles)
    rpf.setTimeSlotLength(1000/sr)
    %% run
    tic
    rpf.run();
    runtime = [runtime toc];
    %% check
    h = rpf.getHistogram_itaResult();
    curves{end+1} = h(sourceID,receiverID).time;
    RT30 = [RT30 rpf.getT30(0,1)];
    resolution = [resolution nParticles];
    step_size = [step_size 1/sr];
    
    scene_data.etc_step = rpf.timeSlotLength / 1000;
    scene_data.etc_duration = rpf.filterLength / 1000;
    
    save("..\\examples\\out\\ihtapark_raven.mat", ...
        'curves',...
        "RT30",...
        "runtime",...
        "resolution","step_size")
    
    material_data_json = jsonencode(scene_data, PrettyPrint=true);
    fid = fopen('..\\examples\\resources\\ihtapark_scene.json','w');
    fprintf(fid,'%s',material_data_json);
    fclose(fid);

end