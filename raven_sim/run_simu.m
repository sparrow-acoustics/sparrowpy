function run_simu(sceneID)
    if nargin==0
        sceneID='ihtapark';
    end
    
    
    switch sceneID
        case 'ihtapark'
            receiverID=5;
            sourceID=1;
            duration = 1200;
            src=[];
            rec=[];
            nParticles = 2000000;
            sr = 500;
    
        case 'seminar'
            receiverID=1;
            sourceID=1;
            duration = 1200;
            src=[.119,2.880,1.203];
            rec=[.439,-.147,1.230];
            nParticles = 50000;
            sr = 500;
           
        case 'diffuse_room'
            receiverID=1;
            sourceID=1;
            duration = 1200;
            src=[2,2,2];
            rec=[2,3,2];
            nParticles = [50 100 500 1000 5000 10000 50000 100000];
            sr = [50 100 500 1000 5000];
    end
    
    
    clear rpf
    rpf=itaRavenProject(strcat(sceneID,'.rpf'));

    
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


    if ~isempty(src) && length(src)==3
            rpf.setSourcePositions(src);
            sourceID=1;
    end
    if ~isempty(rec) && length(rec)==3
            rpf.setReceiverPositions(rec);
            receiverID=1;
    end

    
    
    for n = nParticles
        for sampling = sr
            rpf.setNumParticles(n)
            rpf.setTimeSlotLength(1000/sampling)
            %% run
            tic
            rpf.run();
            runtime = [runtime toc];
            %% check
            h = rpf.getHistogram_itaResult();
            curves{end+1} = h(sourceID,receiverID).time;
            RT30 = [RT30 rpf.getT30(0,1)];
            resolution = [resolution nParticles];
            step_size = [step_size rpf.timeSlotLength / 1000];
        end
    end
    
    

    out.simu_output = struct('etc',curves, ...
                    'RT30',RT30, ...
                    'resolution',resolution, ...
                    'step_size',step_size, ...
                    'runtime',runtime);

    out.scene_data = compile_conditions(rpf,sourceID,receiverID);
    

    write_simu_conditions(out);

end

function [scene_data] = compile_conditions(raven_data, sourceID, receiverID)

    material_list = convertCharsToStrings(raven_data.getRoomMaterialNames());
    
    scene_data=struct();
    scene_data.f = raven_data.freqVector3rd;
    scene_data.T = raven_data.getTemperature();
    scene_data.H = raven_data.getHumidity();
    scene_data.P = raven_data.getPressure();
    scene_data.air_att = determineAirAbsorptionParameter(raven_data.getTemperature(), ...
                                                        raven_data.getPressure(), ...
                                                        raven_data.getHumidity());
    scene_data.sound_speed = raven_data.getSoundSpeed();
    
    positions = raven_data.getSourcePosition();
    scene_data.source.position = positions(sourceID,:);
    positions = raven_data.getReceiverPosition();
    scene_data.receiver.position = positions(receiverID,:);

    
    for material = material_list
        [a,s] = raven_data.getMaterial(char(material));
        scene_data.materials.(material).absorption = a;
        scene_data.materials.(material).scattering = s;
    end

    scene_data.etc_duration = raven_data.filterLength / 1000;

end

function write_simu_conditions(scene_data,scene_id,base_path)
    if nargin<3
        base_path = '..\\examples\\resources\\';
    end
    if nargin<2
        scene_id = 'ihtapark';
    end

    simu_data_json = jsonencode(scene_data, PrettyPrint=true);
    fid = fopen(strcat(base_path,scene_id,'_scene.json'),'w');
    fprintf(fid,'%s',simu_data_json);
    fclose(fid);
end

