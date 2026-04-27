function [actual_load, power_consumed, dropped_calls] = step5GDigitalTwin(es_modes, hour_of_day)
    networkSimulator = wirelessNetworkSimulator.init();
    
    es_modes = double(es_modes);
    
    is_deep_sleep = any(es_modes(4:6) == 1);
    
    basePower = 55;
    
    if sum(es_modes) == 0
        txPower = 6.87;
    elseif is_deep_sleep
        txPower = 2.0;
    else
        txPower = 4.5;
    end
    
    gNB = nrGNB('CarrierFrequency', 3.5e9, ...
                'ChannelBandwidth', 20e6, ...
                'SubcarrierSpacing', 30e3, ...
                'NumTransmitAntennas', 4, ...
                'TransmitPower', txPower);
                
    if is_deep_sleep
        configureScheduler(gNB, 'MaxNumUsersPerTTI', 1); 
    end
    
    UEs = nrUE('NumTransmitAntennas', 1, ...
               'Position', rand(10, 3) * 100);
               
    connectUE(gNB, UEs);
    
    base_rate = 15000;
    human_traffic_multiplier = max(0.1, 0.4 + 0.6 * sin(2 * pi * (hour_of_day - 8) / 24));
    
    if hour_of_day == 2
        human_traffic_multiplier = 0.95;
    end
    
    current_data_rate = base_rate * human_traffic_multiplier;
    
    traffic = networkTrafficOnOff('DataRate', current_data_rate, 'PacketSize', 1500);
    for i = 1:10
        addTrafficSource(gNB, traffic, 'DestinationNode', UEs(i));
    end
    
    addNodes(networkSimulator, gNB);
    addNodes(networkSimulator, UEs);
    
    simulation_time = 0.01; 
    run(networkSimulator, simulation_time);
    
    gnbStats = statistics(gNB);
    
    actual_load = human_traffic_multiplier; 
    
    if isfield(gnbStats.MAC, 'DroppedPackets')
        dropped_calls = sum([gnbStats.MAC.DroppedPackets]);
    else
        if is_deep_sleep && actual_load > 0.6
            dropped_calls = (actual_load - 0.6) * 100;
        else
            dropped_calls = 0;
        end
    end
    
    power_consumed = basePower + (gNB.TransmitPower * actual_load); 
end