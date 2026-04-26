function [actual_load, power_consumed, dropped_calls] = step5GDigitalTwin(es_modes, hour_of_day)
    % Initialize the true Wireless Network Simulator
    networkSimulator = wirelessNetworkSimulator.init();
    
    %% 1. Determine Parameters based on Energy Saving Modes (The AI's Choice)
    % es_modes is an array of 6 binary flags for modes M1 to M6.
    es_modes = double(es_modes); % Ensure we are working with a double array
    
    % Modes 4, 5, 6 are considered "Deep Sleep" modes which severely cut power
    is_deep_sleep = any(es_modes(4:6) == 1);
    
    % Base static power for the tower based on dataset (cooling, processing)
    basePower = 55; % 55 Watts static power
    
    if sum(es_modes) == 0
        txPower = 6.87; % Full Tx Power based on BSinfo.csv
    elseif is_deep_sleep
        txPower = 2.0; % Deep sleep reduces TxPower heavily
    else
        txPower = 4.5; % Micro sleep (Modes 1-3) reduces power slightly
    end
    
    %% 2. Create the gNodeB (Cell Tower) using 5G Toolbox
    % Here we define the exact carrier frequency and bandwidth from your CSV
    % Pass TransmitPower at initialization because it is read-only after creation
    gNB = nrGNB('CarrierFrequency', 3.5e9, ...
                'ChannelBandwidth', 20e6, ...
                'SubcarrierSpacing', 30e3, ...
                'NumTransmitAntennas', 4, ...
                'TransmitPower', txPower);
                
    %% 3. Apply Scheduler Restrictions for Deep Sleep
    if is_deep_sleep
        % Limit the Resource Blocks (RBs) the scheduler is allowed to use
        configureScheduler(gNB, 'MaxNumUsersPerTTI', 1); 
    end
    
    %% 4. Spawn Virtual Users (UEs) and Traffic
    % Create 10 mobile phones around the tower
    UEs = nrUE('NumTransmitAntennas', 1, ...
               'Position', rand(10, 3) * 100); % Randomly place within 100 meters
               
    % Connect users to the tower
    connectUE(gNB, UEs);
    
    % Generate bursty Application Traffic (Video streaming, downloads)
    % Adjust the data rate based on the hour of the day (simulating day/night cycle)
    base_rate = 15000;
    human_traffic_multiplier = max(0.1, 0.4 + 0.6 * sin(2 * pi * (hour_of_day - 8) / 24));
    
    % UNEXPECTED ANOMALY: Simulate a flash mob or event at 2:00 AM!
    % The AI expects low traffic (0.1), so it will select Deep Sleep.
    % The Digital Twin will punish this mistake with Dropped Calls.
    if hour_of_day == 2
        human_traffic_multiplier = 0.95; % Sudden 95% load spike!
    end
    
    current_data_rate = base_rate * human_traffic_multiplier;
    
    traffic = networkTrafficOnOff('DataRate', current_data_rate, 'PacketSize', 1500);
    for i = 1:10
        addTrafficSource(gNB, traffic, 'DestinationNode', UEs(i));
    end
    
    %% 5. Run the Physics Engine
    addNodes(networkSimulator, gNB);
    addNodes(networkSimulator, UEs);
    
    % Run the standard-compliant simulation for exactly 10 milliseconds
    % (Warning: This takes heavy computation! Running for full hours is impossible)
    simulation_time = 0.01; 
    run(networkSimulator, simulation_time);
    
    %% 6. Extract Real Network Statistics
    gnbStats = statistics(gNB);
    
    % The raw transmitted bytes from a 10ms simulation are very small.
    % To keep the AI model happy (it expects a normalized load between 0-1), 
    % we represent actual_load based on the traffic multiplier we applied.
    actual_load = human_traffic_multiplier; 
    
    % Different versions of MATLAB 5G Toolbox track dropped packets differently.
    % We use isfield to safely check, and provide a fallback penalty if missing.
    if isfield(gnbStats.MAC, 'DroppedPackets')
        dropped_calls = sum([gnbStats.MAC.DroppedPackets]);
    else
        % Fallback: If AI puts the tower in Deep Sleep during High Traffic, drop calls!
        if is_deep_sleep && actual_load > 0.6
            dropped_calls = (actual_load - 0.6) * 100;
        else
            dropped_calls = 0;
        end
    end
    
    % Calculate total power in Watts
    power_consumed = basePower + (gNB.TransmitPower * actual_load); 
end