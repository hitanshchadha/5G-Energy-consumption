import sys
import os
import matlab.engine
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Force UTF-8 encoding for standard output to support emojis in Windows terminals
sys.stdout.reconfigure(encoding='utf-8')

# Import everything we need from LSTM.py
from LSTM import (
    EnergyPredictorAttention,
    recommend_es_mode,
    load_and_preprocess_data,
    TelecomDataset,
    train_model,
    evaluate_model
)
def start_oran_controller(model, le_bs, scaler, target_scaler):
    print("Starting MATLAB Engine... (This takes 10-15 seconds)")
    eng = matlab.engine.start_matlab()
    print("MATLAB Engine Connected! 5G Environment is live.\n")
    
    # Initialize variables
    target_bs = 'B_0'
    bs_encoded_id = le_bs.transform([target_bs])[0]
    
    # In a real scenario, this is our lookback window data
    # For simulation, we'll initialize a dummy 24-hour history array
    historical_dynamic = np.zeros((24, 11)) 
    static_data = np.array([4.0, 6.87, 20.0]) # Antennas, TXPower, Bandwidth
    
    current_es_modes = [0.0]*6 # Start with no sleep modes active
    
    # Arrays to track metrics for plotting
    history_hours = []
    history_load = []
    history_power = []
    history_dropped = []
    history_active_modes = []
    
    try:
        # Simulate a 24-hour run to hit the midday peak
        for hour in range(24):
            clock_hour = hour % 24
            print(f"{'='*40}")
            print(f"🕒 TIME: {clock_hour}:00")
            
            # ---------------------------------------------------------
            # 1. READ FROM PHYSICAL WORLD (MATLAB)
            # ---------------------------------------------------------
            # We pass the current ES modes (array) to MATLAB, and MATLAB returns 
            # the load it experienced, the power it used, and if any calls dropped.
            # nargout=3 tells Python to expect 3 return variables.
            actual_load, power_consumed, dropped_calls = eng.step5GDigitalTwin(
                matlab.double(current_es_modes), 
                float(clock_hour), 
                nargout=3
            )
            
            print(f"📡 [gNodeB Telemetry] Load: {actual_load*100:.1f}% | Power Used: {power_consumed:.2f} W")
            if dropped_calls > 0:
                print(f"⚠️  WARNING: {dropped_calls:.1f}% CALLS DROPPED! SLEEP MODE TOO DEEP!")

            # ---------------------------------------------------------
            # 2. AI OPTIMIZATION PHASE (PYTHON)
            # ---------------------------------------------------------
            # Update our historical window with the new reality
            scaled_actual_load = scaler.transform(np.array([[actual_load, 0, 0, 0]]))[0, 0]
            
            # Predict cyclical time features for the NEXT hour
            next_hour = (clock_hour + 1) % 24
            next_time_features = np.array([
                np.sin(2 * np.pi * next_hour / 24.0), np.cos(2 * np.pi * next_hour / 24.0),
                0.0, 1.0 # Dummy day of week
            ])
            
            print("🧠 [RIC Optimizer] Running AI Energy Fingerprint model...")
            
            # Call the recommender function we wrote earlier
            optimal_combo, expected_power = recommend_es_mode(
                model, bs_encoded_id, historical_dynamic, static_data, 
                scaled_actual_load, actual_load, target_scaler, next_time_features
            )
            
            # Convert binary combo to array of modes
            next_es_modes = list(optimal_combo)
            active_modes = [i+1 for i, m in enumerate(next_es_modes) if m == 1.0]
            
            if not active_modes:
                mode_str = "0 (No Sleep)"
            else:
                mode_str = ", ".join([str(m) for m in active_modes])
                
            print(f"⚙️  [RIC Decision] Sending Command -> ACTIVATE ES_MODES: [{mode_str}]")
            
            # Track history for graphing
            history_hours.append(clock_hour)
            history_load.append(actual_load * 100)
            history_power.append(power_consumed)
            history_dropped.append(dropped_calls)
            history_active_modes.append(max(active_modes) if active_modes else 0)
            
            # ---------------------------------------------------------
            # 3. APPLY ACTION FOR NEXT LOOP
            # ---------------------------------------------------------
            current_es_modes = next_es_modes
            
            # Shift history array and append newest data (Simplified)
            historical_dynamic = np.roll(historical_dynamic, -1, axis=0)
            
            # Wait 2 seconds so you can watch the simulation run in the terminal
            time.sleep(2) 
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        eng.quit()
        print("MATLAB Engine shut down.")
        
        # Draw the relevant graphs
        print("\n📊 Generating simulation performance graphs...")
        plt.figure(figsize=(12, 10))
        
        # 1. Load vs Dropped Calls
        plt.subplot(3, 1, 1)
        plt.plot(history_hours, history_load, label="Network Load (%)", color='blue', marker='o')
        plt.bar(history_hours, history_dropped, label="Dropped Calls (Penalty)", color='red', alpha=0.6)
        plt.title("24-Hour Network Load & Dropped Calls")
        plt.ylabel("Percentage / Penalty")
        plt.legend()
        plt.grid(True)
        
        # 2. Energy Consumption
        plt.subplot(3, 1, 2)
        plt.plot(history_hours, history_power, label="Actual Energy Consumed (W)", color='green', marker='s')
        plt.title("24-Hour Energy Consumption")
        plt.ylabel("Energy (W)")
        plt.legend()
        plt.grid(True)
        
        # 3. Sleep Modes
        plt.subplot(3, 1, 3)
        plt.step(history_hours, history_active_modes, label="Max Active ES Mode (0=Normal, 6=Deep Sleep)", color='purple', where='mid')
        plt.title("AI-Selected Energy Saving Mode Over Time")
        plt.xlabel("Hour of the Day")
        plt.ylabel("ES Mode (0-6)")
        plt.yticks(range(7))
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("simulation_results.png")
        print("✅ Graph saved as 'simulation_results.png'")

if __name__ == "__main__":
    print("--- STEP 1: LOAD DATA AND PREPARE MODEL ---")
    df_train, df_test, le_bs, scaler, target_scaler = load_and_preprocess_data()
    
    num_unique_bs = len(le_bs.classes_)
    trained_model = EnergyPredictorAttention(num_bs=num_unique_bs)
    
    model_path = "trained_model.pth"
    if os.path.exists(model_path):
        print(f"Loading previously trained model from {model_path}...")
        # Since the model might have been saved on a different device or same device, 
        # map_location='cpu' ensures it loads safely.
        trained_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        trained_model.eval()
        print("Model loaded successfully!\n")
    else:
        lookback = 24 
        train_dataset = TelecomDataset(df_train, lookback=lookback)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        print("Training model... (This will take a moment)")
        trained_model = train_model(trained_model, train_loader, epochs=10)
        
        # Save the trained weights for future use
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model Training Complete and saved to {model_path}!\n")
    
    print("--- STEP 1.5: EVALUATE MODEL PERFORMANCE ---")
    test_dataset = TelecomDataset(df_test, lookback=24)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    evaluate_model(trained_model, test_loader, target_scaler)
    
    print("\n--- STEP 2: START ORAN CONTROLLER ---")
    start_oran_controller(trained_model, le_bs, scaler, target_scaler)