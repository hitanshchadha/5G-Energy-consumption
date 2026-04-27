import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from huggingface_hub import login, hf_hub_download
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_and_preprocess_data():
    print("Loading dataset from Hugging Face...")
    login(token="")
    
    print("Downloading dataset...")
    cl_path = hf_hub_download(repo_id="netop/5G-Network-Energy-Consumption", filename="CLstat.csv", repo_type="dataset")
    cl_data = pd.read_csv(cl_path)
    
    bs_path = hf_hub_download(repo_id="netop/5G-Network-Energy-Consumption", filename="BSinfo.csv", repo_type="dataset")
    bs_info = pd.read_csv(bs_path)
    
    ec_path = hf_hub_download(repo_id="netop/5G-Network-Energy-Consumption", filename="ECstat.csv", repo_type="dataset")
    ec_data = pd.read_csv(ec_path)

    print("Merging datasets...")
    merged = pd.merge(cl_data, bs_info, on=['BS', 'CellName'], how='left')
    
    agg_funcs = {
        'load': 'mean',
        'ESMode1': 'max', 'ESMode2': 'max', 'ESMode3': 'max', 
        'ESMode4': 'max', 'ESMode5': 'max', 'ESMode6': 'max',
        'Antennas': 'sum', 'TXpower': 'sum', 'Bandwidth': 'mean'
    }
    bs_level_data = merged.groupby(['Time', 'BS']).agg(agg_funcs).reset_index()
    final_data = pd.merge(bs_level_data, ec_data, on=['Time', 'BS'], how='inner')
    
    final_data['Time'] = pd.to_datetime(final_data['Time'])
    final_data = final_data.sort_values(by=['BS', 'Time']).reset_index(drop=True)
    
    final_data['Hour'] = final_data['Time'].dt.hour
    final_data['DayOfWeek'] = final_data['Time'].dt.dayofweek
    
    final_data['Hour_sin'] = np.sin(2 * np.pi * final_data['Hour'] / 24.0)
    final_data['Hour_cos'] = np.cos(2 * np.pi * final_data['Hour'] / 24.0)
    final_data['Day_sin'] = np.sin(2 * np.pi * final_data['DayOfWeek'] / 7.0)
    final_data['Day_cos'] = np.cos(2 * np.pi * final_data['DayOfWeek'] / 7.0)
    
    le = LabelEncoder()
    final_data['BS_encoded'] = le.fit_transform(final_data['BS'])
    final_data['raw_load'] = final_data['load'] 
    
    scaler = StandardScaler()
    continuous_cols = ['load', 'Antennas', 'TXpower', 'Bandwidth']
    final_data[continuous_cols] = scaler.fit_transform(final_data[continuous_cols])
    
    target_scaler = StandardScaler()
    final_data['Energy_scaled'] = target_scaler.fit_transform(final_data[['Energy']])
    
    train_data, test_data = [], []
    for bs, group in final_data.groupby('BS'):
        split_idx = int(len(group) * 0.8) 
        train_data.append(group.iloc[:split_idx])
        test_data.append(group.iloc[split_idx:])
        
    df_train = pd.concat(train_data).reset_index(drop=True)
    df_test = pd.concat(test_data).reset_index(drop=True)
    
    return df_train, df_test, le, scaler, target_scaler

class TelecomDataset(Dataset):
    def __init__(self, df, lookback=24):
        self.X_dynamic, self.X_static, self.bs_ids, self.y = [], [], [], []
        
        feature_cols = [
            'load', 'ESMode1', 'ESMode2', 'ESMode3', 'ESMode4', 'ESMode5', 'ESMode6',
            'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos'
        ]
        
        for bs, group in df.groupby('BS_encoded'):
            values = group[feature_cols].values
            static_values = group[['Antennas', 'TXpower', 'Bandwidth']].values
            target = group['Energy_scaled'].values
            
            for i in range(len(values) - lookback):
                self.X_dynamic.append(values[i:i+lookback]) 
                self.X_static.append(static_values[i+lookback])
                self.bs_ids.append(bs)
                self.y.append(target[i+lookback])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X_dynamic[idx], dtype=torch.float32),
                torch.tensor(self.X_static[idx], dtype=torch.float32),
                torch.tensor(self.bs_ids[idx], dtype=torch.long),
                torch.tensor(self.y[idx], dtype=torch.float32))


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, lstm_output):
        attn_weights = self.attention(lstm_output) 
        attn_weights = torch.softmax(attn_weights, dim=1) 
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        return context_vector

class EnergyPredictorAttention(nn.Module):
    def __init__(self, num_bs, embedding_dim=16, dynamic_input_size=11, static_input_size=3, hidden_size=64):
        super(EnergyPredictorAttention, self).__init__()
        self.bs_embedding = nn.Embedding(num_embeddings=num_bs, embedding_dim=embedding_dim)
        
        self.lstm = nn.LSTM(input_size=dynamic_input_size, hidden_size=hidden_size, batch_first=True)
        self.attention = TemporalAttention(hidden_size)
        
        self.fc1 = nn.Linear(hidden_size + static_input_size + embedding_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x_dynamic, x_static, bs_id):
        embeds = self.bs_embedding(bs_id)
        
        lstm_out, _ = self.lstm(x_dynamic)
        context_vector = self.attention(lstm_out)
        
        combined = torch.cat((context_vector, x_static, embeds), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        return self.output(x).squeeze()


def train_model(model, train_loader, epochs=15):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_dyn, x_stat, bs, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            
            if torch.rand(1).item() < 0.15:
                x_dyn_masked = x_dyn.clone()
                x_dyn_masked[:, :, 0] = 0.0
                predictions = model(x_dyn_masked, x_stat, bs)
            else:
                predictions = model(x_dyn, x_stat, bs)
                
            loss = criterion(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
    return model

def evaluate_model(model, test_loader, target_scaler):
    model.eval()
    all_preds, all_actuals = [], []
    
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for x_dyn, x_stat, bs, y in test_loader:
            preds = model(x_dyn, x_stat, bs)
            all_preds.extend(preds.numpy())
            all_actuals.extend(y.numpy())
            
    preds_kwh = target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    actuals_kwh = target_scaler.inverse_transform(np.array(all_actuals).reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(actuals_kwh, preds_kwh)
    rmse = np.sqrt(mean_squared_error(actuals_kwh, preds_kwh))
    mape = mean_absolute_percentage_error(actuals_kwh, preds_kwh) * 100
    
    print(f"--- FINAL MODEL ACCURACY ---")
    print(f"MAE:  {mae:.2f} W")
    print(f"RMSE: {rmse:.2f} W")
    print(f"MAPE: {mape:.2f}%")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sample_size = min(len(actuals_kwh), 5000)
    plt.scatter(actuals_kwh[:sample_size], preds_kwh[:sample_size], alpha=0.5, color='blue', s=10)
    min_val = min(min(actuals_kwh[:sample_size]), min(preds_kwh[:sample_size]))
    max_val = max(max(actuals_kwh[:sample_size]), max(preds_kwh[:sample_size]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.title('Overall: Predicted vs Actual Energy (W)')
    plt.xlabel('Actual Energy (W)')
    plt.ylabel('Predicted Energy (W)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    errors = actuals_kwh - preds_kwh
    plt.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.title('Prediction Error Distribution (Residuals)')
    plt.xlabel('Error (Actual - Predicted) (W)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(actuals_kwh[:100], label='Actual', color='green', marker='o', markersize=4)
    plt.plot(preds_kwh[:100], label='Predicted', color='orange', linestyle='--', marker='x', markersize=4)
    plt.title('Time-Series Tracking (Slice 1)')
    plt.xlabel('Time Step (Hour)')
    plt.ylabel('Energy (W)')
    plt.legend()
    plt.grid(True)
    
    
    plt.tight_layout()
    plt.savefig('model_evaluation_extended.png')
    print("✅ Extended model evaluation graphs saved to 'model_evaluation_extended.png'")
    
    return preds_kwh, actuals_kwh


def is_valid_combo(combo, raw_load_forecast):
    m1, m2, m3, m4, m5, m6 = combo

    if (m5 == 1 or m6 == 1) and (m1 == 1 or m2 == 1): return False
    
    if raw_load_forecast > 0.3 and (m4 == 1 or m5 == 1 or m6 == 1): return False
    
    if raw_load_forecast > 0.6 and (m1 == 1 or m2 == 1 or m3 == 1): return False
    
    if raw_load_forecast > 0.7 and sum(combo) > 0: return False
    
    return True

def recommend_es_mode(model, bs_encoded_id, historical_dynamic_data, static_data, 
                      forecasted_load_scaled, raw_load_forecast, target_scaler, next_hour_time_features):
    model.eval()
    es_combinations = list(itertools.product([0.0, 1.0], repeat=6))
    
    best_energy = float('inf')
    best_mode = None
    
    with torch.no_grad():
        for combo in es_combinations:
            if not is_valid_combo(combo, raw_load_forecast):
                continue
                
            next_step_features = np.concatenate(([forecasted_load_scaled], list(combo), next_hour_time_features))
            hypothetical_window = np.vstack([historical_dynamic_data[1:], next_step_features])
            
            dyn_tensor = torch.tensor(hypothetical_window, dtype=torch.float32).unsqueeze(0)
            stat_tensor = torch.tensor(static_data, dtype=torch.float32).unsqueeze(0)
            bs_tensor = torch.tensor([bs_encoded_id], dtype=torch.long)
            
            pred_scaled = model(dyn_tensor, stat_tensor, bs_tensor).item()
            pred_energy = target_scaler.inverse_transform([[pred_scaled]])[0][0]
            
            if pred_energy < best_energy:
                best_energy = pred_energy
                best_mode = combo
                
    return best_mode, best_energy

if __name__ == "__main__":
    df_train, df_test, le_bs, scaler, target_scaler = load_and_preprocess_data()
    
    lookback = 24
    train_dataset = TelecomDataset(df_train, lookback=lookback)
    test_dataset = TelecomDataset(df_test, lookback=lookback)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    num_unique_bs = len(le_bs.classes_)
    model = EnergyPredictorAttention(num_bs=num_unique_bs)
    
    model = train_model(model, train_loader, epochs=10) 
    
    evaluate_model(model, test_loader, target_scaler)
    
    target_bs = 'B_0'
    bs_encoded_id = le_bs.transform([target_bs])[0]
    
    b0_data = df_test[df_test['BS'] == target_bs].iloc[-lookback:].copy()
    
    feature_cols = [
        'load', 'ESMode1', 'ESMode2', 'ESMode3', 'ESMode4', 'ESMode5', 'ESMode6',
        'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos'
    ]
    historical_dynamic = b0_data[feature_cols].values
    static_data = b0_data[['Antennas', 'TXpower', 'Bandwidth']].values[-1] 
    
    raw_load_forecast = 0.25
    dummy_input = np.zeros((1, 4))
    dummy_input[0, 0] = raw_load_forecast 
    scaled_forecast = scaler.transform(dummy_input)[0, 0] 
    
    next_hour_time_features = np.array([
        np.sin(2 * np.pi * 3 / 24.0), np.cos(2 * np.pi * 3 / 24.0),
        np.sin(2 * np.pi * 1 / 7.0), np.cos(2 * np.pi * 1 / 7.0)
    ])
    
    print(f"\n--- OPTIMIZATION RESULTS FOR {target_bs} ---")
    optimal_modes, predicted_energy = recommend_es_mode(
        model, bs_encoded_id, historical_dynamic, static_data, 
        scaled_forecast, raw_load_forecast, target_scaler, next_hour_time_features
    )
    
    print(f"Raw Load Forecast: {raw_load_forecast*100}% capacity")
    print(f"Recommended ES Modes (1 to 6): {optimal_modes}")
    print(f"Expected Energy Consumption: {predicted_energy:.2f} kWh")