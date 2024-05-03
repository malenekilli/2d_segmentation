import pandas as pd

try:
    # Load the log file
    log_df = pd.read_csv('gpu_power_usage.log', names=['timestamp', 'power_usage_watts'])

    # Extract wattage and convert to float
    log_df['power_usage_watts'] = log_df['power_usage_watts'].str.extract('(\d+\.\d+)').astype(float)

    # Convert power usage to kilowatts
    log_df['power_usage_kW'] = log_df['power_usage_watts'] / 1000

    # Calculate the total energy consumption in kWh
    # Assuming that the log is in seconds intervals
    total_energy_kWh = log_df['power_usage_kW'].sum() / 3600  # Divide by 3600 to convert seconds to hours

    # Assuming the Tesla Model 3 efficiency is 0.15 kWh per kilometer
    model_3_efficiency = 0.15

    # Calculate the possible distance driven
    distance_km = total_energy_kWh / model_3_efficiency

    print(f"Total energy consumption: {total_energy_kWh} kWh")
    print(f"Estimated distance that could be driven by a Tesla Model 3: {distance_km:.2f} km")
except FileNotFoundError:
    print("Log file not found. Please ensure that the logging script is running.")
except pd.errors.EmptyDataError:
    print("Log file is empty. No data to process.")
except Exception as e:
    print(f"An error occurred: {e}")
