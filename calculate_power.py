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

    print(f"Total energy consumption: {total_energy_kWh} kWh")
except FileNotFoundError:
    print("Log file not found. Please ensure that the logging script is running.")
except pd.errors.EmptyDataError:
    print("Log file is empty. No data to process.")
