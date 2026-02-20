import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator as KneedleLocator

# Preparation: creating a designated folder to save the graphics
output_dir = "analysis_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"directory: {output_dir} was created successfully")


def load_and_prep_data(file_path):
    """
    Loads the dataset, merges date and time columns, and cleans the data.

    Args:
        file_path (str): Path to the source .txt or .csv file.
    Returns:
        pd.DataFrame: Cleaned time-series DataFrame with a Datetime index.
    """
    print("Loading Data...")

    # Read dataset with attention to inconsistent data types and nulls
    df = pd.read_csv(file_path, sep = ";", low_memory = False)

    # Create a unified timestamp and set as index for time-series operations
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst = True)
    df = df.set_index("Datetime")
    df = df.drop(["Date", "Time"], axis = 1)

    # Force conversion of all columns to numeric; non-numeric values (like '?')
    # are coerced to NaN to ensure data consistency for subsequent filling and analysis.
    df = df.apply(pd.to_numeric, errors = "coerce")
    for col in df.columns:
        df[col] = df[col].astype(float)

    # Forward fill (ffill) to handle remaining missing values by propagating last valid observation
    df = df.ffill()

    print("Data loaded and cleaned!")
    print("Shape of the final dataframe:", df.shape)
    return df


def detect_outliers(df):
    """
    Identifies anomalies in daily power consumption using the
    3-Sigma rule (standard deviation).
    """
    print("Detecting outliers...")

    # Resample to daily frequency for high-level trend analysis
    daily = df["Global_active_power"].resample("d").sum().to_frame()

    mean_val = daily["Global_active_power"].mean()
    std_val = daily["Global_active_power"].std()

    # Flag values exceeding 3 standard deviations from the mean
    daily["is_outlier"] = (np.abs(daily["Global_active_power"] - mean_val) > (3 * std_val))

    # Visualization of the anomalies
    plt.figure(figsize = (12, 5))
    plt.plot(daily.index, daily["Global_active_power"])
    outliers = daily[daily["is_outlier"]]
    plt.scatter(outliers.index, outliers["Global_active_power"], color = "red", label = "Outliers")
    plt.title("Identified Outliers")
    plt.savefig(f"{output_dir}/01_outliers.png", dpi=300)
    plt.close()

    print("Graphic 01 (outliers) saved")
    return daily


def run_elbow_method(df, max_k = 6):
    """
    Determines the optimal number of clusters for K-Means using the Elbow method.
    """
    print('Running Elbow Method...')
    hourly = df["Global_active_power"].resample("h").sum().to_frame()
    hourly["hour"] = hourly.index.hour
    hourly["date"] = hourly.index.date
    cluster_input = hourly.pivot(index = "date", columns = "hour", values = "Global_active_power" ).dropna()

    scaler = StandardScaler()
    scaled_feature = scaler.fit_transform(cluster_input)

    inertia_values = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 5)
        kmeans.fit(scaled_feature)
        inertia_values.append(kmeans.inertia_)

    # Visualization of the Elbow curve
    plt.figure(figsize = (8, 4))
    plt.plot(range(1, max_k + 1), inertia_values, marker = "o")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(range(1, max_k + 1))
    plt.grid(True, alpha = 0.3)
    plt.savefig(f"{output_dir}/02_elbow_method.png", dpi=300)
    plt.close()
    print("Graphic 02 (elbow method) saved")

    # Kneedle
    kl = KneedleLocator(range(1, max_k + 1), inertia_values, curve = 'convex', direction = "decreasing", S=1.0)
    optimal_k = kl.knee
    print(f"Optimal number of clusters determined by Elbow method: {optimal_k}")
    return optimal_k


def perform_clustering(df, n_clusters):
    """
    Groups days with similar consumption profiles using K-Means clustering.
    Transform raw time-series into a 24-hour feature matrix.
    """
    print("Performing clustering...")

    # Preparation: Aggregation by hour
    hourly = df["Global_active_power"].resample("h").sum().to_frame()
    hourly["hour"] = hourly.index.hour
    hourly["date"] = hourly.index.date

    # Pivot-table: rows = specific dates, columns = hours (0-23)
    cluster_input = hourly.pivot(index = "date", columns = "hour", values = "Global_active_power" ).dropna()

    # Scaling is crucial for distance-based algorithms like K-Means
    scaler = StandardScaler()
    scaled_feature = scaler.fit_transform(cluster_input)

    # Initialize and fit K-Means
    kmeans= KMeans(n_clusters = n_clusters, random_state = 42, n_init = 5)
    cluster_input["cluster"] = kmeans.fit_predict(scaled_feature)

    # Visualization of average cluster profiles
    plt.figure(figsize = (10, 6))
    for i in range(n_clusters):
        # Calculation of the average profile for each cluster
        profile =  cluster_input[cluster_input["cluster"] == i].drop("cluster", axis = 1).mean()
        plt.plot(profile, label = f"Cluster {i}", linewidth = 2)

    plt.title("identified clusters")
    plt.xlabel("hour")
    plt.ylabel("usage in kW")
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.savefig(f"{output_dir}/03_clusters.png", dpi=300)
    plt.close()
    print("Clustering done!")
    return cluster_input

def analyze_routines(df):
    """
    Visualizes average energy consumption by household sub-zones across the day.
    """
    print("Analyzing routines...")
    df["hour"] = df.index.hour

    # Group by hour to identify time-based usage patterns
    kitchen = df.groupby("hour")["Sub_metering_1"].mean()
    laundry_room = df.groupby("hour")["Sub_metering_2"].mean()
    heater_conditioner = df.groupby("hour")["Sub_metering_3"].mean()

    # Visualization of the usage in the different household-spaces
    plt.figure(figsize = (10, 6))
    plt.fill_between(kitchen.index, kitchen, color = "blue", alpha = 0.4, label = "kitchen")
    plt.fill_between(laundry_room.index, laundry_room, color = "green", alpha = 0.4, label = "laundry room")
    plt.fill_between(heater_conditioner.index, heater_conditioner, color = "orange", alpha = 0.4, label = "water-heater & air-conditioner")
    plt.title("Routines per time of day")
    plt.legend()
    plt.savefig(f"{output_dir}/04_routines.png", dpi=300)
    plt.close()
    print("Routines analyzed!")
    return kitchen, laundry_room, heater_conditioner


# Main function
if __name__ == "__main__":
    # Ensure dataset follows the source format from UCI Machine Learning Repository
    path = ".idea/household_power_consumption.txt"

    if os.path.exists(path):
        # Execution of the pipeline
        data = load_and_prep_data(path)
        detect_outliers(data)
        optimal_k = run_elbow_method(data)

        perform_clustering(data, n_clusters = optimal_k)
        analyze_routines(data)
        print("Analysis done!")



