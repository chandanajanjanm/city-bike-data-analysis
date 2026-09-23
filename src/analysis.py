from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# PROJECT PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parents[1]

TRIPS_PATH = BASE_DIR / "data" / "processed" / "citibike_cleaned_features.csv"
WEATHER_PATH = BASE_DIR / "data" / "processed" / "citibike_daily_weather.csv"

FIGURES_DIR = BASE_DIR / "reports" / "figures"
TABLES_DIR = BASE_DIR / "reports" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# LOAD TRIP DATA
# ==============================

df = pd.read_csv(TRIPS_PATH, low_memory=False)

print("Trip dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# ==============================
# 1. BASIC INFORMATION
# ==============================

df.describe().to_csv(TABLES_DIR / "dataset_summary.csv")


# ==============================
# 2. TRIP DURATION DISTRIBUTION
# ==============================

plt.figure(figsize=(10, 6))

sns.histplot(df["trip_duration"], bins=50, kde=True)

plt.title("Trip Duration Distribution")
plt.xlabel("Duration (minutes)")
plt.ylabel("Number of Trips")
plt.xlim(0, 60)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "trip_duration_distribution.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 6))

sns.histplot(np.log1p(df["trip_duration"]), bins=50, kde=True)

plt.title("Log Trip Duration Distribution")
plt.xlabel("Log(Duration)")
plt.ylabel("Number of Trips")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "log_trip_duration_distribution.png", dpi=300)
plt.close()


print("\nAverage trip duration:",
      round(df["trip_duration"].mean(), 2), "minutes")

print("Median trip duration:",
      round(df["trip_duration"].median(), 2), "minutes")


# ==============================
# 3. TRIPS BY HOUR
# ==============================

hourly_trips = df.groupby("hour").size()

hourly_trips.to_csv(TABLES_DIR / "trips_by_hour.csv")

plt.figure(figsize=(10, 6))

sns.barplot(x=hourly_trips.index, y=hourly_trips.values)

plt.title("Number of Trips by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.xticks(range(24))
plt.tight_layout()

plt.savefig(FIGURES_DIR / "trips_by_hour.png", dpi=300)
plt.close()


# ==============================
# 4. TRIPS BY DAY OF WEEK
# ==============================

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

daily_trips = (
    df["day_of_week"]
    .value_counts()
    .reindex(day_order)
)

daily_trips.to_csv(TABLES_DIR / "trips_by_day_of_week.csv")

plt.figure(figsize=(10, 6))

sns.barplot(x=daily_trips.index, y=daily_trips.values)

plt.title("Number of Trips by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Trips")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "trips_by_day_of_week.png", dpi=300)
plt.close()


# ==============================
# 5. MEMBER VS CASUAL
# ==============================

member_summary = (
    df.groupby("member_casual")["trip_duration"]
    .agg(["mean", "median", "count"])
    .reset_index()
)

member_summary.to_csv(
    TABLES_DIR / "member_casual_summary.csv",
    index=False
)

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="member_casual",
    y="trip_duration"
)

plt.title("Trip Duration: Member vs Casual")
plt.xlabel("User Type")
plt.ylabel("Trip Duration (minutes)")
plt.ylim(0, 60)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "member_casual_duration.png", dpi=300)
plt.close()


# ==============================
# 6. TOP 10 START STATIONS
# ==============================

top_stations = (
    df["start_station_name"]
    .value_counts()
    .head(10)
)

top_stations.to_csv(
    TABLES_DIR / "top_10_start_stations.csv"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_stations.values,
    y=top_stations.index
)

plt.title("Top 10 Start Stations")
plt.xlabel("Number of Trips")
plt.ylabel("Station Name")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "top_10_start_stations.png", dpi=300)
plt.close()


# ==============================
# 7. TRIP DISTANCE DISTRIBUTION
# ==============================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["distance_km"].dropna(),
    bins=50,
    kde=True
)

plt.title("Trip Distance Distribution")
plt.xlabel("Distance (km)")
plt.ylabel("Number of Trips")
plt.xlim(0, 15)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "distance_distribution.png", dpi=300)
plt.close()


print("\nAverage distance:",
      round(df["distance_km"].mean(), 2), "km")


# ==============================
# 8. WEEKEND VS WEEKDAY
# ==============================

weekend_summary = (
    df.groupby("is_weekend")["trip_duration"]
    .agg(["mean", "median", "count"])
    .reset_index()
)

weekend_summary.to_csv(
    TABLES_DIR / "weekend_weekday_summary.csv",
    index=False
)

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="is_weekend",
    y="trip_duration"
)

plt.title("Trip Duration: Weekday vs Weekend")
plt.xlabel("Weekend (False = Weekday)")
plt.ylabel("Trip Duration (minutes)")
plt.ylim(0, 60)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "weekend_weekday_duration.png", dpi=300)
plt.close()


# ==============================
# LOAD WEATHER DATA
# ==============================

weather_df = pd.read_csv(WEATHER_PATH)

print("\nWeather dataset shape:", weather_df.shape)

weather_df["trip_date"] = pd.to_datetime(weather_df["trip_date"])


# ==============================
# 9. TEMPERATURE VS RIDES
# ==============================

plt.figure(figsize=(8, 6))

sns.regplot(
    data=weather_df,
    x="temperature_mean",
    y="ride_count",
    scatter_kws={"alpha": 0.6}
)

plt.title("Temperature vs Daily Ride Count")
plt.xlabel("Average Temperature")
plt.ylabel("Daily Ride Count")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "temperature_vs_rides.png", dpi=300)
plt.close()


# ==============================
# 10. PRECIPITATION VS RIDES
# ==============================

plt.figure(figsize=(8, 6))

sns.regplot(
    data=weather_df,
    x="rain_sum",
    y="ride_count",
    scatter_kws={"alpha": 0.6}
)

plt.title("Rainfall vs Daily Ride Count")
plt.xlabel("Rainfall")
plt.ylabel("Daily Ride Count")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "rainfall_vs_rides.png", dpi=300)
plt.close()


# ==============================
# 11. WEATHER CORRELATION
# ==============================

numeric_columns = [
    "ride_count",
    "temperature_mean",
    "rain_sum",
    "wind_speed_max"
]

correlation = weather_df[numeric_columns].corr()

correlation.to_csv(
    TABLES_DIR / "weather_correlation.csv"
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Weather and Ride Count Correlation")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "weather_correlation_heatmap.png", dpi=300)
plt.close()


# ==============================
# 12. WEATHER SUMMARY
# ==============================

weather_summary = weather_df[
    [
        "ride_count",
        "temperature_mean",
        "rain_sum",
        "wind_speed_max"
    ]
].describe()

weather_summary.to_csv(
    TABLES_DIR / "weather_summary.csv"
)


# ==============================
# COMPLETION MESSAGE
# ==============================

print("\nComplete EDA analysis finished successfully!")
print("Charts saved in:", FIGURES_DIR)
print("Tables saved in:", TABLES_DIR)