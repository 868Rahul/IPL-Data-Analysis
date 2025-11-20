import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================
# CREATE OUTPUT FOLDERS
# =============================
os.makedirs("../output/plots", exist_ok=True)
os.makedirs("../output/tables", exist_ok=True)

# =============================
# LOAD CLEANED DATA
# =============================
df = pd.read_csv("../data/cleaned_batting_card.csv")

print("Dataset Loaded:", df.shape)
print(df.columns)

# =====================================================
# 1️⃣ TOP 10 BATSMEN (RUNS)
# =====================================================
top_batsmen = (
    df.groupby("fullname")["runs"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Batsmen (Runs):")
print(top_batsmen)

# Save table
top_batsmen.to_csv("../output/tables/top10_batsmen_runs.csv")

plt.figure(figsize=(10,5))
sns.barplot(x=top_batsmen.values, y=top_batsmen.index)
plt.title("Top 10 Batsmen by Total Runs")
plt.xlabel("Runs")
plt.ylabel("Batsman")
plt.tight_layout()
plt.savefig("../output/plots/top10_batsmen_runs.png")
plt.show()

# ==== PIE CHART ====
plt.figure(figsize=(8,8))
plt.pie(top_batsmen.values, labels=top_batsmen.index, autopct="%1.1f%%")
plt.title("Runs Contribution (Top 10 Batsmen)")
plt.savefig("../output/plots/top10_batsmen_runs_pie.png")
plt.show()


# =====================================================
# 2️⃣ STRIKE RATE LEADERS (MIN 100 BALLS) - FINAL FIX
# =====================================================

# Convert columns to numeric
df["ballsfaced"] = pd.to_numeric(df["ballsfaced"], errors="coerce")
df["strikerate"] = pd.to_numeric(df["strikerate"], errors="coerce")

# Drop rows where these are missing
df_clean = df.dropna(subset=["ballsfaced", "strikerate"])

# Group data by batsman
grouped = df_clean.groupby("fullname").agg({
    "ballsfaced": "sum",
    "strikerate": "mean"
})

# Filter only players with >= 100 total balls faced
qualified = grouped[grouped["ballsfaced"] >= 100]

if qualified.empty:
    print("\n⚠ No player qualifies even after grouping!")
else:
    # Top 10 strike rate
    top_strike_rate = qualified["strikerate"].sort_values(ascending=False).head(10)

    print("\nTop Strike Rate (Min 100 Balls Total):")
    print(top_strike_rate)

    # Save table
    top_strike_rate.to_csv("../output/tables/top_strike_rate.csv")

    # Bar Plot
    plt.figure(figsize=(10,5))
    sns.barplot(x=top_strike_rate.values, y=top_strike_rate.index)
    plt.title("Top Strike Rate (Min 100 Balls Total)")
    plt.xlabel("Strike Rate")
    plt.ylabel("Batsman")
    plt.tight_layout()
    plt.savefig("../output/plots/top_strike_rate.png")
    plt.show()

    # Pie Chart
    plt.figure(figsize=(8,8))
    plt.pie(top_strike_rate.values, labels=top_strike_rate.index, autopct="%1.1f%%")
    plt.title("Strike Rate Share (Top 10)")
    plt.savefig("../output/plots/top_strike_rate_pie.png")
    plt.show()


# =====================================================
# 3️⃣ MOST FOURS & SIXES
# =====================================================
top_fours = df.groupby("fullname")["fours"].sum().sort_values(ascending=False).head(10)
top_sixes = df.groupby("fullname")["sixes"].sum().sort_values(ascending=False).head(10)

print("\nTop Fours:")
print(top_fours)
print("\nTop Sixes:")
print(top_sixes)

top_fours.to_csv("../output/tables/top_fours.csv")
top_sixes.to_csv("../output/tables/top_sixes.csv")

plt.figure(figsize=(10,5))
sns.barplot(x=top_sixes.values, y=top_sixes.index)
plt.title("Top 10 Six Hitters")
plt.xlabel("Sixes")
plt.ylabel("Batsman")
plt.tight_layout()
plt.savefig("../output/plots/top10_sixes.png")
plt.show()

# ==== PIE CHART ====
plt.figure(figsize=(8,8))
plt.pie(top_sixes.values, labels=top_sixes.index, autopct="%1.1f%%")
plt.title("Sixes Share (Top 10)")
plt.savefig("../output/plots/top10_sixes_pie.png")
plt.show()


# =====================================================
# 4️⃣ TEAM-WISE RUN CONTRIBUTION
# =====================================================
team_runs = (
    df.groupby("home_team")["runs"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTeam-wise Runs:")
print(team_runs)

team_runs.to_csv("../output/tables/team_runs.csv")

plt.figure(figsize=(12,6))
sns.barplot(x=team_runs.index, y=team_runs.values)
plt.xticks(rotation=45)
plt.title("Total Runs by Team")
plt.ylabel("Runs")
plt.tight_layout()
plt.savefig("../output/plots/team_runs.png")
plt.show()

# ==== PIE CHART ====
plt.figure(figsize=(9,9))
plt.pie(team_runs.values, labels=team_runs.index, autopct="%1.1f%%")
plt.title("Runs Contribution by Teams")
plt.savefig("../output/plots/team_runs_pie.png")
plt.show()


# =====================================================
# 5️⃣ SEASON-WISE RUN TREND
# =====================================================
season_trend = (
    df.groupby("season")["runs"]
    .sum()
)

print("\nSeason-wise Run Trend:")
print(season_trend)

season_trend.to_csv("../output/tables/season_trend.csv")

plt.figure(figsize=(10,5))
plt.plot(season_trend.index, season_trend.values, marker="o")
plt.title("IPL Season-wise Total Runs")
plt.xlabel("Season")
plt.ylabel("Total Runs")
plt.grid(True)
plt.tight_layout()
plt.savefig("../output/plots/season_trend.png")
plt.show()

# ==== PIE CHART ====
plt.figure(figsize=(9,9))
plt.pie(season_trend.values, labels=season_trend.index, autopct="%1.1f%%")
plt.title("Season Contribution to Total Runs")
plt.savefig("../output/plots/season_trend_pie.png")
plt.show()

print("\n✔ ALL GRAPHS SAVED IN:  output/plots")
print("✔ ALL TABLES SAVED IN: output/tables")
