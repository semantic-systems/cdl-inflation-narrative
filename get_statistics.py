import pandas as pd
import plotly.express as px

inflation_has_cause_file_names = ["./outputs/llama3/inflation/1.1.inflation_has_cause_2018.csv",
                                  "./outputs/llama3/inflation/1.1.inflation_has_cause_2019.csv",
                                  "./outputs/llama3/inflation/1.1.inflation_has_cause_2020.csv",
                                  "./outputs/llama3/inflation/1.1.inflation_has_cause_2021.csv",
                                  "./outputs/llama3/inflation/1.1.inflation_has_cause_2022.csv",
                                  "./outputs/llama3/inflation/1.1.inflation_has_cause_2023.csv"]

deflation_has_cause_file_names = ["./outputs/llama3/deflation/1.2.deflation_has_cause_2018.csv",
                                  "./outputs/llama3/deflation/1.2.deflation_has_cause_2019.csv",
                                  "./outputs/llama3/deflation/1.2.deflation_has_cause_2020.csv",
                                  "./outputs/llama3/deflation/1.2.deflation_has_cause_2021.csv",
                                  "./outputs/llama3/deflation/1.2.deflation_has_cause_2022.csv",
                                  "./outputs/llama3/deflation/1.2.deflation_has_cause_2023.csv"]

change_of_prices_file_names = ["./outputs/llama3/change_in_prices/2.1.change_in_prices_2018.csv",
                               "./outputs/llama3/change_in_prices/2.1.change_in_prices_2020.csv",
                               "./outputs/llama3/change_in_prices/2.1.change_in_prices_2022.csv",
                               "./outputs/llama3/change_in_prices/2.1.change_in_prices_2019.csv",
                               "./outputs/llama3/change_in_prices/2.1.change_in_prices_2021.csv",
                               "./outputs/llama3/change_in_prices/2.1.change_in_prices_2023.csv"]

direction_of_change_file_names = ["./outputs/llama3/change_in_prices/2.2.change_direction_2018.csv",
                                  "./outputs/llama3/change_in_prices/2.2.change_direction_2020.csv",
                                  "./outputs/llama3/change_in_prices/2.2.change_direction_2022.csv",
                                  "./outputs/llama3/change_in_prices/2.2.change_direction_2019.csv",
                                  "./outputs/llama3/change_in_prices/2.2.change_direction_2021.csv",
                                  "./outputs/llama3/change_in_prices/2.2.change_direction_2023.csv"]

file_names = change_of_prices_file_names

stats = []
df_has_cause = []

for i, name in enumerate(file_names):
    df = pd.read_csv(name)
    yes_df = df.loc[df['response'].str.startswith("Yes")]
    no_df = df.loc[df['response'].str.startswith("No")]
    na_df = df.loc[df['response'].str.startswith("0")]
    year = name.split(".csv")[0].split("_")[-1]
    stats.append({"Year": year, "Type": "#has_cause", "Count": len(yes_df)})
    stats.append({"Year": year, "Type": "#no_cause", "Count": len(no_df)})
    stats.append({"Year": year, "Type": "#na", "Count": len(na_df)})
    print(f"Year: {year} - #articles: {len(df)} - #has_cause: {len(yes_df)} - #no_cause: {len(no_df)} - #na: {len(na_df)}")
    df_has_cause.append(yes_df)

stats_df = pd.DataFrame(stats)
df_has_cause = pd.concat(df_has_cause)
df_has_cause.to_csv("./outputs/llama3/change_in_prices/has_cause_df.csv")

fig = px.bar(stats_df, x="Year", y="Count", color="Type",
             title="Zero-shot classification with Llama 3 \n (whether or not the cause of change in prices is mentioned)")
fig.write_image("./outputs/llama3/change_in_prices/stats_classification_has_cause.png")

stats_has_cause_df = stats_df.loc[stats_df["Type"] == "#has_cause"]
fig = px.bar(stats_has_cause_df, x="Year", y="Count",
             title="Distribution of articles mentioning causes for change in prices")
fig.write_image("./outputs/llama3/change_in_prices/has_cause_distribution.png")



