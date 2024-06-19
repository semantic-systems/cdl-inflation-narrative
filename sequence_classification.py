import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path


prediction_col = "response"


def binary_classification_inflation(snippet, prompt):
    try:
        url = 'https://turbo.skynet.coypu.org/'
        request = requests.post(url, json={"messages": [{"role": "user",
                                           "content": f"{prompt}\n{snippet}\nanswer:\n"}],
                                "temperature": 0.1,
                                "max_new_tokens": 10,
                                 "key": "M7ZQL9ELMSDXXE86"}).json()
        return request[0].get('choices')[0].get("message").get("content")
    except Exception as e:
        return e


def annotate_event_type(df, year, prompt, forced=False):
    if not forced and prediction_col in df.columns:
        return df
    else:
        predictions = []
        for i, snippet in tqdm(enumerate(df["body"].values)):
            print(i)
            print(snippet)
            result = binary_classification_inflation(snippet, prompt)
            print(result)
            print(" ")
            predictions.append(result)
            if i % 20 == 0:
                with open("has_cause.txt", "w") as f:
                    for s in predictions:
                        f.write(str(s)+"\n")
        df[prediction_col] = predictions
        df.to_csv(Path(f"./has_cause_{year}.csv"), index=False)
        return df


if __name__ == "__main__":
    for year in [1994, 2023, 2018, 2019, 2020, 2021, 2022]:
        df_path = Path(f"./data/inflation_mentioned_news_{year}.csv")
        df = pd.read_csv(df_path)
        file = open(f"./prompts/has_cause.txt", "r")
        prompt = file.read()
        annotate_event_type(df, year=year, prompt=prompt, forced=True)
