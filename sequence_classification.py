import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from vllm import LLM, SamplingParams


prediction_col = "response"


def llama3_skynet_api(snippet, prompt):
    try:
        url = 'https://turbo.skynet.coypu.org/'
        request = requests.post(url, json={"messages": [{"role": "user",
                                           "content": f"{prompt}\n{snippet}\nanswer:\n"}],
                                "temperature": 0.1,
                                "max_new_tokens": 10}).json()
        return request.get("generated_text")
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

def run_llama3_vllm():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    file = open(f"./prompts/has_cause.txt", "r")
    prompt = file.read()
    for year in tqdm([2023, 2018, 2019, 2020, 2021, 2022]):
        df_path = Path(f"./data/DJN/inflation_mentioned_news_{year}.csv")
        df = pd.read_csv(df_path)
        print(f"Year: {year}")
        print(f"Number of articles: {len(df)}")
        df = df.drop_duplicates(subset=['body'])
        print(f"Number of articles (deduplicated): {len(df)}")
        prompts = [f"{prompt}\n{snippet}\nanswer:\n" for snippet in df["body"].values]
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        df[prediction_col] = generated_texts
        df.to_csv(Path(f"./outputs/llama3/has_cause_{year}.csv"), index=False)
        return df

if __name__ == "__main__":
    run_llama3_vllm()
