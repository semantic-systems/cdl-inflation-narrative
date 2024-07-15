import pandas as pd
import requests
from pathlib import Path
from vllm import LLM, SamplingParams
import argparse


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
                with open("1.1.inflation_has_cause.txt", "w") as f:
                    for s in predictions:
                        f.write(str(s)+"\n")
        df[prediction_col] = predictions
        df.to_csv(Path(f"./has_cause_{year}.csv"), index=False)
        return df

def run_llama3_vllm_inflation_1_1():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)
    file = open(f"prompts/1.1.inflation_has_cause.txt", "r")
    prompt = file.read()
    for year in [2023, 2018, 2019, 2020, 2021, 2022]:
        df_path = Path(f"./data/DJN/inflation_mentioned_news_{year}.csv")
        df = pd.read_csv(df_path)
        print(f"Year: {year}")
        print(f"Number of articles: {len(df)}")
        df = df.drop_duplicates(subset=['body'])
        print(f"Number of articles (deduplicated): {len(df)}")
        prompts = [f"{prompt}\n{snippet}\nanswer:\n" for snippet in df["body"].values]
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        df["prompts"] = prompts
        df[prediction_col] = generated_texts
        df.to_csv(Path(f"./outputs/llama3/inflation/1.1.inflation_has_cause_{year}.csv"), index=False)

def run_llama3_vllm_deflation_1_2():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)
    file = open(f"prompts/1.2.deflation_has_cause.txt", "r")
    prompt = file.read()
    for year in [2023, 2018, 2019, 2020, 2021, 2022]:
        df_path = Path(f"./data/DJN/deflation_prices_{year}.csv")
        df = pd.read_csv(df_path)
        print(f"Year: {year}")
        print(f"Number of articles: {len(df)}")
        df = df.drop_duplicates(subset=['body'])
        print(f"Number of articles (deduplicated): {len(df)}")
        prompts = [f"{prompt}\n{snippet}\nanswer:\n" for snippet in df["body"].values]
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        df["prompts"] = prompts
        df[prediction_col] = generated_texts
        df.to_csv(Path(f"./outputs/llama3/deflation/1.2.deflation_has_cause_{year}.csv"), index=False)

def run_llama3_vllm_change_in_prices_2_1():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)
    file = open(f"prompts/2.1.change_in_prices.txt", "r")
    prompt = file.read()
    for year in [2023, 2018, 2019, 2020, 2021, 2022]:
        df_path = Path(f"./data/DJN/inflation_deflation_prices_{year}.csv")
        df = pd.read_csv(df_path)
        print(f"Year: {year}")
        print(f"Number of articles: {len(df)}")
        df = df.drop_duplicates(subset=['body'])
        print(f"Number of articles (deduplicated): {len(df)}")
        prompts = [f"{prompt}\n{snippet}\nanswer:\n" for snippet in df["body"].values]
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        df["prompts"] = prompts
        df["answer_change_in_prices"] = generated_texts
        df.to_csv(Path(f"./outputs/llama3/change_in_prices/2.1.change_in_prices_{year}.csv"), index=False)

def run_llama3_vllm_change_in_prices_2_2():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)
    file = open(f"prompts/2.2.change_direction.txt", "r")
    prompt = file.read()
    for year in [2023, 2018, 2019, 2020, 2021, 2022]:
        df_path = Path(f"./outputs/llama3/change_in_prices/2.1.change_in_prices_{year}.csv")
        df = pd.read_csv(df_path)
        print(f"Year: {year}")
        print(f"Number of articles: {len(df)}")
        df = df.loc[df["answer_change_in_prices"].str.startswith("Yes")]
        df["prompts"] = 'User: ' + df['prompts'] + '\nSystem: ' + df['answer_change_in_prices']
        prompts = [f"{prompt_history}\nUser: {prompt}\n\nSystem: " for prompt_history in df["prompts"].values]
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        df["prompts"] = prompts
        df["answer_change_direction"] = generated_texts
        df.to_csv(Path(f"./outputs/llama3/change_in_prices/2.2.change_direction_{year}.csv"), index=False)

def run_llama3_vllm_one_hop_dag_3_1():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)
    file = open(f"prompts/3.1.one_hop_dag.txt", "r", encoding="utf-8")
    prompt = file.read()
    prev_file = open(f"prompts/1.1.inflation_has_cause.txt", "r")
    prev_prompt = prev_file.read()
    for year in [2023, 2018, 2019, 2020, 2021, 2022]:
        df_path = Path(f"./outputs/llama3/inflation/1.1.inflation_has_cause_{year}.csv")
        df = pd.read_csv(df_path)
        prev_prompts = [f"{prev_prompt}\n{snippet}\nanswer:\n" for snippet in df["body"].values]
        df["prompts"] = prev_prompts
        print(f"Year: {year}")
        print(f"Number of articles: {len(df)}")
        df = df.loc[df[prediction_col].str.startswith("Yes")]
        df["prompts"] = 'User: ' + df['prompts'] + '\nSystem: ' + df['response']
        prompts = [f"{prompt_history}\nUser: {prompt}\n\nSystem: " for prompt_history in df["prompts"].values]
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        df["prompts"] = prompts
        df["answer_one_hop_dag"] = generated_texts
        df.to_csv(Path(f"./outputs/llama3/one_hop_dag/3.1.one_hop_dag_{year}.csv"), index=False)

def run_llama3_vllm_one_hop_dag_3_2():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)
    file = open(f"prompts/3.2.one_hop_dag.txt", "r", encoding="utf-8")
    prompt = file.read()
    for year in [2023, 2018, 2019, 2020, 2021, 2022]:
        df_path = Path(f"./outputs/llama3/one_hop_dag/3.1.one_hop_dag_{year}.csv")
        df = pd.read_csv(df_path)
        print(f"Year: {year}")
        print(f"Number of articles: {len(df)}")
        prompts = [f"{prompt_history}\n\nSystem: {df["answer_one_hop_dag"].values[i]}\n\nUser: {prompt}\n\nSystem: " for i, prompt_history in enumerate(df["prompts"].values)]
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        df["prompts"] = prompts
        df["answer_one_hop_dag_2"] = generated_texts
        df.to_csv(Path(f"./outputs/llama3/one_hop_dag/3.2.one_hop_dag_{year}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", help="inflation/deflation/change_of_prices/direction_of_change")
    args = parser.parse_args()
    if args.exp == "inflation":
        run_llama3_vllm_inflation_1_1()
    if args.exp == "deflation":
        run_llama3_vllm_deflation_1_2()
    if args.exp == "change_of_prices":
        run_llama3_vllm_change_in_prices_2_1()
    if args.exp == "direction_of_change":
        run_llama3_vllm_change_in_prices_2_2()
    if args.exp == "one_hop_dag":
        run_llama3_vllm_one_hop_dag_3_1()
    if args.exp == "one_hop_dag_2":
        run_llama3_vllm_one_hop_dag_3_2()