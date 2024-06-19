import gzip
import glob
from tqdm import tqdm
import pandas as pd

headline_start_strings = '<headline brand-display="DJ" >'
headline_end_strings = '</headline>'
body_start_strings = '<text>'
body_end_strings = '</text>'
date_start_strings = 'display-date="'
date_end_strings = '" >'

replace_tokens = [("</p>\n<pre>\n \n", ""),
                  (" \n </pre>\n<p>\n", "."),
                  ("<text>\n<pre>\n \n", ""),
                  ("</pre>\n<p>\n  (END) Dow Jones Newswires</p>\n<p>\n  ", ""),
                  ("</p>\n</text>\n", ""),
                  ("</p>\n<p>\n    ", ""),
                  ("<text>\n<p>\n   ", ""),
                  (" </p>\n<p>\n ", "")]


years = list(range(2004, 2024))

for year in tqdm(years):
    headlines = []
    bodies = []
    displaydates = []
    print(year)
    files = glob.glob(f"./data/{year}*.nml.gz")

    for file in files:
        with gzip.open(file,'rt', encoding="iso-8859-1") as f:
            save_this_line_as_body = False
            for line in f:
                if date_start_strings in line:
                    displaydate = line.split(date_start_strings)[-1].split(date_end_strings)[0]
                if headline_end_strings in line:
                    headlines.append(line.split(headline_end_strings)[0])
                    body = ""
                if body_start_strings in line:
                    save_this_line_as_body = True
                if save_this_line_as_body:
                    body += line
                if body_end_strings in line:
                    #for old, new in replace_tokens:
                    #    body = body.replace(old, new)
                    bodies.append(body)
                    displaydates.append(displaydate)
                    save_this_line_as_body = False

    inflation_mentioned_dates = []
    inflation_mentioned_headlines = []
    inflation_mentioned_bodies = []

    for i, body in enumerate(bodies):
        if "inflation" in body:
            inflation_mentioned_dates.append(displaydates[i])
            inflation_mentioned_headlines.append(headlines[i])
            inflation_mentioned_bodies.append(bodies[i])

    inflation_mentioned_news = {"date": inflation_mentioned_dates,
                                "headline": inflation_mentioned_headlines,
                                "body": inflation_mentioned_bodies}
    df = pd.DataFrame.from_dict(inflation_mentioned_news)
    df.to_csv(f'./data/inflation_mentioned_news_{year}.csv')
