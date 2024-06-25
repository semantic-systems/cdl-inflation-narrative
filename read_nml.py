import gzip
import glob
from tqdm import tqdm
import pandas as pd

headline_start_strings = '<headline brand-display="DJ" >'
headline_end_strings = '</headline>'
body_start_strings = '<text>'
body_end_strings = '</text>'
accession_number_start_strings = 'accession-number="'
accession_number_end_strings = '" '
djn_subject_start_string = "<djn-subject>"
djn_subject_end_string = "</djn-subject>"
date_start_strings = 'display-date="'
date_end_strings = '" >'
included_subject_code = ["N/DJIB", "N/DJG", "N/GPRW", "N/DJAN", "AWSJ", "WSJE", "N/PREL", "N/NRG", "N/DJBN", "N/AWP", "N/BRNS", "N/JNL", "N/WAL", "N/WLS", "N/WSJ"]

replace_tokens = [("</p>\n<pre>\n \n", ""),
                  (" \n </pre>\n<p>\n", "."),
                  ("<text>\n<pre>\n \n", ""),
                  ("</pre>\n<p>\n  (END) Dow Jones Newswires</p>\n<p>\n  ", ""),
                  ("</p>\n</text>\n", ""),
                  ("</p>\n<p>\n    ", ""),
                  ("<text>\n<p>\n   ", ""),
                  (" </p>\n<p>\n ", "")]


years = list(range(1984, 2024))

for year in tqdm(years):
    headlines = []
    bodies = []
    accession_numbers = []
    djn_subject_codes = []
    displaydates = []
    print(year)
    files = glob.glob(f"./data/DJN/{year}*.nml.gz")

    for file in files:
        with gzip.open(file,'rt', encoding="iso-8859-1") as f:
            save_this_line_as_body = False
            save_this_line_as_subject_code = False
            for line in f:
                # meta
                if accession_number_start_strings in line:
                    accession_number = line.split(accession_number_start_strings)[-1].split(accession_number_end_strings)[0]
                    accession_numbers.append(accession_number)
                if date_start_strings in line:
                    displaydate = line.split(date_start_strings)[-1].split(date_end_strings)[0]
                    subject_code = ""
                # subject code
                if djn_subject_start_string in line:
                    save_this_line_as_subject_code = True
                if save_this_line_as_subject_code:
                    subject_code += line
                if djn_subject_end_string in line:
                    djn_subject_codes.append(subject_code)
                    save_this_line_as_subject_code = False
                # headline
                if headline_end_strings in line:
                    headlines.append(line.split(headline_end_strings)[0])
                    body = ""
                # body
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
    inflation_mentioned_accession_numbers = []
    inflation_mentioned_subject_codes = []

    for i, body in enumerate(bodies):
        if ("inflation" in body) or ("prices" in body) or ("inflation" in headlines[i]) or ("prices" in headlines[i]):
            inflation_mentioned_dates.append(displaydates[i])
            inflation_mentioned_headlines.append(headlines[i])
            inflation_mentioned_bodies.append(bodies[i])
            inflation_mentioned_accession_numbers.append(accession_numbers[i])
            inflation_mentioned_subject_codes.append(djn_subject_codes[i])

    inflation_mentioned_news = {"date": inflation_mentioned_dates,
                                "accession_number": inflation_mentioned_accession_numbers,
                                "subject_code": inflation_mentioned_subject_codes,
                                "headline": inflation_mentioned_headlines,
                                "body": inflation_mentioned_bodies}
    df = pd.DataFrame.from_dict(inflation_mentioned_news)
    df.to_csv(f'./data/inflation_mentioned_news_{year}.csv')
