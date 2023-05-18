import glob
import json
import os

import pandas as pd


def to_excel(data):
    df = pd.DataFrame(data=data)
    # df_property_list = df[['idRequest', 'properties']]
    # df_clause_list = df[['idRequest', 'clauses']]
    # df_intervener_list = df[['idRequest', 'interveners']]
    # df_period_list = df[['idRequest', 'periods']]

    # df_date = df[['idRequest', 'documentDate', 'capital', 'protocol', 'notary']]
    # df_date.to_excel("output/irph_single.xlsx", index=False)

    print("Dictionary converted into excel...")
    df.to_excel(f"output/{JSON_NAME}.xlsx", index=False)


def run(json_path):
    try:
        df_json_list = list()
        json_list = sorted(glob.glob(os.path.join(json_path, "*.json")))
        for f_json in json_list:
            with open(f_json, 'r', encoding="utf-8") as outfile:
                doc_json = json.load(outfile)
                if doc_json:
                    df_json_list.append(doc_json)
        # json_str = json.dumps(df_json_list)
        # json_str.to_excel()
        to_excel(df_json_list)
    except Exception as e:
        print("ERROR: " + str(e))


if __name__ == '__main__':
    JSON_NAME = ''
    JSON_PATH = rf''
    run(JSON_PATH)
