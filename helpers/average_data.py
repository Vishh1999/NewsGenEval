import pandas as pd
import numpy as np
import json

metric_names = ['Compression_Ratio_results', 'Novel_N_Gram_results',
                'Distinct_Score_results', 'Readability_results',
                'Jaccard_Similarity_results',
                'ROUGE_results', 'BERT_results',
                'Keyword_Recall_results', 'Q2_results',
                'LLM_as_a_Judge_results']


def average_data_by_group(group, bin, metric_names):
    df = pd.read_json("wikinews_data_results_V2.json")
    if bin != 'All':
        df = df[df[group] == bin]
    short_df_dict = df.to_dict(orient='records')

    # Initialize 10 empty lists, one per metric
    acc = [[] for _ in metric_names]

    # Collect DataFrames
    for d in short_df_dict:  # loop through dicts
        for i, m in enumerate(metric_names):
            acc[i].append(pd.DataFrame(d[m]))

    metric_df_list = []
    for _l in acc:
        df = pd.concat(_l)
        if 'version' in df.columns:
            avg_df = df.groupby("version", as_index=False).mean(numeric_only=True)
            metric_df_list.append(avg_df)

    summary_df = pd.concat([df.set_index("version") for df in metric_df_list],
                           axis=1
                           ).reset_index()
    metric_df_list = [summary_df] + metric_df_list
    metric_names = ["Combined_results"] + metric_names

    # Write to one Excel file
    with pd.ExcelWriter(f"{bin}_aggregated_results.xlsx", engine="xlsxwriter") as writer:
        for df, name in zip(metric_df_list, metric_names):
            df.to_excel(writer, sheet_name=name, index=False)

    return pd.concat(metric_df_list)


for bin_name in ['Short', 'Medium', 'Long', 'All']:
    average_data_by_group('word_count_bin', bin_name, metric_names)

for bin_name in ['Business & Technology',
                 'Entertainment',
                 'Sports',
                 'Science & Environment',
                 'Politics & Policy']:
    average_data_by_group('category', bin_name, metric_names)
