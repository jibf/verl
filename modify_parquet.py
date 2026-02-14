


import pandas as pd

def modify_parquet(input_file, output_file, column_name):

    df = pd.read_parquet(input_file)
    pattern = "Note: A rule-based method has identified the following objects as potentially involved in an accident:\nNo accident detected.\nHowever, these track IDs may be inaccurate and should only be used as a reference. Always prioritize observable visual evidence from the video for your final analysis.\n"
    for idx, data in enumerate(df["prompt"]):
        df["prompt"][idx][1]["content"] = data[1]["content"].replace(pattern, "")


    df.to_parquet(output_file, index=False)


if __name__ == "__main__":
    input_file = "/shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/test_with_normal_balanced.parquet"
    output_file = "/shared/jibf/data/MM_AU/CAP-DATA_chunks/qwen8b_results/test_with_normal_balanced_wo_rulebase.parquet"
    modify_parquet(input_file, output_file, "prompt")

