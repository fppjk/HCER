import pandas as pd
import json
import ast

def process_csv_and_generate_json(csv_filepath, output_json_filepath="test_flickr30k.json"):
    """
    Read the CSV file, filter rows where 'split' == 'test', process the 'raw' column, 
    and generate a JSON file with 'filename' as the key. 
    
    Args:
        csv_filepath (str): Path to the input CSV file.  
        output_json_filepath (str): Path to the output JSON file (default: "output.json").
    """
    try:
        # 1. READ CSV FILES
        df = pd.read_csv(csv_filepath)
        print(f"File read successfully: {csv_filepath}")
        print(f"Original column name: {df.columns.tolist()}")

        # 2.  split == 'test' 
        df_test = df[df['split'] == 'test'].copy()
        print(f"After filtering out the rows where 'split' equals 'test', the remaining data consists of {len(df_test)} rows.")

        # 3. Discard the "sentids" column
        if 'sentids' in df_test.columns:
            df_test.drop(columns=['sentids'], inplace=True)
            print("Discard the "sentids" column")
        else:
            print("⚠️ 'sentids' NOT FOUND。")

        # 4.  raw data
        def parse_raw_data(raw_str):
            try:
                data_list = ast.literal_eval(raw_str)
                if isinstance(data_list, list):
                    return [str(item) for item in data_list]
                return [] 
            except (ValueError, SyntaxError) as e:
                # print(f"ERROR: {e} for data: {raw_str}")
                return []

        df_test['raw_list'] = df_test['raw'].apply(parse_raw_data)


        json_output = {}
        for index, row in df_test.iterrows():
            filename = row['filename']

            data_entry = {
                "raw": row['raw_list'],
                "split": row['split'],
                "img_id": row['img_id']
            }
            json_output[filename] = data_entry


        with open(output_json_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)

        print(f"\n OUTPUT_JSON_FILEPATH is : {output_json_filepath}")
        print(f"{len(json_output)}。")

    except FileNotFoundError:
        print(f"❌ Error: File not found {csv_filepath}")
    except KeyError as e:
        print(f"❌ Error: The CSV file is missing necessary columns. Please verify if the column names include 'raw', 'sentids', 'split', 'filename', and 'img_id'. The missing columns are...: {e}")
    except Exception as e:
        print(f"❌ An unknown error occurred: {e}")


csv_file = './data/flickr30k/flickr_annotations_30k.csv'
output_file = './datasets/test_flickr30k.json'


if __name__ == "__main__":
    process_csv_and_generate_json(csv_file, output_file)
