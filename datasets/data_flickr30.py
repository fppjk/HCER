import pandas as pd
import json
import ast # 用于安全地解析字符串中的列表

def process_csv_and_generate_json(csv_filepath, output_json_filepath="test_flickr30k.json"):
    """
    读取CSV文件，筛选 'split' == 'test' 的行，处理 'raw' 列，
    并生成以 'filename' 为键的 JSON 格式文件。

    Args:
        csv_filepath (str): 输入的 CSV 文件路径。
        output_json_filepath (str): 输出的 JSON 文件路径 (默认: "output.json")。
    """
    try:
        # 1. 读取 CSV 文件
        # 假设 CSV 文件是逗号分隔，并有表头
        df = pd.read_csv(csv_filepath)
        print(f"✅ 成功读取文件: {csv_filepath}")
        print(f"原始列名: {df.columns.tolist()}")

        # 2. 筛选出 split == 'test' 的行
        df_test = df[df['split'] == 'test'].copy()
        print(f"✅ 筛选 'split' == 'test' 后，剩余 {len(df_test)} 行数据。")

        # 3. 舍弃 sentids 列
        if 'sentids' in df_test.columns:
            df_test.drop(columns=['sentids'], inplace=True)
            print("✅ 舍弃 'sentids' 列。")
        else:
            print("⚠️ 'sentids' 列不存在，跳过舍弃。")

        # 4. 处理 raw 列数据
        # raw 列的数据格式是 '["句子1","句子2",...,"句子5"]'
        def parse_raw_data(raw_str):
            # ast.literal_eval 安全地将字符串解析为 Python 结构（列表）
            try:
                # 尝试解析成列表
                data_list = ast.literal_eval(raw_str)
                # 确保解析结果是列表，并且将其转换为句子的字符串列表
                if isinstance(data_list, list):
                    return [str(item) for item in data_list]
                return [] # 解析失败或不是列表，返回空列表
            except (ValueError, SyntaxError) as e:
                # 如果解析失败，可能是因为数据格式不标准，记录错误或返回默认值
                # print(f"解析错误: {e} for data: {raw_str}")
                return []

        df_test['raw_list'] = df_test['raw'].apply(parse_raw_data)
        print("✅ 成功处理 'raw' 列，将其转换为 Python 句子列表。")

        # 5. 生成 JSON 格式文件
        # 准备最终的字典结构
        json_output = {}
        for index, row in df_test.iterrows():
            filename = row['filename']
            # 创建一个包含所需数据的字典
            data_entry = {
                "raw": row['raw_list'],
                "split": row['split'],
                "img_id": row['img_id']
                # 其他列可以根据需要添加
            }
            # 使用 filename 作为键
            json_output[filename] = data_entry

        # 将字典写入 JSON 文件
        with open(output_json_filepath, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 确保中文字符正确写入
            json.dump(json_output, f, indent=4, ensure_ascii=False)

        print(f"\n✨ 任务完成! JSON 文件已成功生成到: {output_json_filepath}")
        print(f"文件包含 {len(json_output)} 个条目。")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {csv_filepath}")
    except KeyError as e:
        print(f"❌ 错误: CSV 文件中缺少必要的列。请检查列名是否包含 'raw', 'sentids', 'split', 'filename', 'img_id'。缺少的列: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


# 请将 'your_input_file.csv' 替换为您实际的 CSV 文件路径
csv_file = './data/flickr30k/flickr_annotations_30k.csv'
output_file = './datasets/test_flickr30k.json'

# 调用函数执行操作
if __name__ == "__main__":
    process_csv_and_generate_json(csv_file, output_file)