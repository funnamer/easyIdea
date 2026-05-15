import json
import os
import glob


def convert_hard_logic_format(input_file, output_file=None):
    """
    将 hard_logic_py 字段从字符串格式转换为真正的JSON数组格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件
    """
    # 读取原始文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查是否存在 hard_logic_py 字段且是字符串类型
    if 'hard_logic_py' in data and isinstance(data['hard_logic_py'], str):
        try:
            # 尝试将字符串解析为JSON数组
            parsed_logic = json.loads(data['hard_logic_py'])
            # 替换为解析后的数组
            data['hard_logic_py'] = parsed_logic
            print(f"成功转换: {input_file}")
        except json.JSONDecodeError as e:
            print(f"解析失败 {input_file}: {e}")
            return False
    
    # 确定输出文件路径
    if output_file is None:
        output_file = input_file
    
    # 写入转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return True


def batch_convert_directory(directory_path, pattern="*.json"):
    """
    批量转换目录下所有匹配的JSON文件
    
    Args:
        directory_path: 目录路径
        pattern: 文件匹配模式
    """
    # 获取所有匹配的JSON文件
    json_files = glob.glob(os.path.join(directory_path, pattern))
    
    success_count = 0
    fail_count = 0
    
    for file_path in json_files:
        if convert_hard_logic_format(file_path):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n转换完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")


if __name__ == "__main__":
    # 示例用法：
    
    # 1. 转换单个文件
    single_file = "/Users/funnamer/Desktop/agentic_oss/llm_dsl_ft/data/test/20250320170206819941.json"
    if os.path.exists(single_file):
        convert_hard_logic_format(single_file)
        print(f"单文件转换完成: {single_file}")
    
    # 2. 批量转换整个目录
    # test_dir = "/Users/funnamer/Desktop/agentic_oss/llm_dsl_ft/data/test"
    # batch_convert_directory(test_dir)
    
    # 3. 批量转换res目录
    res_dir = "/Users/funnamer/Desktop/agentic_oss/llm_dsl_ft/res2"
    batch_convert_directory(res_dir)
