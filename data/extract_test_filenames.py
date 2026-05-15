#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 test 目录的前 x 个文件名（不含后缀）写入到 txt 文件中
"""

import os
import sys


def extract_filenames(test_dir, output_file, count):
    """
    提取 test 目录的前 count 个文件名（不含后缀）并写入输出文件
    
    Args:
        test_dir: test 目录路径
        output_file: 输出 txt 文件路径
        count: 要提取的文件数量
    """
    # 获取 test 目录下的所有文件
    files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    
    # 按文件名排序
    files.sort()
    
    # 取前 count 个文件
    selected_files = files[:count]
    
    # 去除文件后缀，只保留文件名
    filenames_without_ext = [os.path.splitext(f)[0] for f in selected_files]
    
    # 写入到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename in filenames_without_ext:
            f.write(filename + '\n')
    
    print(f"已将 {len(filenames_without_ext)} 个文件名写入到 {output_file}")


if __name__ == "__main__":
    # 默认参数
    default_count = 100  # 默认提取前100个文件
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置路径
    test_dir = os.path.join(script_dir, 'test_phase2')
    output_file = os.path.join(script_dir, 'phase2_split.txt')
    
    # 从命令行参数获取文件数量，如果没有提供则使用默认值
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print(f"错误: 请输入有效的数字")
            sys.exit(1)
    else:
        count = default_count
        print(f"未指定文件数量，使用默认值: {count}")
    
    # 检查 test 目录是否存在
    if not os.path.exists(test_dir):
        print(f"错误: test 目录不存在: {test_dir}")
        sys.exit(1)
    
    # 执行提取
    extract_filenames(test_dir, output_file, count)
