import os

def merge_files(folder_path, output_file="merged_files.txt"):
    """
    将指定文件夹中的.py、.yml、.json文件合并到一个txt文件中
    每个文件内容前会加上文件名作为开头
    
    参数:
        folder_path: 要处理的文件夹路径
        output_file: 输出的txt文件名，默认是"merged_files.txt"
    """
    # 确保文件夹路径存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    # 支持的文件扩展名
    supported_extensions = ('.py', '.yml', '.json')
    
    # 获取文件夹中所有支持的文件
    files_to_merge = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                files_to_merge.append(file_path)
    
    if not files_to_merge:
        print(f"在文件夹 '{folder_path}' 中没有找到 .py, .yml 或 .json 文件")
        return
    
    # 合并文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in files_to_merge:
            filename = os.path.basename(file_path)
            print(f"正在处理: {filename}")
            
            # 写入文件名作为开头
            outfile.write(f"===== {filename} =====\n")
            
            # 读取并写入文件内容
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n\n")  # 在文件之间添加空行分隔
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {str(e)}")
                outfile.write(f"[错误: 无法读取此文件 - {str(e)}]\n\n")
    
    print(f"合并完成，共处理了 {len(files_to_merge)} 个文件")
    print(f"结果已保存到: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # 在这里替换为你的文件夹路径
    target_folder = "/home/dell/Project-HCL/BaseLine/flexdm_pt/prompt/topytorch"  # 可以是相对路径或绝对路径
    merge_files(target_folder)