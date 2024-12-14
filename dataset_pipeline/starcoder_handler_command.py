import json
import re
import os
import subprocess
import time
from utile import *
import sys

# get main + test + combined code
def extract_and_save_python_code(response, test, id, output_dir):
    # Extract the Python code between the triple backticks
    match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    match1 = re.search(r'```python\n(.*?)\n```', test, re.DOTALL)
    
    if match:
        solution_code = match.group(1)      
    else:
        solution_code = ""
        # print(f"No Python code found in the {id}.")
        with open("total_logger_Nofound.txt", 'a') as f:
            f.write(f"No found error: {id}\n")
            f.write(f"No Python code found in the {id}.\n")
    if match1:
        test_code = match1.group(1)
    else:
        test_code = test
        with open("total_logger_Nofound.txt", 'a') as f:
            f.write(f"No found error: {id}\n")
            f.write(f"No test code found in the {id}.\n")

    output = os.path.join(output_dir,f"{id}")
    os.makedirs(output, exist_ok=True)    
    # Write the extracted code to a .py file
    filename = f'main_{id}.py'
    file_path = os.path.join(output, filename)
    with open(file_path, 'w') as file:
        file.write(solution_code)  
           
    filename = f'tests_{id}.py'
    file_path = os.path.join(output, filename)
    with open(file_path, 'w') as file:
        file.write(test_code) 

    filename = f'main_tests_{id}.py'
    file_path = os.path.join(output, filename)
    with open(file_path, 'w') as file:
        line_numbered_code = add_line_numbers(solution_code)
        file.write(line_numbered_code)
        file.write("\n\n ")
        file.write("### Here is the test code:\n")
        file.write(test_code)

def create_test_script(py_file_path):
    # 读取原Python文件的内容
    with open(py_file_path, 'r', encoding='utf-8') as file:
        original_content = file.readlines()
    
    indented_content = ['    ' + line for line in original_content]

    # 构建新的测试脚本内容
    tester_function = f"""
def tester():
{''.join(indented_content)}
"""

    main_function = """
if __name__ == "__main__":
    import pdb
    pdb.run("tester()")
"""

    test_script_content = tester_function + main_function

    # 确定新测试脚本的路径
    base_path, original_filename = os.path.split(py_file_path)
    new_filename = f"PDBscript_{original_filename}"
    new_file_path = os.path.join(base_path, new_filename)

    # 写入新的测试脚本
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(test_script_content)

def run_script(main_tests_file, output_file, logger_file, total_logger_dir):
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f"Running the script {output_file} now.")
    total_logger1 = os.path.join(total_logger_dir, "total_logger.txt")
    total_logger2 = os.path.join(total_logger_dir, "total_logger2.txt")
    with open(total_logger1, 'a') as total_log_file:
        total_log_file.write(f"Output file: {output_file}\n")
        total_log_file.write(f"Start time: {start_time_str}\n")
        # original_path = os.getcwd()
        try:
            # os.chdir("temp_files")
            subprocess.run(
                ['python', 'tree_handler2.py', main_tests_file, output_file, logger_file],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600  
            )
            # os.chdir(original_path)
            # print("Script output:", result.stdout.decode())
        except Exception as e:
            # os.chdir(original_path)
            end_time = time.time()
            end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            total_log_file.write(f"End time: {end_time_str}\n")
            total_log_file.write(f"Duration: {end_time - start_time:.2f} seconds\n")
            total_log_file.write(f"Status: Failed with exception: {e}\n\n")
            # print(f"An error occurred: {e}")
            
            # 记录到 total_logger2.txt 文件
            with open(total_logger2, 'a') as timeout_log_file:
                timeout_log_file.write(f"Output file: {output_file}\n")
                timeout_log_file.write(f"Start time: {start_time_str}\n")
                timeout_log_file.write(f"End time: {end_time_str}\n")
                timeout_log_file.write(f"Duration: {end_time - start_time:.2f} seconds\n")
                timeout_log_file.write(f"Status: Failed with exception: {e}\n\n")
            
            # print(f"Script {output_file} timed out after 300 seconds.")
        else:
            end_time = time.time()
            end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            total_log_file.write(f"End time: {end_time_str}\n")
            total_log_file.write(f"Duration: {end_time - start_time:.2f} seconds\n")
            total_log_file.write("Status: Completed\n\n")

# solve original dataset
def extract_and_save_script_tests(dataset_path, output_dir):
    jsons = read_jsonl_file(dataset_path)
    for problem_dict in jsons:
        id = problem_dict['id']
        if skipping_scripts(id, output_dir):
            continue
        extract_and_save_python_code(problem_dict['response'], problem_dict['tests'], id, output_dir)
        create_test_script(f"{output_dir}/{id}/main_tests_{id}.py")
        run_script(f"{output_dir}/{id}/main_tests_{id}.py", f"{output_dir}/{id}/PDBscript_main_tests_{id}.py", f"{output_dir}/{id}/logger_PDBscript_{id}.txt", output_dir)

def skipping_scripts(id, total_logger_dir):
    total_logger1 = os.path.join(total_logger_dir, "total_logger.txt")
    try:
        with open(total_logger1, 'r+') as total_log_file:
            file_content = total_log_file.read()
            id_searcher = f"_{id}.py"
            if id_searcher in file_content:
                print(f"Skipping the script {id}.")
                # total_log_file.write(f"SKIP the script {id}.\n")
                return True
            else:
                return False
    except FileNotFoundError:
        return False

if __name__ == "__main__":
    path_file = sys.argv[1]
    path_directory = sys.argv[2]
    extract_and_save_script_tests(path_file,path_directory)
