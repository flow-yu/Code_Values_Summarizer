import json
import re
import math

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# load dict from json file
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def add_line_numbers(code: str) -> str:
    lines = code.split('\n')
    in_multiline_comment = False
    result = []
    line_number = 1

    for line in lines:
        stripped_line = line.strip()
        
        # Check if we are in a multiline comment
        if stripped_line.startswith('"""') and stripped_line.endswith('"""') and len(stripped_line) > 3:
            result.append(line)
            continue
        elif stripped_line.startswith('"""'):
            in_multiline_comment = not in_multiline_comment
            result.append(line)
            continue
        elif in_multiline_comment:
            result.append(line)
            continue
        
        # Skip empty lines, import statements, from...import statements, and single-line comments
        if not stripped_line or stripped_line.startswith('import') or (stripped_line.startswith('from') and 'import' in stripped_line) or stripped_line.startswith('#'):
            result.append(line)
        else:
            result.append(f"{line}    ##line:({line_number})")
            line_number += 1

    return '\n'.join(result)


#元素必是 str类型
def get_rerun_list(logger_file_path):
    with open(logger_file_path, 'r') as file:
        lines = file.readlines()
    
    rerun_list = []
    rerun_list_found = False
    
    # Check if the rerun_list is explicitly mentioned at the end
    for line in reversed(lines):
        if line.startswith("The rerun_list:") or line.startswith("The incorrect_form_list:"):
            rerun_list_found = True
            rerun_list_str = line.strip().split(": ")[1]
            if rerun_list_str:
                rerun_list = rerun_list_str.strip("[]").split(", ")
                rerun_list = [elem.strip("'") for elem in rerun_list]
            break
    
    # If rerun_list is not found, collect unique IDs from the Output file lines
    if not rerun_list_found:
        output_file_pattern = re.compile(r"Output file:.*?PDBscript_main_tests_(\d+)\.py")
        ids_set = set()
        
        for line in lines:
            match = output_file_pattern.search(line)
            if match:
                ids_set.add(match.group(1))
        
        rerun_list = list(ids_set)
    
    return rerun_list

def check_backslash_usage(py_path: str) -> bool:
    with open(py_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        stripped_line = line.rstrip()
        if stripped_line.endswith('\\'):
            return True

    return False

def is_logfile_normal(log_file_path, py_path):
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]# 去掉每行末尾的空白符
            while lines and lines[-1] == '':
                lines.pop()
        # 检查grammer error
        if len(lines) < 4:
            #/ error
            if check_backslash_usage(py_path):
                return -2
            
            return -1
        
        # 检查是否完成debug
        if lines[-4].strip() == '--Return--':
        
            if lines[-3].strip() != '> <string>(1)<module>()->None':
                return 0
            
            # 检查包含的属性字典中是否有'__return__': None
            if "'__return__': None" not in lines[-2]:
                return 0
        else:
            if "self.quitting = True" not in lines[-3]:
                return 0
            
            if "TypeError: 'dict' object is not callable" not in lines[-2]:
                return 0
        return 1
    
    except Exception as e:
        print(f"Error reading logfile: {e}")
        return 0
    
def get_logger_length(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]# 去掉每行末尾的空白符
            while lines and lines[-1] == '':
                lines.pop()
        return len(lines)
    except Exception as e:
        print(f"Error reading logfile: {e}")
        return 0    
    
import re

def has_from_import_statement(response):
    """
    Checks if a code snippet contains a 'from ... import ...' statement.
    
    Args:
        code_snippet (str): The code snippet to check.
    
    Returns:
        bool: True if a 'from ... import ...' statement is found, False otherwise.
    """
    # Regular expression to match 'from ... import ...' statements
    match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    if match:
        solution_code = match.group(1)      
    else:
        solution_code = ""

    pattern = re.compile(r'^\s*from\s+\w+(\.\w+)*\s+import\s+.+', re.MULTILINE)
    
    # Search for the pattern in the code snippet
    match1 = pattern.search(solution_code)
    
    return bool(match1)

def compute_bleu_2gram(predictions, references):
    assert len(predictions) == len(references), "Predictions and references must be of the same length."
    
    total_ref_len = 0
    total_pred_len = 0
    
    # Counters for matches and total n-grams in predictions
    correct_1gram = 0
    guess_1gram = 0
    
    correct_2gram = 0
    guess_2gram = 0
    
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        
        # Update lengths
        total_ref_len += len(ref_tokens)
        total_pred_len += len(pred_tokens)
        
        # Count 1-gram matches
        ref_unigrams = {}
        for w in ref_tokens:
            ref_unigrams[w] = ref_unigrams.get(w, 0) + 1
        
        pred_unigrams = {}
        for w in pred_tokens:
            pred_unigrams[w] = pred_unigrams.get(w, 0) + 1
        
        # Matches for 1-grams
        for w, cnt in pred_unigrams.items():
            correct_1gram += min(cnt, ref_unigrams.get(w, 0))
        
        guess_1gram += len(pred_tokens)
        
        # Count 2-gram matches
        ref_bigrams = {}
        for i in range(len(ref_tokens)-1):
            bg = (ref_tokens[i], ref_tokens[i+1])
            ref_bigrams[bg] = ref_bigrams.get(bg, 0) + 1
        
        pred_bigrams = {}
        for i in range(len(pred_tokens)-1):
            bg = (pred_tokens[i], pred_tokens[i+1])
            pred_bigrams[bg] = pred_bigrams.get(bg, 0) + 1
        
        # Matches for 2-grams
        for bg, cnt in pred_bigrams.items():
            correct_2gram += min(cnt, ref_bigrams.get(bg, 0))
        
        guess_2gram += max(len(pred_tokens)-1, 0)
    
    # Compute precisions
    p1 = (correct_1gram / guess_1gram) if guess_1gram > 0 else 0.0
    p2 = (correct_2gram / guess_2gram) if guess_2gram > 0 else 0.0
    
    # Compute brevity penalty
    if total_pred_len == 0:
        # If prediction is empty, BLEU = 0
        return [0.0, p1*100, p2*100]
    
    ratio = total_ref_len / total_pred_len
    if ratio > 1.0:
        # candidate shorter than reference
        bp = math.exp(1 - ratio)
    else:
        # candidate length >= reference length
        bp = 1.0
    
    # Compute BLEU: geometric mean of p1 and p2
    # BLEU = BP * exp((log p1 + log p2)/2)
    if p1 == 0.0 or p2 == 0.0:
        bleu = 0.0
    else:
        bleu = bp * math.exp((math.log(p1) + math.log(p2))/2)
    
    # Convert to percentages
    return [bleu*100, p1*100, p2*100]