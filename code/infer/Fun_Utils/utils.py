import os
import re
import random


def Code_FirstPreProcess(source_code):
    # 移除注释、空白行 
    return_pass_code = re.sub(r'/\*[\s\S]*?\*/', '', source_code)
    return_pass_code = re.sub(r'(?<=\s)//.*$|^\s*//.*$', '', return_pass_code, flags=re.MULTILINE)
    return_pass_code = '\n'.join([line.rstrip() for line in return_pass_code.splitlines() if line.strip()])
    return return_pass_code


def Code_CompeletProcess(first_pass_code):

    second_pass_code = first_pass_code.replace('\r\n', '\n').replace('\r', '\n')     # 将所有换行符标准化为 \n
    second_pass_code = '\n'.join(line.rstrip() for line in second_pass_code.split('\n'))# 移除行首和行尾的空白字符
    second_pass_code = re.sub(r'\s+$', '', second_pass_code, flags=re.MULTILINE)# 移除行尾的空格
    second_pass_code = re.sub(r'\n\s*\n', '\n\n', second_pass_code)# 将多个连续空行减少为一个空行

    return second_pass_code

def is_balanced_brace(line):
    left_braces = line.count('{')
    right_braces = line.count('}')
    return left_braces, right_braces

def Delete_TestCode(code):
    lines = code.split('\n')
    result_lines = []
    skip_next_block = False 
    brace_balance = 0  # 大括号平衡计数器

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if '@Test' in line:# 检查是否为 @Test 注解
            skip_next_block = True
            i += 1  
            continue

        if skip_next_block:
            left_braces, right_braces = is_balanced_brace(lines[i])
            brace_balance += left_braces - right_braces
           
            if brace_balance == 0: # 如果大括号平衡恢复到 0，说明块结束
                skip_next_block = False
            i += 1
            continue

        result_lines.append(lines[i])
        i += 1

    return '\n'.join(result_lines)

def Insert_Point(code, file_path, coefficient=0.9):
    '''
        根据代码的长度随机选取插入点的数量，最终返回{"code": code, "point":[插入点对应的代码行]}，
        注意： 插入点不会选取import、 from开头的行。
    '''

    lines = code.split('\n')

    filtered_lines = []
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith(('import', 'from')) and not stripped_line.startswith('//') and not stripped_line.startswith('/*') and not stripped_line.endswith('*/'):
            filtered_lines.append(i+1)  # 记录合适的行号


    if len(filtered_lines) == 0:
        return None, 0 
            
    print(f"{filtered_lines}")
   
    min_length = max(1, len(filtered_lines))
    num_insert_points = random.randint(1, min_length)

    max_points = min(3, int(coefficient * num_insert_points))  # 系数coefficient控制插入点的数量
    num_insert_points = max(1, max_points)
    insert_points = random.sample(filtered_lines, num_insert_points)

    insert_points.sort()

    comment = f"// {file_path}, Insert points:{insert_points}"
    code_with_comment = '\n'.join( [comment] + lines)

    return {"code": code_with_comment, "points": insert_points}, len(insert_points)



if __name__ == "__main__":
    pass