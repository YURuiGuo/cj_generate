
import os
import random
import json

from Fun_Utils.infer_mutliGPUs import process_candidates_and_generate_files
from Fun_Utils.utils import Delete_TestCode, Insert_Point, Code_CompeletProcess, Code_FirstPreProcess



def process_content(file_content, file_path):
    c =  ''.join(file_content)
    # 删除注释以及多余空行
    first_pass_code = Code_FirstPreProcess(c)

    comp_code = Code_CompeletProcess(first_pass_code)

    del_code = Delete_TestCode(comp_code) # 删除所有@Test代码段

    res, numbers= Insert_Point(del_code,file_path)

    return res, numbers


def get_allcandidate(c_path):

    print(">>> Get All Candidate File ...")

    candidate_files = []

    insert_num = 0

    for root, _, files in os.walk(c_path):
        for file in files:
            if file.endswith('.cj'):  # 先筛选以 .cj 结尾的文件，再查找以 main() 开头的行
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.readlines() 
                        for line in file_content:
                            if line.strip().startswith("main()"):
                                content, c = process_content(file_content, file_path)
                                if content == None:
                                    break
                                insert_num = insert_num + c
                                candidate_files.append(content)
                                # print(content)
                                break 
                except (IOError, UnicodeDecodeError):
                    print(f"Error reading file: {file_path}")  # 文件读取失败

    print(f">>> Insert Points: {insert_num}")
    return candidate_files

def main():
    print("Hello, Python!")

    corpus_path = r"/data1/amax/CodeT5_FT/projectByCo/Corpus"
    save_path = r"/data1/amax/CodeT5_FT/Generate_file/files"
    checkpoint = "/data1/amax/CodeT5_FT/CodeT5_2b/output_dir"

    candidate_file_path = r"/data1/amax/CodeT5_FT/Generate_file/code/candidate_files_list.json"
    if os.path.exists(candidate_file_path):
        with open(candidate_file_path, 'r') as f:
            candidate_file = json.load(f)
    else: 
        candidate_file = get_allcandidate(corpus_path)
        random.shuffle(candidate_file)
        
        with open(candidate_file_path, 'w') as f:
            json.dump(candidate_file, f, indent=4)
        # >>> Insert Points: 3951
    
    # 处理候选文件并生成代码
    print(">>> Generate Seed File ...")
    process_candidates_and_generate_files(candidate_file, checkpoint, save_path)
    print("Processing completed.")

if __name__ == "__main__":
    main()




