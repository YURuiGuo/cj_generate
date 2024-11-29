import json
import re
import numpy as np

def extract_data_from_file(file_path):

    scores = []

    capture_output = False
    capture_predict = False

    total_matches = 0

    output_lines = []
    predict_lines = []

    with open(file_path, 'r') as file:
        for line in file:
            # Start capturing output
            if line.strip() == "[correct_output]:":
                capture_output = True
                capture_predict = False
                continue
            # Start capturing predict
            if line.strip() == "[predict]:":
                capture_predict = True
                capture_output = False
                continue
            # Stop capturing output and save it
            if capture_output and line.startswith("[predict]:"):
                capture_output = False
                continue
            # Stop capturing predict and save it
            if capture_predict and line.startswith("    Scores: "):
                capture_predict = False

                if ''.join(predict_lines).strip().lower() == ''.join(output_lines).strip().lower():
                    total_matches += 1
                output_lines = []
                predict_lines = []
                
                new_line = line.split('    Scores: ')[1].replace("'", '"')
                list_from_string = json.loads(new_line)
                scores.append(list_from_string[0])
                continue

            # Capture lines for output
            if capture_output:
                output_lines.append(line)
            # Capture lines for predict
            if capture_predict:
                predict_lines.append(line)
            
            

    print(f"{len(scores)}")


    # 计算精确匹配率
    exact_match_ratio = total_matches / len(scores)

    # 计算平均ROUGE分数（分别计算 ROUGE-1, ROUGE-2, ROUGE-L）
    avg_rouge_scores = {}
    for metric in scores[0]:  # ROUGE-1, ROUGE-2, ROUGE-L 等
        avg_rouge_scores[metric] = np.mean([score[metric]['f'] for score in scores])

    return exact_match_ratio, avg_rouge_scores




# 假设文件路径是 'data.txt'
file_path1 = r"/data1/amax/CodeT5_FT/evalinfo.tex"
file_path2 = r"/data1/amax/CodeT5_FT/evalinfo2.tex"



# 打开源文件和目标文件
with open(file_path2, 'r') as source_file, open(file_path1, 'a') as dest_file:
    # 读取源文件内容并追加到目标文件
    content = source_file.read()
    dest_file.write(content)


exact_match_ratio, avg_rouge_scores = extract_data_from_file(file_path1)
print(f"Exact Match Ratio: {exact_match_ratio}")
print(f"Average ROUGE Scores: {avg_rouge_scores}")

# print("Scores:", scores)
# print("correct_output:", correct_output)
# print("predict:", predict)

# 11279
# Exact Match Ratio: 0.3014451635783314
# Average ROUGE Scores: {'rouge-1': 0.6034104230232546, 'rouge-2': 0.41215999570818174, 'rouge-l': 0.6005521591715218}


# Exact Match Ratio: 0.010882502975684407
# Average ROUGE Scores: {'rouge-1': 0.5818202690726011, 'rouge-2': 0.3780254678993411, 'rouge-l': 0.576475366508176}
# 483
# Exact Match Ratio: 0.2650103519668737
# Average ROUGE Scores: {'rouge-1': 0.5818202690726011, 'rouge-2': 0.3780254678993411, 'rouge-l': 0.576475366508176}

# 合并后的最终结果：
# 11762
# Exact Match Ratio: 0.2999489882673015
# Average ROUGE Scores: {'rouge-1': 0.602523835337643, 'rouge-2': 0.4107582802744401, 'rouge-l': 0.5995634590476996}

# 结果分析：
# Exact Match Ratio (EMR)： 
#           生成的Code 与 目标Code是否完全匹配， 有30%完全匹配，在代码补全任务中不算高，也不算特别低
# ROUGE Scores：
# （1）ROUGE-1 衡量的是生成代码与参考代码之间的单词级别匹配：
#               ROUGE-1 高于 0.5 是一个值得肯定的成绩，说明模型能够在生成代码时匹配到大部分关键字和标识符
# （2）ROUGE-2 生成代码和参考代码中二元词汇对的匹配:
#               不是一个非常高的分数，尤其是对于复杂代码补全任务，通常期望这个分数接近 0.5 或更高。如果二元匹配较低，可能是因为生成的代码在语法结构或代码逻辑上与参考代码存在差异。
#  (3) ROUGE-L ROUGE-L 衡量的是最长公共子序列（LCS）的匹配程度，关注生成代码与参考代码的整体结构是否一致。
#               结果是中等偏上的，通常对于代码补全任务，ROUGE-L 得分越高越好，理想情况下希望接近 0.7 或以上。ROUGE-L 低于 0.6 可能意味着模型在保持代码逻辑顺序和结构方面还有改进空间。