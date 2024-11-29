from rouge import Rouge
import numpy as np
import json
from transformers import T5ForConditionalGeneration, AutoTokenizer

# 假设eval_data是一个列表，其中每个元素是一个元组（input_context, correct_output）


with open("/home/amax/Unit_Test/projectByCo/JSON/split_data_validate_new.json", "r") as file:
    eval_data = json.load(file) 
for item in eval_data:
    item['input'] += '<extra_id_0>'
print(f"  ====> DataSet: {len(eval_data)}")



def evaluate_model(eval_data, model, tokenizer, device):
    rouge = Rouge()
    total_matches = 0
    all_rouge_scores = []
    
    cou = 1
    for item in eval_data:
        if cou <  11280:
            cou+=1
            continue
        input_context = item['input']
        correct_output = item['output']
        inputs = tokenizer.encode(input_context, max_length=512, padding="max_length",truncation=True,return_tensors="pt").to(device)

        outputs = model.generate(inputs, max_length=100)
        predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 精确匹配率（忽略大小写）
        if predicted_output.lower() == correct_output.lower():
            total_matches += 1

        # ROUGE分数
        scores = rouge.get_scores(predicted_output, correct_output)
        print(f"[Index]: {cou}")
        print(f"[Input]:")
        print(f"{input_context}")

        print(f"[correct_output]:")
        print(f"{correct_output}")

        print(f"[predict]:")
        print(f"{predicted_output}")
        print(f"    Scores: {scores}")
        cou+=1
        all_rouge_scores.append(scores[0])  # 获取第一个（通常是 ROUGE-1, ROUGE-2, ROUGE-L）

    # 计算精确匹配率
    exact_match_ratio = total_matches / len(eval_data)

    # 计算平均ROUGE分数（分别计算 ROUGE-1, ROUGE-2, ROUGE-L）
    avg_rouge_scores = {}
    for metric in all_rouge_scores[0]:  # ROUGE-1, ROUGE-2, ROUGE-L 等
        avg_rouge_scores[metric] = np.mean([score[metric]['f'] for score in all_rouge_scores])

    return exact_match_ratio, avg_rouge_scores


checkpoint = "/data1/amax/CodeT5_FT/CodeT5_2b/output_dir"
# 加载模型和tokenizer
model = T5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
device = "cuda"
model.to(device)

# 执行评估
exact_match_ratio, avg_rouge_scores = evaluate_model(eval_data, model, tokenizer, device)
print(f"Exact Match Ratio: {exact_match_ratio}")
print(f"Average ROUGE Scores: {avg_rouge_scores}")



