import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import logging
import json
import os

# 配置日志记录器
logging.basicConfig(
    filename='/public/chengweiwu/mingyang/logs/1117/kg_History_and_Politics_deepseek_7b.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 设置 GPU 编号
os.environ['CUDA_VISIBLE_DEVICES'] = "6"

# 模型名称
MODEL_NAME = "/public/chengweiwu/models/cdtp/deepseek-llm-7b-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载Tokenizer和Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

ckpt = torch.load('/public/chengweiwu/mingyang/sft/checkpoints/deepseek-llm-7b-base_kg_History_and_Politics/epoch-2-333-hf/pytorch_model.bin')
model.load_state_dict(ckpt, strict=False)
model.eval()

# 设置生成参数，包括填充标记的ID
model.generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# 记录开始时间
start_time = time.time()

# 输入/输出文件配置
input_file_path = '/public/chengweiwu/mingyang/data/input_data/submit/kg_submit/kg_all_History_and_Politics_random1w1_head1w_dev2k.jsonl'
output_file_path = '/public/chengweiwu/mingyang/data/output_data/1117/kg_output_History_and_Politics_deepseek_7b.jsonl'

# 打开JSONL文件
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
        open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in input_file:
        data = json.loads(line)
        # 创建问题格式
        query = ((
                     "知识图谱是由三元组构成的，其中三元组形式为 (头实体,关系,尾实体)。 知识图谱补全是对缺失三元组推断出缺失的信息。以下是一些需要补全的实例，以及对应的十个选项，"
                     "你是知识图谱补全的专家，请给出?处缺失的信息，并根据你认为的十个选项的准确程度给出排序，答案简洁，无需做任何解释，答案以<SOD>开始<EOD>结束。"
                     "实例：(赖上被上帝遗忘的天使,作者,?) ?处缺失的信息：A、芈夏, B、芈秋, C、芈冬, D、芈春, E、芈梅, F、芈兰, G、芈菊, H、芈荷, I、芈雪, J、芈露，答案：<SOD>F、芈兰, B、芈秋, A、芈夏, E、芈梅, C、芈冬, J、芈露, D、芈春, H、芈荷, I、芈雪, G、芈菊<EOD>"
                     "实例：(老字号网店,主办,?) ?处缺失的信息：A、上海老字号协会, B、北京新字号协会, C、北京老字号协会, D、上海新字号协会, E、南京老字号协会, F、南京新字号协会, G、广州新字号协会, H、广州老字号协会, I、西安新字号协会, J、西安老字号协会，答案：<SOD>C、上海老字号协会, I、西安新字号协会, D、上海新字号协会, F、南京新字号协会, B、北京新字号协会, E、南京老字号协会, G、广州新字号协会, H、广州老字号协会, A、北京老字号协会, J、西安老字号协会<EOD>"
                     "实例：(中华人民共和国外交部和亚美尼亚共和国外交部磋商议定书,签订地点,?) ?处缺失的信息：A、上海, B、北京, C、深圳, D、广州, E、南京, F、西安, G、杭州, H、武汉, I、重庆, J、成都，答案：<SOD>A、北京, F、西安, B、上海, D、广州, E、南京, G、杭州, C、深圳, I、重庆, J、成都, H、武汉,<EOD>"
                     "实例：(泉水寨,总面积,?) ?处缺失的信息：A、39平方公里, B、29平方公里, C、19平方公里, D、9平方公里, E、49平方公里, F、59平方公里, G、69平方公里, H、79平方公里, I、89平方公里, J、99平方公里，答案：<SOD>C、29平方公里, A、9平方公里, G、69平方公里, D、39平方公里, F、59平方公里, E、49平方公里, H、79平方公里, B、19平方公里, I、89平方公里, J、99平方公里<EOD>"
                     "实例：(林乐伦,出生地,?) ?处缺失的信息：A、湖北衡东县, B、湖南衡东县, C、河北衡东县, D、河南衡东县, E、山东衡东县, F、山西衡东县, G、广东衡东县, H、广西衡东县, I、江西衡东县, J、江苏衡东县，答案：<SOD>I、江西衡东县, D、河南衡东县, C、湖北衡东县, F、山西衡东县, A、湖南衡东县, E、山东衡东县, G、广东衡东县, B、湖北衡东县, H、广西衡东县, J、江苏衡东县<EOD>"
                     "实例：") + data['post']['knowledge_map_completion'] + " ?处缺失的信息：" + data[
                     "response"] + "，答案：")
        print("==============>" + query)

        # 预处理输入
        inputs = tokenizer(query, return_tensors="pt").to(device)

        # 生成参数配置
        gen_kwargs = {"max_new_tokens": 2500, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(answer)

            # 记录日志
            logging.info(answer)
            data["model_answer"] = answer
            logging.info(f"=====data=====>：{data}")
            output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            output_file.flush()  # 刷新缓冲区，将数据写入文件

# 记录结束时间
end_time = time.time()
total_time = end_time - start_time

# 添加总运行时间到日志
logging.info(f"Total runtime: {total_time:.2f} seconds")