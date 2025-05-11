import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import json
import os

# todo:1需要改 日志文件地址  配置日志记录器
logging.basicConfig(
    filename='/public/chengweiwu/mingyang/logs/1117/kg_History_and_Politics_glm-4-9b.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

# todo:2需要改 cuda卡
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
# todo:5 model_path
MODEL_PATH = "/public/chengweiwu/models/cdtp/glm-4-9b"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
)
ckpt = torch.load('/public/chengweiwu/mingyang/sft/checkpoints/glm-4-9b_kg_History_and_Politics/epoch-2-333-hf/pytorch_model.bin')
model.load_state_dict(ckpt, strict=False)
model.eval()

# 记录开始时间
start_time = time.time()

# todo:3需要改输入/输出文件
# 打开JSONL文件
with (open('/public/chengweiwu/mingyang/data/input_data/submit/kg_submit/kg_all_History_and_Politics_random1w1_head1w_dev2k.jsonl', 'r',
          encoding='utf-8') as input_file, \
        open('/public/chengweiwu/mingyang/data/output_data/1117/kg_output_History_and_Politics_glm-4-9b.jsonl',
             'w',
             encoding='utf-8') as output_file):
    for line in input_file:
        data = json.loads(line)
        # 完成知识库补全任务1028
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

        inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                               add_generation_prompt=True,
                                               tokenize=True,
                                               padding='longest',
                                               return_tensors="pt",
                                               return_dict=True
                                               )

        inputs = inputs.to(device)

        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            try:
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
            except Exception as e:
                logging.error(f"模型生成过程中发生错误: {str(e)}")
                logging.error(f"生成的输入: {inputs}")
                continue
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            # logging.info(tokenizer.decode(outputs[0], skip_special_tokens=True))
            # 处理模型回复
            data["model_answer"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # logging.info(f"=====data=====>：{data}")
            output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            output_file.flush()

# 记录结束时间
end_time = time.time()
total_time = end_time - start_time

# 添加总运行时间到日志
logging.info(f"Total runtime: {total_time:.2f} seconds")
