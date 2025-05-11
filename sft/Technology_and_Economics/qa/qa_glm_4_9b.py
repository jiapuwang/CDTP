import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import json
import os

# todo:1需要改 日志文件地址  配置日志记录器
logging.basicConfig(
    filename='/public/chengweiwu/mingyang/logs/1120/qa_Technology_and_Economics_glm-4-9b.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

# todo:2需要改 cuda卡
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
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
).eval()

# 记录开始时间
start_time = time.time()

# todo:3需要改输入/输出文件
# 打开JSONL文件
with open('/public/chengweiwu/mingyang/data/input_data/submit/qa_submit/qa_all_Technology_and_Economics_random_1w1_head1w_dev2k.jsonl', 'r',
          encoding='utf-8') as input_file, \
        open('/public/chengweiwu/mingyang/data/output_data/1120/qa_output_Technology_and_Economics_glm-4-9b.jsonl',
             'w',
             encoding='utf-8') as output_file:
    for line in input_file:
        data = json.loads(line)
        query = (("请回答下边的问题，要求答案根据选项的准确程度给出排序，答案简洁，无需做任何解释，答案以<SOD>开始<EOD>结束。"
                  "问题：(±)反式菊酸的分子式是什么？A、C9H16O2, B、C10H16O2, C、C11H16O2, D、C11H16O3, E、C12H18O2, F、C10H15O2, G、C10H16O3, H、C9H15O2, I、C11H15O2, J、C10H17O2，答案：<SOD>B、C10H16O2, J、C10H17O2, D、C11H16O3, F、C10H15O2, I、C11H15O2, A、C9H16O2, H、C9H15O2, C、C11H16O2, E、C12H18O2, G、C10H16O3<EOD>"
                  "问题：11·15马鲁古海地震的地震级数是几级？A、6.2, B、8.9, C、7.2, D、3.5, E、7.5, F、6.8, G、7.0, H、5.9, I、8.2, J、7.3，答案：<SOD>B、8.9, F、6.8, H、5.9, C、7.2, D、3.5, I、8.2, G、7.0, A、6.2, E、7.5, J、7.3<EOD>"
                  "问题：破山击出自哪部作品？A、海贼王, B、龙珠, C、破坏王, D、火影忍者, E、死神, F、全职猎人, G、妖精的尾巴, H、进击的巨人, I、银魂, J、勇者斗恶龙，答案：<SOD>C、破坏王, F、全职猎人, H、进击的巨人, B、龙珠, I、银魂, E、死神, A、海贼王, J、勇者斗恶龙, D、火影忍者, G、妖精的尾巴<EOD>"
                  "问题：龙芯2号的研制单位是哪个？A、中国科学研究院自动化所, B、中国科学研究院软件所, C、中国科学研究院计算技术研究所, D、中国科学研究院化工所, E、中国社科院, F、清华大学, G、北京大学, H、复旦大学, I、上海交通大学, J、浙江大学，答案：<SOD>C、中国科学研究院计算技术研究所, B、中国科学研究院软件所, D、中国科学研究院化工所, A、中国科学研究院自动化所, E、中国社科院, F、清华大学, J、浙江大学, G、北京大学, I、上海交通大学, H、复旦大学<EOD>")
                 + "问题：" + data["response"] + "，答案：")

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