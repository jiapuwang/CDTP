import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import json
import os
import re

# todo:1需要改 日志文件地址  配置日志记录器
logging.basicConfig(
    filename='/public/chengweiwu/mingyang/logs/1117/text_gen_Humanities_and_Society_glm-4-9b.log',
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
ckpt = torch.load('/public/chengweiwu/mingyang/sft/checkpoints/glm-4-9b_text_gen_Humanities_and_Society/epoch-2-339-hf/pytorch_model.bin')
model.load_state_dict(ckpt, strict=False)
model.eval()

# 记录开始时间
start_time = time.time()


# 删除最后一个 ")" 后紧跟的 ","
def remove_last_comma_after_parenthesis(text):
    # 使用正则表达式寻找最后一个 ")" 后紧跟 "," 的位置并删除 ","
    return re.sub(r'\)\s*,$', ')', text)


# todo:3需要改输入/输出文件
# 打开JSONL文件
with open('/public/chengweiwu/mingyang/data/input_data/submit/textGen_submit/textGen_all_Humanities_and_Society_random_1w1_head1w_dev2k.jsonl', 'r',
          encoding='utf-8') as input_file, \
        open('/public/chengweiwu/mingyang/data/output_data/1117/text_gen_output_Humanities_and_Society_glm-4-9b.jsonl',
             'w',
             encoding='utf-8') as output_file:
    for line in input_file:
        data = json.loads(line)
        # ######### 文本生成
        str = ""
        for element in data['triples']:
            lists = element.split("\t")
            if len(lists) < 3:
                print(f"Error: lists 长度不足 3，实际长度为 {len(lists)}。数据为：{lists}")
                continue
            list_str = "(" + lists[0] + "," + lists[1] + "," + lists[2] + "),"
            str = str + list_str
        submit_str = remove_last_comma_after_parenthesis(str)
        query = ((
                     "三元组是知识图谱中的一种结构化表示形式，用于描述实体之间的关系。一个三元组由三个元素组成：[(头实体,关系,尾实体)]。"
                     "目标文本生成旨在根据输入的结构化三元组生成与三元组对应的自然语言文本。"
                     "你是文本生成领域的专家，请基于输入三元组：[(头实体,关系,尾实体)]生成目标文本，"
                     "以下是一些需要生成目标文本的实例，要求生成的目标文本简洁，只需给出生成的目标文本，无需做任何解释，生成的目标文本以<SOD>开始<EOD>结束。"
                     "实例：三元组：[(9·11国家纪念博物馆,组成,国家纪念馆和国家博物馆),(9·11国家纪念博物馆,地理位置,世贸中心遗址),(9·11国家纪念博物馆,目的,纪念在9·11事件中失去生命的美国人民)]"
                     "生成的文本：<SOD>9·11国家纪念博物馆是为了纪念在9·11事件中失去生命的美国人民，它分为国家纪念馆和国家博物馆两个部分，位于纽约“9·11”恐怖袭击的发生地——世贸中心遗址。<EOD>"
                     "实例：三元组：[(Biography\"John Travolta,导演，Adam Friedman),(Biography\"John Travolta,主演,Halle Berry)]"
                     "生成的文本：<SOD>《\"Biography\"John Travolta》是由Adam Friedman执导，Halle Berry、Nicolas Cage主演的一部影片。<EOD>"
                     "实例：三元组：[(范例表示,外文名,case representation),(范例表示,所属学科,信息科学技术),(范例表示,公布年度,2008年)]"
                     "生成的文本：<SOD>范例表示（case representation）是2008年公布的信息科学技术名词。<EOD>"
                     "实例：三元组：[(矩磁材料,归属,磁性材料)]"
                     "生成的文本：<SOD>矩磁材料是具有矩形磁滞回线、剩余磁感强度Br和工作时最大磁感应强度Bm的比值,即Br/Bm接近于1以及矫顽磁力较小的磁性材料。<EOD>"
                     "实例：三元组：[(一种困惑感,导演,克劳德·朱特),(一种困惑感,编剧,克劳德·朱特、Clément Perron),(一种困惑感,主演,Jacques Gagnon、Lyne Champagne)]"
                     "生成的文本：<SOD>《一种困惑感》是由克劳德·朱特执导的电影，由克劳德·朱特、Clément Perron担任编剧，Jacques Gagnon、Lyne Champagne等主演。<EOD>"
                     "实例：三元组：[") + submit_str + "] 生成的文本：")
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

