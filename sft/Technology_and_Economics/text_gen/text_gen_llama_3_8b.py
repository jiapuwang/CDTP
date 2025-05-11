import re
import time
import torch
import transformers
import logging
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置日志记录器
logging.basicConfig(
    filename='/public/chengweiwu/mingyang/logs/1117/text_gen_Technology_and_Economics_llama_3_8b.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
MODEL_ID = "/public/chengweiwu/models/cdtp/Meta-Llama-3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
ckpt = torch.load('/public/chengweiwu/mingyang/sft/checkpoints1118/Meta-Llama-3-8B_text_gen_Technology_and_Economics/epoch-2-339-hf/pytorch_model.bin')
model.load_state_dict(ckpt, strict=False)
model.eval()

# 使用transformers的管道接口
pipeline = transformers.pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.bfloat16
)

# 记录开始时间
start_time = time.time()

def remove_last_comma_after_parenthesis(text):
    # 使用正则表达式寻找最后一个 ")" 后紧跟 "," 的位置并删除 ","
    return re.sub(r'\)\s*,$', ')', text)

# 输入/输出文件配置
input_file_path = '/public/chengweiwu/mingyang/data/input_data/submit/textGen_submit/textGen_all_Technology_and_Economics_random_1w1_head1w_dev2k.jsonl'
output_file_path = '/public/chengweiwu/mingyang/data/output_data/1117/text_gen_output_Technology_and_Economics_llama_3_8b.jsonl'

with open(input_file_path, 'r', encoding='utf-8') as input_file, \
        open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in input_file:
        data = json.loads(line)
        # 创建问题格式
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

        # 生成响应
        response = pipeline(query, max_length=2500, do_sample=True)

        # 提取回答
        answer = response[0]['generated_text']
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