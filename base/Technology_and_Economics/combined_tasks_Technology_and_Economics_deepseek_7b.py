import multiprocessing
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import logging
import json
import os
import re

def configure_logging(log_file):
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['CUDA_VISIBLE_DEVICES'] = "4"

def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer, model, device

def process_task(input_file_path, output_file_path, model, tokenizer, device, query_template):
    start_time = time.time()
    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            query = query_template(data)
            inputs = tokenizer(query, return_tensors="pt").to(device)
            gen_kwargs = {"max_new_tokens": 2500, "do_sample": True, "top_k": 1}
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(answer)
                logging.info(answer)
                data["model_answer"] = answer
                logging.info(f"=====data=====>：{data}")
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                outfile.flush()  # 刷新缓冲区，将数据写入文件
    logging.info(f"Task completed in {time.time() - start_time} seconds.")

def qa_template(data):
    query = (("请回答下边的问题，要求答案根据选项的准确程度给出排序，答案简洁，无需做任何解释，答案以<SOD>开始<EOD>结束。"
                      "问题：(±)反式菊酸的分子式是什么？A、C9H16O2, B、C10H16O2, C、C11H16O2, D、C11H16O3, E、C12H18O2, F、C10H15O2, G、C10H16O3, H、C9H15O2, I、C11H15O2, J、C10H17O2，答案：<SOD>B、C10H16O2, J、C10H17O2, D、C11H16O3, F、C10H15O2, I、C11H15O2, A、C9H16O2, H、C9H15O2, C、C11H16O2, E、C12H18O2, G、C10H16O3<EOD>"
                      "问题：11·15马鲁古海地震的地震级数是几级？A、6.2, B、8.9, C、7.2, D、3.5, E、7.5, F、6.8, G、7.0, H、5.9, I、8.2, J、7.3，答案：<SOD>B、8.9, F、6.8, H、5.9, C、7.2, D、3.5, I、8.2, G、7.0, A、6.2, E、7.5, J、7.3<EOD>"
                      "问题：破山击出自哪部作品？A、海贼王, B、龙珠, C、破坏王, D、火影忍者, E、死神, F、全职猎人, G、妖精的尾巴, H、进击的巨人, I、银魂, J、勇者斗恶龙，答案：<SOD>C、破坏王, F、全职猎人, H、进击的巨人, B、龙珠, I、银魂, E、死神, A、海贼王, J、勇者斗恶龙, D、火影忍者, G、妖精的尾巴<EOD>"
                      "问题：龙芯2号的研制单位是哪个？A、中国科学研究院自动化所, B、中国科学研究院软件所, C、中国科学研究院计算技术研究所, D、中国科学研究院化工所, E、中国社科院, F、清华大学, G、北京大学, H、复旦大学, I、上海交通大学, J、浙江大学，答案：<SOD>C、中国科学研究院计算技术研究所, B、中国科学研究院软件所, D、中国科学研究院化工所, A、中国科学研究院自动化所, E、中国社科院, F、清华大学, J、浙江大学, G、北京大学, I、上海交通大学, H、复旦大学<EOD>")
                     + "问题：" + data["response"] + "，答案：")
    return query

def text_gen_template(data):
    str = ""
    for element in data['triples']:
        lists = element.split("\t")
        list_str = "(" + lists[0] + "," + lists[1] + "," + lists[2] + "),"
        str = str + list_str
    submit_str = re.sub(r'\)\s*,$', ')', str)

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
    return query

def kg_template(data):
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
    return query

def main():
    model_path = "/public/chengweiwu/models/cdtp/deepseek-llm-7b-base"
    configure_logging('/public/chengweiwu/mingyang/logs/1112/combined_tasks_Technology_and_Economics_deepseek_7b.log')

    tokenizer, model, device = load_model(model_path)

    tasks = [
        {'input_file': '/public/chengweiwu/mingyang/data/input_data/submit/qa_submit/qa_all_Technology_and_Economics_random_1w1_head1w_dev2k.jsonl',
         'output_file': '/public/chengweiwu/mingyang/data/output_data/1112/qa_output_Technology_and_Economics_deepseek_7b.jsonl',
         'template': qa_template},
        {'input_file': '/public/chengweiwu/mingyang/data/input_data/submit/textGen_submit/textGen_all_Technology_and_Economics_random_1w1_head1w_dev2k.jsonl',
         'output_file': '/public/chengweiwu/mingyang/data/output_data/1112/text_gen_output_Technology_and_Economics_deepseek_7b.jsonl',
         'template': text_gen_template},
        {'input_file': '/public/chengweiwu/mingyang/data/input_data/submit/kg_submit/kg_all_Technology_and_Economics_random1w1_head1w_dev2k.jsonl',
         'output_file': '/public/chengweiwu/mingyang/data/output_data/1112/kg_output_Technology_and_Economics_deepseek_7b.jsonl',
         'template': kg_template}
    ]

    # 使用 multiprocessing.Pool 并行运行所有任务
    with multiprocessing.Pool(processes=4) as pool:
        results = []
        for task in tasks:
            results.append(
                pool.apply_async(process_task,
                                 (task['input_file'], task['output_file'], model, tokenizer, device, task['template'])))

        # 等待所有任务完成并处理结果上
        for result in results:
            try:
                result.get()  # 意外的错误将在这里抛出
            except Exception as e:
                logging.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()