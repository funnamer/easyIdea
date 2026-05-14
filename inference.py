import torch
import json
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 从环境变量获取模型路径（绝对路径）
MODEL_ID = os.environ.get("MODEL_PATH")
if not MODEL_ID:
    raise EnvironmentError(
        "未设置 MODEL_PATH 环境变量！请设置模型的绝对路径，例如：\n"
        "export MODEL_PATH=/root/autodl-tmp/model/Qwen/Qwen3-8B"
    )
LORA_PATH = "./qwen3-8b-finetune/final_checkpoint"
DATA_DIR = "../chinatravel/data/phase1"
SPLIT_FILE = "../chinatravel/evaluation/default_splits/tpc_aic_phase1.txt"
OUTPUT_DIR = "../chinatravel/generate_hard_logic/llm_phase1"

# 1. 加载 Tokenizer 和 基座模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 挂载你微调好的 LoRA 权重
model = PeftModel.from_pretrained(base_model, LORA_PATH)
# model = base_model
model.eval()

# 3. 读取ID列表
with open(SPLIT_FILE, 'r', encoding='utf-8') as f:
    ids = [line.strip() for line in f if line.strip()]

print(f"共找到 {len(ids)} 个测试样本")

# 4. 准备系统提示词
FUNC_DOCS = """
(1)day_count(plan)
文档(Docs)：获取行程(plan)中的天数。
返回值(Return)：整数(int)

(2)people_count(plan)
文档(Docs)：获取行程(plan)中的人数。
返回值(Return)：整数(int)

(3)start_city(plan)
文档(Docs)：获取行程(plan)的出发城市。
返回值(Return)：字符串(str)

(4)target_city(plan)
文档(Docs)：获取行程(plan)的目标城市。
返回值(Return)：字符串(str)

(5)allactivities(plan)
文档(Docs)：获取行程(plan)中的所有活动。
返回值(Return)：活动列表(list of activities)

(6)allactivities_count(plan)
文档(Docs)：获取行程(plan)中的活动数量。
返回值(Return)：整数(int)

(7)dayactivities(plan, day)
文档(Docs)：获取特定天数(day，取值为[1, 2, 3, ...])内的所有活动。
返回值(Return)：活动列表(list of activities)

(8)activity_cost(activity)
文档(Docs)：获取特定活动(activity)的费用，不包含交通费用。
返回值(Return)：浮点数(float)

(9)activity_position(activity)
文档(Docs)：获取特定活动(activity)的具体地点或具体名称，比如住宿类活动的名称、餐厅名称、活动名称。如果想获得活动的具体类型，请对餐厅类活动用restaurant_type(activity, target_city(plan))，对景点类活动用attraction_type(activity, target_city(plan))，对住宿类活动用accommodation_type(activity, target_city(plan))。如果没有匹配这三个函数，可以用activity_position(activity)。
返回值(Return)：字符串(str)

(10)activity_price(activity)
文档(Docs)：获取特定活动(activity)的价格，该价格为单人价格。
返回值(Return)：浮点数(float)

(11)activity_type(activity)
文档(Docs)：获取特定活动(activity)的类型，类型包含['breakfast'(早餐), 'lunch'(午餐), 'dinner'(晚餐), 'attraction'(景点), 'accommodation'(住宿), 'train'(火车), 'airplane'(飞机)]。
返回值(Return)：字符串(str)

(12)activity_tickets(activity)
文档(Docs)：获取特定活动(activity)所需的票数，适用于['attraction'(景点), 'train'(火车), 'airplane'(飞机)]类型的活动。
返回值(Return)：整数(int)

(13)activity_transports(activity)
文档(Docs)：获取特定活动(activity)的交通信息。
返回值(Return)：字典列表(list of dict)

(14)activity_start_time(activity)
文档(Docs)：获取特定活动(activity)的开始时间。
返回值(Return)：字符串(str)

(15)activity_end_time(activity)
文档(Docs)：获取特定活动(activity)的结束时间。
返回值(Return)：字符串(str)

(16)activity_time(activity)
文档(Docs)：获取特定活动(activity)的持续时长。
返回值(Return)：整数(int)，单位为分钟(minutes)

(17)innercity_transport_cost(transports)
文档(Docs)：获取市内交通(transports)的总费用。
返回值(Return)：浮点数(float)

(18)poi_recommend_time(city, poi)
文档(Docs)：获取某城市(city)内特定兴趣点(poi)的推荐游览时长，目前仅支持景点类型的兴趣点。
返回值(Return)：整数(int)，单位为分钟(minutes)

(19)poi_distance(city, poi1, poi2)
文档(Docs)：获取某城市(city)内两个兴趣点(poi1、poi2)之间的距离。
返回值(Return)：浮点数(float)，单位为千米(km)

(20)innercity_transport_price(transports)
文档(Docs)：获取市内交通(transports)的价格，该价格为单人价格。
返回值(Return)：浮点数(float)

(21)innercity_transport_distance(transports)
文档(Docs)：获取市内交通(transports)的行程距离。
返回值(Return)：浮点数(float)，单位为千米(km)

(22)metro_tickets(transports)
文档(Docs)：若交通类型为地铁，获取地铁票的数量。
返回值(Return)：整数(int)

(23)taxi_cars(transports)
文档(Docs)：若交通类型为出租车，获取出租车的数量，出租车数量按公式`(people_count(plan) + 3) // 4`计算。
返回值(Return)：整数(int)

(24)room_count(activity)
文档(Docs)：获取住宿类活动(activity)的房间数量。
返回值(Return)：整数(int)

(25)room_type(activity)
文档(Docs)：获取住宿类活动(activity)的房间类型，1代表单人房(single room)，2代表双人房(double room)，取值只能是1或2，禁止使用“大床房”“双床房”或其他表述。
返回值(Return)：整数(int)

(26)restaurant_type(activity, target_city)
文档(Docs)：获取目标城市(target_city)内餐厅类活动(activity)的菜系类型，返回值必须来自['云南菜', '西藏菜', '东北菜', '烧烤', '亚洲菜', '粤菜', '西北菜', '闽菜', '客家菜', '快餐简餐', '川菜', '台湾菜', '其他', '清真菜', '小吃', '西餐', '素食', '日本料理', '江浙菜', '湖北菜', '东南亚菜', '湘菜', '北京菜', '韩国料理', '海鲜', '中东料理', '融合菜', '茶馆/茶室', '酒吧/酒馆', '创意菜', '自助餐', '咖啡店', '本帮菜', '徽菜', '拉美料理', '鲁菜', '新疆菜', '农家菜', '海南菜', '火锅', '面包甜点', '其他中餐']。
返回值(Return)：字符串(str)

(27)attraction_type(activity, target_city)
文档(Docs)：获取目标城市(target_city)内景点类活动(activity)的类型，返回值必须来自['博物馆/纪念馆', '美术馆/艺术馆', '红色景点', '自然风光', '人文景观', '大学校园', '历史古迹', '游乐园/体育娱乐', '图书馆', '园林', '其它', '文化旅游区', '公园', '商业街区']。
返回值(Return)：字符串(str)

(28)accommodation_type(activity, target_city)
文档(Docs)：获取目标城市(target_city)内住宿类活动(activity)的特色，用于判断其是否符合用户需求，返回值必须来自['儿童俱乐部', '空气净化器', '山景房', '私汤房', '四合院', '温泉', '湖畔美居', '电竞酒店', '温泉泡汤', '行政酒廊', '充电桩', '设计师酒店', '民宿', '湖景房', '动人夜景', '行李寄存', '中式庭院', '桌球室', '私人泳池', '钓鱼', '迷人海景', '园林建筑', '老洋房', '儿童泳池', '历史名宅', '棋牌室', '智能客控', '情侣房', '小而美', '特色住宿', '茶室', '亲子主题房', '多功能厅', '洗衣房', '客栈', '自营亲子房', '停车场', 'Boss推荐', '江河景房', '日光浴场', '自营影音房', '厨房', '空调', '网红泳池', '别墅', '免费停车', '洗衣服务', '窗外好景', '酒店公寓', '会议厅', '家庭房', '24小时前台', '商务中心', '提前入园', '农家乐', '智能马桶', '美食酒店', 'SPA', '拍照出片', '海景房', '泳池', '影音房', '管家服务', '穿梭机场班车', '桑拿', '机器人服务', '儿童乐园', '健身室', '洗衣机', '自营舒睡房', '宠物友好', '电竞房', '位置超好', '套房']。
返回值(Return)：字符串(str)

(29)innercity_transport_type(transports)
文档(Docs)：获取市内交通(transports)的类型，返回值必须来自['metro'(地铁), 'taxi'(出租车), 'walk'(步行)]。
返回值(Return)：字符串(str)

(30)innercity_transport_start_time(transports)
文档(Docs)：获取市内交通(transports)的出发时间。
返回值(Return)：字符串(str)

(31)innercity_transport_end_time(transports)
文档(Docs)：获取市内交通(transports)的到达时间。
返回值(Return)：字符串(str)

(32)intercity_transport_type(activity)
文档(Docs)：获取城际交通类活动(activity)的类型，返回值必须来自['train'(火车), 'airplane'(飞机)]。
返回值(Return)：字符串(str)

(33)innercity_transport_time(transports)
文档(Docs)：获取市内交通(transports)的行程时长。
返回值(Return)：整数(int)，单位为分钟(minutes)

(34)intercity_transport_origin(activity)
文档(Docs)：获取城际交通类活动(activity)的出发城市。
返回值(Return)：字符串(str)

(35)intercity_transport_destination(activity)
文档(Docs)：获取城际交通类活动(activity)的到达城市。
返回值(Return)：字符串(str)
"""
system_prompt = f"""You are a travel planning assistant. 
Translate the user's natural language request into a specific JSON format with logical constraints.
{FUNC_DOCS}
You must output ONLY valid JSON containing 'start_city', 'target_city', 'days', 'people_number', and 'hard_logic_py' (a list of python string expressions).
"""

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5. 遍历所有ID进行推理
for idx, sample_id in enumerate(ids, 1):
    print(f"\n{'=' * 60}")
    print(f"处理第 {idx}/{len(ids)} 个样本: {sample_id}")

    # 断点续传：检查结果文件是否已存在
    output_path = os.path.join(OUTPUT_DIR, f"{sample_id}.json")
    if os.path.exists(output_path):
        print(f"跳过: 结果文件已存在 - {output_path}")
        continue

    # 读取对应的JSON文件
    json_path = os.path.join(DATA_DIR, f"{sample_id}.json")
    if not os.path.exists(json_path):
        print(f"警告: 文件不存在 - {json_path}")
        continue

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 使用与训练一致的输入：自然语言描述
    user_input = data.get("nature_language", "")

    user_query = f"{user_input}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 生成回答
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,  # 代码生成任务，调低温度，越低越稳定
            do_sample=False  # 使用贪心解码
        )

    # 截取新生成的内容
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    print(f"输入数据:\n{user_input}")
    print(f"\n模型输出:\n{response}")

    # 保存结果到指定目录
    output_path = os.path.join(OUTPUT_DIR, f"{sample_id}.json")

    # 提取JSON部分：移除可能的思维链标签
    json_str = response.strip()

    # 如果存在<think>标签，提取其后的内容
    if '<think>' in json_str and '</think>' in json_str:
        try:
            json_str = json_str.split('</think>', 1)[1].strip()
        except:
            pass

    # 尝试找到JSON对象的开始和结束位置
    start_idx = json_str.find('{')
    end_idx = json_str.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = json_str[start_idx:end_idx + 1]

    try:
        # 尝试解析模型输出的JSON
        result_data = json.loads(json_str)
        # 将原始数据和模型生成的结果合并
        final_result = {
            "uid": sample_id,  # 添加uid字段
            "nature_language": data.get("nature_language", ""),
            "start_city": result_data.get("start_city", ""),
            "target_city": result_data.get("target_city", ""),
            "days": result_data.get("days", 0),
            "people_number": result_data.get("people_number", 0),
            "hard_logic_py": result_data.get("hard_logic_py", [])
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")
    except json.JSONDecodeError as e:
        print(f"警告: 模型输出不是有效的JSON格式 - {str(e)}")
        print(f"尝试解析的内容:\n{json_str}")
        # 如果解析失败，保存原始数据+空列表
        error_result = {
            "uid": sample_id,  # 添加uid字段
            "nature_language": data.get("nature_language", ""),
            "start_city": data.get("start_city", ""),
            "target_city": data.get("target_city", ""),
            "days": data.get("days", 0),
            "people_number": data.get("people_number", 0),
            "hard_logic_py": [],
            "raw_response": response,
            "extracted_json": json_str,
            "error": f"Invalid JSON format from model: {str(e)}"
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)
        print(f"错误信息已保存到: {output_path}")

    print(f"{'=' * 60}")