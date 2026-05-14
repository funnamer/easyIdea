import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv

_=load_dotenv()
# ===================== 常量定义 =====================
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

SYS_PROMPT = f"""
你是一个专业的旅游规划逻辑解析引擎。请根据用户提供的已知基础行程信息和自然语言补充要求，直接输出包含所有Python逻辑约束的JSON列表。

可用的函数库文档如下：
{FUNC_DOCS}
"""


# ===================== 核心处理函数 =====================

def format_to_messages(example):
    """
    将单条原始数据转换为标准的 Messages 格式
    """
    user_content = example.get("nature_language")
    target_list = example.get("hard_logic_py")

    assistant_content = json.dumps(target_list, ensure_ascii=False, indent=4)

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    return {"messages": messages}


def load_and_prepare_dataset(model_path, data_path):
    """
    封装完整的数据处理流水线
    """
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = 'right'
    # 2. 加载原始数据集
    raw_dataset = load_dataset("json", data_files=data_path, split="train")

    # 3. 处理数据 (关闭多进程避免死锁，4000条数据单核极快)
    processed = raw_dataset.map(
        format_to_messages,
        remove_columns=raw_dataset.column_names,
        num_proc=1
    )

    # 4. 打乱数据
    processed = processed.shuffle(seed=42)

    return processed, tokenizer


_current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_data_path = os.path.join(_current_dir,"data","train", "*.json")

print(_data_path)
model_id = os.environ.get("MODEL_PATH")
if not model_id:
    raise EnvironmentError(
        "未设置 MODEL_PATH 环境变量！请设置模型的绝对路径，例如：\n"
        "export MODEL_PATH=/root/autodl-tmp/model/Qwen/Qwen3-8B"
    )


processed_dataset, tokenizer = load_and_prepare_dataset(model_id, _data_path)

if __name__ == "__main__":
    print(f"\n✅ 成功加载模型: {model_id}")
    print(f"✅ 数据处理完毕！共计 {len(processed_dataset)} 条数据。")

    print("\n" + "=" * 30 + " 1. Messages 格式预览 " + "=" * 30)
    sample_msg = processed_dataset[0]["messages"]
    print(json.dumps(sample_msg, ensure_ascii=False, indent=2))

    print("\n" + "=" * 30 + " 2. Tokenizer 真实输入预览 " + "=" * 30)
    final_text = tokenizer.apply_chat_template(
        sample_msg,
        tokenize=False,
        add_generation_prompt=False
    )
    print(final_text)
    print("=" * 80)