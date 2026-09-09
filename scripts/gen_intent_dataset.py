# xx_school_rag/scripts/gen_intent_dataset.py
# 该脚本用于: 生成「意图识别」二分类训练集（JSONL）。
#   通用知识  -> 闲聊问候 / 与本校无关的常识 / 对助手本身的提问  -> 直接问 LLM
#   专业咨询  -> 与本校迎新/校务相关的具体咨询                    -> 走 RAG 检索
#
# 生成方式：模板槽位组合 + 手写自然问法（可选 --llm 用 DeepSeek 增广）。
# 输出: data/train_data/classify_data/new_student_intent.jsonl（两类严格 50/50，已打乱）
# 运行: python scripts/gen_intent_dataset.py [总条数 默认2500] [--llm]

import json
import os
import random
import sys

SEED = 42
DEFAULT_TOTAL = 2500

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "data", "train_data", "classify_data", "new_student_intent.jsonl")

# ======================================================================
# 专业咨询：校务主题 × 问句框架 + 手写自然问法
# ======================================================================
CAMPUS_TOPICS = [
    "报到", "新生报到", "报到地点", "报到时间", "报到材料", "报到手续",
    "宿舍", "宿舍分配", "宿舍号", "床铺尺寸", "宿舍门禁", "宿舍用电", "宿舍报修", "宿舍卫生",
    "食堂", "食堂营业时间", "食堂充值", "清真食堂",
    "校园卡", "校园卡充值", "校园卡挂失", "校园卡补办", "校园卡服务中心",
    "选课", "选课系统", "补选", "退选", "加课", "选课时间",
    "军训", "军训时间", "军训服装", "军训请假", "免军训", "军训汇演",
    "快递", "菜鸟驿站", "快递地址", "取件码",
    "图书馆", "借书", "还书", "图书馆开放时间", "研讨间预约",
    "社团", "百团大战", "社团招新",
    "学费", "学费缴纳", "奖学金", "助学金", "助学贷款", "生源地贷款", "绿色通道",
    "转专业", "辅修", "学生证", "校徽", "校历",
    "校医院", "医保", "城乡居民医保", "新生体检", "心理测评",
    "请假", "晚归", "门禁卡", "校园网", "网费", "电费",
    "校车", "班车", "地铁到校", "打印店", "文印中心",
    "团组织关系", "党组织关系", "个人档案", "户口迁移", "兵役登记",
    "开学典礼", "入学教育", "迎新晚会", "始业教育",
    "辅导员", "班主任", "学院办公室", "教务处", "宿管中心", "失物招领",
]
CAMPUS_FRAMES = [
    "{t}怎么办理", "{t}在哪里办", "{t}在哪儿", "{t}要带什么材料",
    "{t}什么时候开始", "{t}什么时候截止", "{t}几点到几点", "{t}的流程是什么",
    "关于{t}的规定", "新生{t}要注意什么", "{t}找谁咨询", "{t}要多少钱",
    "我想问一下{t}", "{t}的具体安排", "{t}可以延期吗", "{t}错过了怎么办",
    "{t}需要预约吗", "{t}联系电话是多少", "{t}周末能办吗", "{t}的注意事项",
    "咱们学校{t}怎么弄", "本校{t}是怎么规定的", "{t}这块谁负责", "{t}要去哪个部门",
]
CAMPUS_HANDWRITTEN = [
    "大一新生去哪里报到", "报到当天几点到校合适", "家长可以陪同进校报到吗",
    "报到需要带哪些证件", "外省考生当天赶得及报到吗", "报到迟到两天有影响吗",
    "宿舍是几人间", "宿舍怎么分配的", "床铺是多大尺寸", "宿舍晚上几点关门",
    "宿舍可以用电热水壶吗", "宿舍能带电脑吗", "宿舍能养宠物吗", "宿舍热水器坏了找谁",
    "学校有几个食堂", "食堂几点开门", "食堂能用支付宝吗", "食堂可以刷微信吗",
    "校园卡在哪里领", "校园卡怎么充值", "校园卡丢了怎么补办", "校园卡挂失去哪里",
    "什么时候开始选课", "选课系统网址是多少", "选课没选上怎么办", "怎么申请人工加课",
    "军训什么时候开始", "军训一共多少天", "军训要自己带什么东西", "身体不好能不能免军训",
    "军训服装是自己买吗", "军训能请假吗",
    "快递地址怎么写", "快递在哪里取件", "快递能免费放几天", "菜鸟驿站在哪",
    "从火车站怎么到学校", "从机场怎么去学校", "校门口有地铁吗",
    "图书馆怎么借书", "图书馆几点开门", "本科生最多能借几本书", "图书借了多久要还",
    "社团什么时候招新", "百团大战在哪里举办",
    "学费怎么交", "可以申请助学贷款吗", "家庭困难有绿色通道吗", "奖学金什么时候评",
    "大一能转专业吗", "怎么办理转专业", "学生证什么时候发",
    "校医院在什么位置", "新生要做体检吗", "医保怎么参保", "心理测评是必须做的吗",
    "怎么请假", "晚归会被怎么处理", "门禁卡丢了怎么办",
    "校园网怎么连", "宿舍要交电费吗", "校内有校车吗",
    "党团组织关系怎么转", "个人档案要自己带吗", "要不要办户口迁移",
    "开学典礼什么时候", "迎新晚会在哪看", "辅导员怎么联系", "教务处电话是多少",
    "军训汇演家长能来看吗", "失物招领处在哪里", "校内打印店在哪",
    "新生群怎么加", "班级qq群在哪找", "专业分流什么时候", "有没有新生手册电子版",
    "开学第一周有什么安排", "什么时候正式上课", "课表在哪里查", "空教室怎么查",
]

# ======================================================================
# 通用知识：槽位组合 + 手写
# ======================================================================
CHITCHAT = [
    "你好", "您好", "hi", "hello", "哈喽", "在吗", "在不在", "有人吗", "早上好", "晚上好",
    "你是谁", "你叫什么", "你叫什么名字", "你是什么", "你是机器人吗", "你是AI吗",
    "你是真人吗", "你能做什么", "你会什么", "你能帮我什么忙", "你有什么功能", "怎么用你",
    "谢谢", "谢谢你", "多谢", "太感谢了", "辛苦了", "麻烦你了", "再见", "拜拜", "晚安",
    "你好厉害", "你真棒", "厉害啊", "哈哈哈", "嗯嗯", "好的", "行吧", "收到", "明白了",
    "在干嘛", "你困了吗", "你累不累", "陪我聊聊天", "讲个笑话吧", "我好无聊", "有点累",
    "今天心情不好", "心情有点烦", "你几岁了", "你是男生还是女生", "你喜欢什么", "夸夸我",
    "随便说点什么", "跟我说说话", "你在吗在吗",
]
CONCEPTS = [
    "机器学习", "区块链", "云计算", "人工智能", "大数据", "元宇宙", "量子计算",
    "光合作用", "牛顿第二定律", "相对论", "熵", "GDP", "通货膨胀", "复利", "供求关系",
    "递归", "闭包", "多态", "死锁", "哈希表", "红黑树", "正则表达式", "垃圾回收",
    "碳中和", "温室效应", "厄尔尼诺", "板块构造", "黑洞", "暗物质", "基因编辑",
    "凯恩斯主义", "机会成本", "边际效用", "囚徒困境", "马太效应", "幸存者偏差",
]
TRANSLATE = [
    "苹果", "谢谢", "图书馆", "人工智能", "早上好", "我爱你", "生日快乐",
    "多少钱", "厕所在哪", "我不明白", "再来一杯", "祝你好运",
]
HOWTO = [
    "煮米饭", "煎鸡蛋", "泡一杯好咖啡", "去除衣服上的油渍", "叠衣服更省空间",
    "快速入睡", "缓解焦虑", "克服拖延", "背单词", "提高专注力", "练好普通话",
    "写好一封邮件", "做思维导图", "整理房间", "省钱", "理财入门", "开始跑步",
    "保护视力", "缓解颈椎疼", "戒掉手机瘾",
]
RECOMMEND_TYPES = ["科幻", "悬疑", "历史", "经济学入门", "心理学", "励志", "编程入门", "散文"]
RECOMMEND_FRAMES = ["推荐几本{x}的书", "有什么好看的{x}电影", "{x}方面有什么经典作品", "入门{x}看什么书好"]
KNOWLEDGE = [
    "光速是多少", "水的沸点是多少度", "一年有多少天", "地球的半径是多少",
    "珠穆朗玛峰有多高", "长城有多长", "中国有多少个省", "太阳系有几大行星",
    "为什么天空是蓝色的", "彩虹是怎么形成的", "闪电是怎么产生的", "月亮为什么有阴晴圆缺",
    "四季是怎么形成的", "潮汐是怎么回事", "长江和黄河哪个长", "世界上最大的沙漠是哪个",
    "秦始皇是哪个朝代的", "第一次世界大战是哪一年", "相对论是谁提出的", "DNA是什么",
    "为什么会打哈欠", "为什么海水是咸的", "北极和南极哪个更冷", "人有多少块骨头",
    "地球到月球有多远", "一光年有多远", "声音在水里传播得更快吗", "为什么会晕车",
]
TECH = [
    "Python怎么读取csv文件", "Python怎么去重列表", "什么是递归", "解释一下HTTP协议",
    "什么是机器学习", "如何用Python发送邮件", "什么是RESTful API", "数据库索引有什么用",
    "Git怎么撤销上一次提交", "什么是深度学习", "列表和元组有什么区别", "TCP和UDP有什么区别",
    "怎么用Python爬取网页", "什么是正则表达式", "线程和进程的区别", "什么是虚拟环境",
    "怎么看Python版本", "pip怎么换源", "什么是API", "JSON是什么",
]
EDU = [
    "考研难吗", "考公务员需要什么条件", "雅思和托福有什么区别", "四级怎么准备",
    "大学生怎么规划时间", "怎么提高英语口语", "如何做读书笔记", "怎么锻炼演讲能力",
    "怎么找实习", "简历应该怎么写", "GPA重要吗", "要不要考驾照",
    "大学要不要谈恋爱", "怎么处理室友关系", "社恐怎么办", "怎么交到朋友",
]
ASSISTANT = [
    "你用的什么模型", "你会说英语吗", "你的知识更新到什么时候", "你能上网吗",
    "你能帮我写代码吗", "你能画画吗", "你能翻译吗", "你支持语音吗",
    "你会记住我们的对话吗", "你是谁开发的", "你运行在哪里", "你收费吗",
]


def _q(s):
    r = random.random()
    if r < 0.4:
        return s + "？"
    if r < 0.55:
        return s + random.choice(["呀", "啊", "呢", "吗", "吗？", "呀？"])
    return s


def build_campus():
    items = set()
    for t in CAMPUS_TOPICS:
        for f in CAMPUS_FRAMES:
            items.add(_q(f.format(t=t)))
    for s in CAMPUS_HANDWRITTEN:
        items.add(s)
        items.add(_q(s))
    return list(items)


def build_general():
    items = set(CHITCHAT)
    # 数学（整数结果）
    for _ in range(220):
        a, b = random.randint(2, 99), random.randint(2, 99)
        op = random.choice(["加", "乘以", "减"])
        items.add(_q(f"{a}{op}{b}等于多少"))
        items.add(_q(f"{a * b}除以{a}等于几"))
    # 概念解释
    for c in CONCEPTS:
        for f in ["什么是{x}", "{x}是什么意思", "通俗解释一下{x}", "{x}有什么用", "{x}和什么有关"]:
            items.add(_q(f.format(x=c)))
    # 翻译
    for w in TRANSLATE:
        items.add(_q(f"{w}用英语怎么说"))
        items.add(_q(f"“{w}”翻译成英文"))
    # how-to
    for h in HOWTO:
        for f in ["怎么{x}", "如何{x}", "有什么{x}的技巧", "{x}的正确方法是什么"]:
            items.add(_q(f.format(x=h)))
    # 推荐
    for x in RECOMMEND_TYPES:
        for f in RECOMMEND_FRAMES:
            items.add(_q(f.format(x=x)))
    # 各类常识
    for pool in (KNOWLEDGE, TECH, EDU, ASSISTANT):
        for s in pool:
            items.add(s)
            items.add(_q(s))
    return list(items)


def llm_augment(need_campus, need_general):
    """可选：用 DeepSeek 增广更自然的问法。"""
    sys.path.insert(0, _ROOT)
    from core.llm_client import LLMClient
    llm = LLMClient()
    out = {"专业咨询": set(), "通用知识": set()}
    specs = [
        ("专业咨询", need_campus,
         "大学新生向迎新问答助手咨询本校校务的问题（报到、住宿、食堂、选课、军训、校园卡、"
         "快递、图书馆、社团、学费、医保、请假等），口语化、简短、多样"),
        ("通用知识", need_general,
         "与某所具体大学无关的问题：闲聊问候、生活常识、数学计算、百科、编程常识、"
         "学习方法、对AI助手本身的提问，口语化、简短、多样"),
    ]
    for label, need, desc in specs:
        rounds = max(1, need // 40 + 1)
        for i in range(rounds):
            prompt = (f"请生成 40 条{desc}。要求：每条一行，只输出问题本身，不要编号、"
                      f"不要引号、不要多余说明。避免和常见问法雷同，尽量覆盖不同主题。")
            try:
                text = llm.call(prompt=prompt, system_prompt="你是训练数据生成器，只输出内容。",
                                temperature=1.0)
            except Exception as e:
                print(f"  LLM 增广失败({label} 第{i}轮): {e}")
                break
            for line in (text or "").splitlines():
                line = line.strip().lstrip("0123456789.、-　 ").strip().strip('"“”')
                if 2 <= len(line) <= 40:
                    out[label].add(line)
            print(f"  LLM 增广 {label}: 累计 {len(out[label])} 条")
            if len(out[label]) >= need:
                break
    return list(out["专业咨询"]), list(out["通用知识"])


def main():
    args = [a for a in sys.argv[1:]]
    use_llm = "--llm" in args
    nums = [a for a in args if a.isdigit()]
    total = int(nums[0]) if nums else DEFAULT_TOTAL
    per_class = total // 2
    random.seed(SEED)

    campus = build_campus()
    general = build_general()
    print(f"模板候选：专业咨询 {len(campus)}，通用知识 {len(general)}")

    if use_llm:
        ac, ag = llm_augment(max(0, per_class - len(campus)) + 200,
                             max(0, per_class - len(general)) + 200)
        campus = list(set(campus) | set(ac))
        general = list(set(general) | set(ag))
        print(f"增广后：专业咨询 {len(campus)}，通用知识 {len(general)}")

    def take(pool, n):
        random.shuffle(pool)
        if len(pool) >= n:
            return pool[:n]
        return pool + [random.choice(pool) for _ in range(n - len(pool))]

    rows = ([{"query": q, "label": "专业咨询"} for q in take(campus, per_class)]
            + [{"query": q, "label": "通用知识"} for q in take(general, per_class)])
    random.shuffle(rows)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    uniq = len({r["query"] for r in rows})
    print(f"已写入 {OUT}")
    print(f"总数 {len(rows)}（唯一 {uniq}）| 标签分布 {Counter(r['label'] for r in rows)}")
    for r in rows[:6]:
        print("  ", r)


if __name__ == "__main__":
    main()
