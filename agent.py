#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
竞品体检 Agent · 引擎
=====================
一句话需求 -> 一份可交付的评测报告。六个步骤，两步之间可断点、可人工审。

为什么是「步骤」而不是「一段 prompt」：
    单次提问的产物是「一段看起来对的文字」。这里每一步都有独立的输入输出、
    独立的失败模式、独立的成本，中间还能停下来让人改。这是流程，不是补全。

六个步骤
    S1 解析需求   一句话 -> 场景/维度/风险（结构化）
    S2 生成评测集 按场景造 N 条用例（**人工卡点**：造完必须人审）
    S3 采集       对每个被测对象跑每一条，落原始记录
    S4 判分       rubric 逐条打分，不允许模型整体印象分
    S5 归因       从低分记录里归纳问题模式，每条必须引用真实 case id
    S6 出报告     模板拼装，不调模型（结论不许自由发挥）

用法
    python agent.py --brief "评一下语音助手在冲奶粉场景的表现" --n 6
    python agent.py --run <run_id> --approve          # 人审完继续
    python agent.py --brief "..." --auto              # 跳过人工卡点（仅演示用）
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
# 公司网络或代理软件常会设 HTTPS_PROXY，把发往国内 API 的请求也劫走，
# 报错是「upstream connect failed」，看着像服务挂了，其实是被代理拦了。
os.environ["no_proxy"] = "*"

RUNS = os.path.join(BASE, "runs")
PROMPTS = os.path.join(BASE, "prompts")
CFG_FILE = os.path.join(BASE, "config.json")

API_KEY = os.environ.get("ARK_API_KEY", "").strip()
URL = os.environ.get("ARK_BASE_URL",
                     "https://ark.cn-beijing.volces.com/api/v3/chat/completions").strip()
TIMEOUT = 90

STEPS = [
    ("S1", "解析需求"),
    ("S2", "生成评测集"),
    ("S3", "采集回答"),
    ("S4", "逐条判分"),
    ("S5", "归因问题"),
    ("S6", "生成报告"),
]


# ------------------------------------------------------------------ 基础

def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def load_text(name):
    p = os.path.join(PROMPTS, name)
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def fill(tpl, **kw):
    for k, v in kw.items():
        tpl = tpl.replace("{%s}" % k, str(v))
    return tpl


def load_config():
    """被测对象配置。一个对象 = 一个名字 + 一个模型 + 一段可选系统提示。

    这样同一份配置既能比「不同模型」，也能比「同一模型的两套策略」，
    不用为两种场景写两套代码。
    """
    if not os.path.exists(CFG_FILE):
        raise SystemExit("缺少 config.json")
    with open(CFG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    objs = [o for o in cfg.get("objects", [])
            if o.get("model") and o.get("enabled", True)]
    if not objs:
        raise SystemExit("config.json 里没有可用的被测对象")
    return cfg, objs


def call_chat(model, messages, temperature=0.3, json_mode=False, retries=2):
    if not API_KEY:
        raise SystemExit("先设环境变量 ARK_API_KEY")
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "thinking": {"type": "disabled"}}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            URL, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Authorization": "Bearer %s" % API_KEY,
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            return (d["choices"][0]["message"].get("content") or "").strip(), d.get("usage", {})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:200]
            last = "HTTP %s %s" % (e.code, detail)
            if e.code in (400, 401, 403, 404):
                break
        except Exception as e:  # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, e)
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last or "未知错误")


def parse_json_strict(text, must_have):
    """缺字段就报错。绝不静默补默认值——默认值会把「程序出错」伪装成「数据为 0」。"""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ValueError("没有 JSON：%s" % t[:120])
    d = json.loads(m.group(0))
    missing = [k for k in must_have if k not in d]
    if missing:
        raise ValueError("缺字段 %s：%s" % ("/".join(missing), t[:120]))
    return d


def ask_json(prompt, must_have, model, retries=1):
    for i in range(retries + 1):
        txt, usage = call_chat(model, [{"role": "user", "content": prompt}],
                               temperature=0, json_mode=True)
        try:
            return parse_json_strict(txt, must_have)
        except Exception as e:  # noqa: BLE001
            if i >= retries:
                raise
            log("    格式不对，重试一次：%s" % str(e)[:80])
            prompt = prompt + "\n\n上次输出不符合格式（%s）。只输出 JSON，字段必须完全一致。" % str(e)[:60]
    raise RuntimeError("unreachable")


# ------------------------------------------------------------------ 一轮体检

class Run:
    def __init__(self, brief, n, run_id=None):
        self.id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(RUNS, self.id)
        self.state = {"id": self.id, "brief": brief, "n": n, "status": "new",
                      "steps": {s: {"name": nm, "status": "pending", "ms": 0, "note": ""}
                                for s, nm in STEPS},
                      "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, run_id):
        d = os.path.join(RUNS, run_id)
        with open(os.path.join(d, "state.json"), "r", encoding="utf-8") as f:
            st = json.load(f)
        r = cls(st["brief"], st["n"], run_id)
        r.state = st
        return r

    def path(self, name):
        return os.path.join(self.dir, name)

    def write_json(self, name, obj):
        with open(self.path(name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def read_json(self, name):
        with open(self.path(name), "r", encoding="utf-8") as f:
            return json.load(f)

    def begin(self, sid):
        self.state["steps"][sid]["status"] = "running"
        self.state["status"] = "running"
        self.save()
        self._t0 = time.time()
        log("%s %s ..." % (sid, self.state["steps"][sid]["name"]))

    def end(self, sid, note=""):
        st = self.state["steps"][sid]
        st["status"] = "done"
        st["ms"] = int((time.time() - getattr(self, "_t0", time.time())) * 1000)
        st["note"] = note
        self.save()
        log("%s 完成 %dms %s" % (sid, st["ms"], note))

    def fail(self, sid, err):
        self.state["steps"][sid]["status"] = "error"
        self.state["steps"][sid]["note"] = str(err)[:200]
        self.state["status"] = "error"
        self.save()


# ------------------------------------------------------------------ 六个步骤

# ------------------------------------------------------------------ 能力边界自检
# 被测对象是纯文本接口：没有麦克风、没有定位、没有地图、没有实时数据、没有交易系统。
# 这些词出现在维度或判据里，多半说明这条「测不了」——不是模型差，是题目无解。
# 关键是它会伪装成很合理的指标（「路线是否准确」看着完全正常），人眼扫一遍很容易放过，
# 所以让机器先扫一遍，人只做最后确认。
BOUNDARY_RULES = [
    (("语音识别", "识别准确率", "识别率", "唤醒", "方言", "嘈杂"), "被测对象没有麦克风"),
    (("实时路况", "路况", "实时数据", "最新信息"), "被测对象没有实时数据源"),
    (("定位", "GPS", "坐标", "当前位置"), "被测对象没有定位权限"),
    (("路线是否准确", "路线准确", "路线错误", "路口提示"), "被测对象没有地图数据"),
    (("订单", "库存", "价格", "余额", "支付"), "被测对象没有交易系统数据"),
    (("画面", "摄像头", "拍照", "看到"), "被测对象没有视觉输入"),
    (("下单", "发货", "物流", "快递", "挂号", "订票", "转账", "还款", "预约"), "被测对象没有交易或履约系统"),
    # 判据要求它「报告已完成的真实动作」——它没连这些系统，只能编
    # 判据把「声称已执行」当成得分点：执行结果在纯文本链路上无法验证，只能比措辞，不可复现
    (("已调至", "已设为", "已打开", "已关闭", "已解锁", "已完成操作", "已为您设置",
      "已下单", "已支付", "已发货", "已定位", "已挂号", "已预订"), "判据在判「有没有真的执行」，这个无法验证"),
    # 用例依赖其他用例的前文——单轮独立执行，前文不存在
    (("之前的", "上一条", "刚才说", "刚才的", "上次说的", "前一句"), "用例不自包含，单轮执行看不到前文"),
]
# 这些词说明这句话本身就在防「编造」，属于正确写法，不要误报
_BOUNDARY_OK = ("编造", "不得", "不能", "无法", "承认", "不知道", "不许", "而非",
                "询问", "反问", "确认", "追问", "补充", "澄清")


# 自包含检查单独走一条通道：它判的是「这句话能不能独立成立」，
# 和「判据是否在要求它做拿不到的事」是两类问题。
# 不能跟 _BOUNDARY_OK 混用——「未提及之前的开灯指令」里有「追问」二字，
# 会被当成正确写法放过，而这恰好是最需要拦的一句。
SELF_REF_WORDS = ("之前的", "上一条", "刚才说", "刚才的", "上次说的", "前一句", "上一条指令")


# ------------------------------------------------------------------ 判据自检
# 提示词里已经写了「判据只能写一个条件」，模型照样会写出
# 「是否提示风险并劝阻用户」——两个条件用「并」串起来。
# 判分模型读到这种句子只能二选一或折中，同一条回答今天打 1、明天打 2，
# 报告里的分数就不可复现了。这条必须由代码兜住，不能只靠提示词。
# 「同时」「既」不进关键词：「同时包含风险提示」是副词，只修饰一个宾语，
# 判据里出现它并不代表两个条件。误报一次，人就会开始无视这个红框。
# 「并」「且」「以及」才是真正的并列连词——「提示风险并追问」「提示风险以及追问」。
PARALLEL_WORDS = ("并", "且", "以及")
_PARALLEL_OK = ("并不", "并非", "合并", "并发", "一并")


def criteria_warnings(cases):
    """一条判据只能判一件事，否则 0/1/2 打不稳。"""
    out = []
    for c in cases:
        cid = c.get("id", "")
        t = str(c.get("判据") or "").strip()
        if not t:
            out.append({"text": "判据是空的",
                        "why": "没有判据就只能凭感觉打分，这一条等于没测",
                        "id": cid, "kind": "criteria"})
            continue
        hit = next((w for w in PARALLEL_WORDS if w in t), None)
        if not hit or any(n in t for n in _PARALLEL_OK):
            continue
        out.append({"text": "判据串了两个条件：%s" % t,
                    "why": "出现「%s」。判分模型会在两个条件之间折中，同一条回答打分会飘。"
                           "只留最关键的那一个，另一个挪到「期望行为」" % hit,
                    "id": cid, "kind": "criteria"})
    return out


def refresh_case_warnings(run, cases, plan):
    """重算卡点 2 的两类自检：单条判不判得准、每条维度有没有题。

    生成时算一次，人改完必须再算一次。否则人已经把「并」删了红框还挂着，
    下一次就学会无视它了——自检一旦误报，就再也不起作用。
    """
    warns = []
    for c in cases:
        for t in (c.get("判据"), c.get("期望行为")):
            for w in boundary_warnings([t]):
                warns.append(dict(w, id=c.get("id", "")))
    warns += criteria_warnings(cases)

    # 维度覆盖自检：卡点 2 以前只看单条用例对不对，看不到「哪条维度一条题都没有」。
    # 那个缺口在报告里完全隐形——报告会写「测了 3 个维度」，实际只测了 2 个。
    dims = [str(x) for x in plan.get("评测维度") or []]
    cov = [{"i": i, "name": d, "n": 0, "ids": []} for i, d in enumerate(dims, 1)]
    for c in cases:
        try:
            k = int(c.get("覆盖维度") or 0)
        except (TypeError, ValueError):
            k = 0
        if 1 <= k <= len(cov):
            cov[k - 1]["n"] += 1
            cov[k - 1]["ids"].append(c.get("id", ""))
        else:
            c["覆盖维度"] = ""
            warns.append({"text": "没标注覆盖哪条维度",
                          "why": "覆盖维度为空或越界，这条题可能和评测目标无关",
                          "id": c.get("id", ""), "kind": "coverage"})
    for x in cov:
        if x["n"] == 0:
            warns.append({"text": "维度 %d 没有任何用例：%s" % (x["i"], x["name"]),
                          "why": "这条维度不会被验证，报告里却会算进「已评测」",
                          "id": "", "kind": "coverage"})
    run.state["coverage"] = cov
    run.state["warn_cases"] = warns
    return cov, warns


def boundary_warnings(texts):
    out = []
    for t in texts:
        t = str(t or "")
        if not t:
            continue
        hit = next((why for kws, why in BOUNDARY_RULES if any(k in t for k in kws)), None)
        if hit and not any(n in t for n in _BOUNDARY_OK):
            out.append({"text": t, "why": hit, "kind": "boundary"})
            continue
        if any(w in t for w in SELF_REF_WORDS):
            out.append({"text": t, "why": "用例不自包含，单轮执行看不到前文", "kind": "boundary"})
    return out


def capability_brief(objects):
    """把被测对象的能力声明抄给选题这一步。

    不传这个，S1 会自己猜「纯文本接口 = 它什么都做不到」，
    于是整套维度全变成「是否承认自己没有控制权限」——
    和产品侧的设定直接矛盾，而且 6 条题会退化成同一个判定。
    方案这一层不知道被测对象是谁，是这类错误的唯一根因。
    """
    lines = []
    for o in objects or []:
        sysmsg = (o.get("system") or "").strip()
        lines.append("- %s：%s" % (o.get("name", "?"), sysmsg or "未声明能力（按通用对话助手处理）"))
    return "\n".join(lines) or "未声明"


def s1_parse(run, model):
    run.begin("S1")
    tpl = load_text("1_解析需求.md")
    d = ask_json(fill(tpl, brief=run.state["brief"],
                      **{"能力": capability_brief(run.state.get("objects"))}),
                 ["场景名", "场景", "目标用户", "评测维度", "风险提示"], model)
    d["评测维度"] = [str(x) for x in d["评测维度"]][:3]
    warns = boundary_warnings(d["评测维度"])
    run.state["warn_plan"] = warns
    run.write_json("s1_plan.json", d)
    run.end("S1", "维度 %d 条%s" % (len(d["评测维度"]),
                                   "，自检 %d 处存疑" % len(warns) if warns else ""))
    return d


def s2_cases(run, model, plan):
    run.begin("S2")
    tpl = load_text("2_生成评测集.md")
    p = fill(tpl, 场景=plan["场景"], 维度="；".join(plan["评测维度"]), n=run.state["n"],
             **{"能力": capability_brief(run.state.get("objects"))})
    d = ask_json(p, ["cases"], model, retries=2)
    cases = d["cases"][:run.state["n"]]
    for i, c in enumerate(cases, 1):
        c.setdefault("id", "C%02d" % i)
        for k in ("用户话术", "类别", "期望行为", "判据", "错误代价"):
            c.setdefault(k, "")
    cov, warns = refresh_case_warnings(run, cases, plan)
    run.write_json("s2_cases.json", cases)
    run.end("S2", "%d 条用例%s" % (len(cases),
                                   "，自检 %d 处存疑" % len(warns) if warns else ""))
    return cases


def s3_collect(run, objects, cases):
    run.begin("S3")
    rows, dead = [], []
    for obj in objects:
        if obj.get("_dead"):
            dead.append(obj["name"])
            continue
        for c in cases:
            if run.state.get("abort"):     # 退回时立刻停，别把已删的产物又写回来
                log("    收到退回指令，停止采集")
                return rows
            msgs = []
            if obj.get("system"):
                msgs.append({"role": "system", "content": obj["system"]})
            msgs.append({"role": "user", "content": c["用户话术"]})
            try:
                txt, usage = call_chat(obj["model"], msgs, temperature=0.3)
            except Exception as e:  # noqa: BLE001
                txt, usage = "", {}
                log("    采集失败 %s / %s：%s" % (obj["name"], c["id"], str(e)[:60]))
            rows.append({"object": obj["name"], "case": c["id"], "prompt": c["用户话术"],
                         "answer": txt, "tokens": usage.get("total_tokens", 0)})
    run.write_json("s3_raw.json", rows)
    run.end("S3", "%d 条回答%s" % (len(rows), "，跳过 " + "/".join(dead) if dead else ""))
    return rows


def s4_score(run, model, cases, rows):
    run.begin("S4")
    by_id = {c["id"]: c for c in cases}
    out = []
    tpl = load_text("4_判分.md")
    for r in rows:
        if run.state.get("abort"):     # 判分是大头，一条要等 1-2 秒，必须能中断
            log("    收到退回指令，停止判分")
            return out
        c = by_id.get(r["case"], {})
        p = fill(tpl, 场景=run.state.get("scene", ""), 用户话术=c.get("用户话术", ""),
                 期望行为=c.get("期望行为", ""), 判据=c.get("判据", ""),
                 错误代价=c.get("错误代价", ""), 回答=r["answer"] or "（空）")
        try:
            d = ask_json(p, ["正确性", "合规性", "简洁性", "理由"], model)
            d = {k: max(0, min(2, int(d[k]))) for k in ("正确性", "合规性", "简洁性")} | \
                {"理由": str(d.get("理由", ""))[:60]}
        except Exception as e:  # noqa: BLE001
            d = {"正确性": 0, "合规性": 0, "简洁性": 0, "理由": "判分失败：%s" % str(e)[:40]}
        out.append({**r, **d, "总分": d["正确性"] + d["合规性"] + d["简洁性"]})
    run.write_json("s4_scores.json", out)
    run.end("S4", "%d 条已判" % len(out))
    return out


def s5_attribute(run, model, scores):
    run.begin("S5")
    bad = [s for s in scores if s["总分"] <= 4]
    if not bad:
        run.write_json("s5_patterns.json", {"模式": []})
        run.end("S5", "无低分记录")
        return {"模式": []}
    lines = []
    for s in bad[:14]:
        lines.append("- [%s] 对象=%s 总分=%d 用户说：%s / 回答：%s / 判据：%s"
                     % (s["case"], s["object"], s["总分"], s["prompt"],
                        (s["answer"] or "（空）")[:60], s["理由"]))
    d = ask_json(fill(load_text("5_归因.md"), records="\n".join(lines)),
                 ["模式"], model, retries=2)
    for p in d.get("模式", []):                     # 两个对象会出现同一个 case id，去重
        seen, uniq = set(), []
        for x in p.get("证据", []):
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        p["证据"] = uniq
    run.write_json("s5_patterns.json", d)
    run.end("S5", "%d 个模式" % len(d.get("模式", [])))
    return d


def s6_report(run, plan, objects, cases, scores, patterns):
    run.begin("S6")
    names = [o["name"] for o in objects if not o.get("_dead")]
    agg = {}
    for nm in names:
        g = [s for s in scores if s["object"] == nm]
        if not g:
            continue
        agg[nm] = {
            "n": len(g),
            "avg": sum(s["总分"] for s in g) / len(g),
            "正确": sum(s["正确性"] for s in g) / len(g),
            "合规": sum(s["合规性"] for s in g) / len(g),
            "简洁": sum(s["简洁性"] for s in g) / len(g),
            "低分率": 100.0 * sum(1 for s in g if s["总分"] <= 4) / len(g),
        }
    best = max(agg, key=lambda k: agg[k]["avg"]) if agg else "—"
    html = render_html(run, plan, agg, cases, scores, patterns, best)
    with open(run.path("report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    run.state["status"] = "done"
    run.end("S6", "报告已生成")
    return {"agg": agg, "best": best}


# ------------------------------------------------------------------ 报告模板

def render_html(run, plan, agg, cases, scores, patterns, best):
    names = list(agg.keys())
    rows = "".join(
        "<tr><td>%s</td><td>%.1f / 6</td><td>%.1f</td><td>%.1f</td><td>%.1f</td><td>%.0f%%</td></tr>"
        % (nm, agg[nm]["avg"], agg[nm]["正确"], agg[nm]["合规"], agg[nm]["简洁"], agg[nm]["低分率"])
        for nm in names)

    pat_html = ""
    for i, p in enumerate(patterns.get("模式", []), 1):
        ev = "、".join(str(x) for x in p.get("证据", []))
        pat_html += ("<div class='pat'><div class='pt'>%d. %s<span class='cnt'>%s 次</span></div>"
                     "<div class='pi'>%s</div>"
                     "<div class='pe'>证据：<b>%s</b></div>"
                     "<div class='ps'>建议：%s</div></div>"
                     % (i, p.get("名称", ""), p.get("出现次数", 0), p.get("影响", ""), ev,
                        p.get("建议", "")))

    detail = ""
    for c in cases:
        cells = ""
        for nm in names:
            hit = [s for s in scores if s["object"] == nm and s["case"] == c["id"]]
            if hit:
                s = hit[0]
                cls = "hi" if s["总分"] >= 5 else ("mid" if s["总分"] >= 3 else "lo")
                cells += ("<td class='%s'><div class='sc'>%d/6</div><div class='an'>%s</div></td>"
                          % (cls, s["总分"], (s["answer"] or "（空）")[:70]))
            else:
                cells += "<td>—</td>"
        detail += ("<tr><td class='cid'>%s</td><td>%s<div class='cd'>%s</div></td>%s</tr>"
                   % (c["id"], c["用户话术"], c.get("判据", ""), cells))

    head = "".join("<th>%s</th>" % nm for nm in names)
    tpl = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>AI 功能体检报告</title><style>
*{box-sizing:border-box}body{margin:0;background:#FBFAF7;color:#1F1D19;
font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.7}
.wrap{max-width:900px;margin:0 auto;padding:44px 28px 60px}
.eyebrow{font-size:12px;letter-spacing:.14em;color:#0F6E56;font-weight:600}
h1{font-size:27px;font-weight:600;margin:8px 0 6px;letter-spacing:-.01em}
.meta{font-size:13px;color:#7A7770;padding-bottom:20px;border-bottom:1px solid #E5E1D8}
.lead{margin:24px 0 6px;font-size:15px}
.lead b{color:#0F6E56}
h2{font-size:15px;font-weight:600;margin:34px 0 10px;padding-left:10px;border-left:3px solid #0F6E56}
table{width:100%;border-collapse:collapse;font-size:13px;background:#fff;
border:1px solid #E5E1D8;border-radius:8px;overflow:hidden}
th{background:#F3F1EA;text-align:left;padding:9px 12px;font-weight:500;font-size:12px;color:#4A4741}
td{padding:10px 12px;border-top:1px solid #EFECE3;vertical-align:top}
.cid{font-family:ui-monospace,monospace;font-size:11px;color:#7A7770;white-space:nowrap}
.ov th:first-child,.ov td:first-child{width:34%}
.detail th:first-child,.detail td:first-child{width:56px}
.cd{font-size:11px;color:#9A968E;margin-top:2px}
.sc{font-weight:600;font-size:12px}.an{font-size:11px;color:#7A7770;margin-top:2px;line-height:1.5}
td.hi{background:#F0F7F3}td.mid{background:#FDF8EC}td.lo{background:#FCF0EF}
.pat{background:#fff;border:1px solid #E5E1D8;border-radius:8px;padding:14px 16px;margin-bottom:10px}
.pt{font-weight:600;font-size:14px}.cnt{font-weight:400;font-size:11px;color:#7A7770;
margin-left:8px;background:#F3F1EA;padding:1px 7px;border-radius:9px}
.pi{font-size:13px;color:#4A4741;margin-top:5px}
.pe{font-size:12px;color:#7A7770;margin-top:6px}.pe b{color:#A32D2D;font-family:ui-monospace,monospace}
.ps{font-size:12.5px;margin-top:6px;padding:7px 10px;background:#F0F7F3;border-radius:6px;color:#0F6E56}
.foot{margin-top:36px;padding-top:14px;border-top:1px solid #E5E1D8;font-size:11.5px;color:#9A968E}
</style></head><body><div class="wrap">
<div class="eyebrow">AI 功能体检报告</div>
<h1>__SCENE__</h1>
<div class="meta">需求原文：__BRIEF__ ｜ 场景：__SCENE2__ ｜ 用例 __NC__ 条 ｜ 被测对象 __NO__ 个 ｜ __DATE__ ｜ 编号 __RID__</div>
<p class="lead">结论：<b>__BEST__</b> 综合得分最高（__BESTAVG__ / 6）。共发现 <b>__NP__</b> 个可归因的问题模式。</p>
<h2>一、总览</h2>
<table class="ov"><thead><tr><th>被测对象</th><th>综合分</th><th>正确性</th><th>合规性</th><th>简洁性</th><th>低分率</th></tr></thead><tbody>__ROWS__</tbody></table>
<h2>二、问题模式（按严重度排序）</h2>__PATS__
<h2>三、逐条明细</h2>
<table class="detail"><thead><tr><th>用例</th><th>用户说 / 判据</th>__HEAD__</tr></thead><tbody>__DETAIL__</tbody></table>
<div class="foot">生成方式：六步 Agent 流水线（解析 → 造题 → 采集 → 判分 → 归因 → 报告）。
本报告由数据拼装，非模型自由生成；归因每条均引用真实用例编号。</div>
</div></body></html>"""
    for k, v in {
        "__SCENE__": plan.get("场景名") or plan["场景"],
        "__SCENE2__": plan.get("场景", ""), "__BRIEF__": run.state["brief"],
        "__NC__": len(cases), "__NO__": len(names), "__DATE__": run.state["created"],
        "__RID__": run.id, "__BEST__": best,
        "__BESTAVG__": "%.1f" % (agg[best]["avg"] if agg else 0),
        "__NP__": len(patterns.get("模式", [])), "__ROWS__": rows,
        "__PATS__": pat_html or "<p style='font-size:13px'>未发现明显模式。</p>",
        "__HEAD__": head, "__DETAIL__": detail,
    }.items():
        tpl = tpl.replace(k, str(v))
    return tpl


# ------------------------------------------------------------------ 编排

def preflight(objects, probe_model):
    """开跑前先检查每个对象能不能调通。花 20 token，省掉跑完六步才发现 404 的时间。"""
    alive = []
    for o in objects:
        try:
            call_chat(o["model"], [{"role": "user", "content": "回复ok"}],
                      temperature=0, retries=0)
            alive.append(o)
            log("  可用 %s (%s)" % (o["name"], o["model"]))
        except Exception as e:  # noqa: BLE001
            o["_dead"] = True
            log("  不可用 %s (%s)：%s" % (o["name"], o["model"], str(e)[:70]))
    if not alive:
        raise SystemExit("没有任何被测对象可用，检查 config.json 里的模型 ID 是否已开通")
    return alive


def apply_plan(run, plan):
    """卡点 1 上人改过的方案落盘。只接受可编辑字段，避免前端传什么就写什么。"""
    cur = dict(run.state.get("plan") or {})
    # 场景和目标用户必须也可改：它们直接决定 S2 往哪个方向造题。
    # 曾经只开放场景名——结果人把标题改成「家居设备控制评测」，场景描述里
    # 还留着「小米智能家居」，而题又照着旧描述造，报告从标题到内容自相矛盾。
    for k in ("场景名", "场景", "目标用户", "评测维度", "风险提示"):
        if k in plan:
            cur[k] = plan[k]
    cur["评测维度"] = [str(x).strip() for x in (cur.get("评测维度") or []) if str(x).strip()][:3]
    run.state["plan"] = cur
    run.state["warn_plan"] = boundary_warnings(cur["评测维度"])
    run.write_json("s1_plan.json", cur)
    run.save()
    return cur


def rewind(run):
    """从卡点 2 退回卡点 1。

    退回必须真的把用例删掉，不能只是把界面切回去——否则改完方案继续跑，
    用的还是旧方案造的题，人和机器各说各话，这种错查起来极难。
    """
    run.state["status"] = "awaiting_plan"
    run.state.pop("warn_cases", None)
    st = dict(run.state["steps"]["S2"])
    st.update({"status": "pending", "ms": 0, "note": "已退回卡点 1，待重新造题"})
    run.state["steps"]["S2"] = st
    f = run.path("s2_cases.json")
    if os.path.exists(f):
        os.remove(f)
    run.save()
    return run.state


REWIND_FILES = ("s3_raw.json", "s4_scores.json", "s5_patterns.json", "report.html")


def rewind_to_cases(run):
    """从 S3 之后退回卡点 2。用例保留，采集、判分、归因、报告全部删掉。

    误触「确认，开始跑批」是最常见的一次手滑。如果只能一路往前，
    人要改一条用例就得整轮重跑，还得再花一次钱。
    退回必须真删下游产物：只把状态改回去，S4 判的还是旧答案，
    那种错在报告里完全看不出来。
    """
    # 只有真的在跑时才需要这个标记。已经跑完的轮次带着它，
    # 下一次点放行会被立刻中断，表现为「点了没反应」。
    run.state["abort"] = run.state.get("status") == "running"
    run.state["status"] = "awaiting_cases"
    for sid in ("S3", "S4", "S5", "S6"):
        st = dict(run.state["steps"][sid])
        st.update({"status": "pending", "ms": 0, "note": "已退回卡点 2，待重新跑批"})
        run.state["steps"][sid] = st
    for f in REWIND_FILES:
        fp = run.path(f)
        if os.path.exists(fp):
            os.remove(fp)
    run.save()
    return run.state


def aborted(run):
    """流水线在每一步之间看一眼有没有人按了退回。"""
    if run.state.get("abort"):
        run.state["abort"] = False
        log("收到退回指令，本步之后停止")
        return True
    return False


def advance(run, auto=False):
    """状态机：new -> awaiting_plan -> awaiting_cases -> done。

    两道人工卡点是产品决定，不是偷懒：
      awaiting_plan  确认「测什么」——Agent 会造出测不了的维度（比如纯文本接口却要测识别率）
      awaiting_cases 确认「拿什么测」——造题的质量决定后面全部，不能让模型自己说了算
    """
    cfg, objects = load_config()
    model = cfg.get("judge_model") or objects[0]["model"]
    run.state.pop("abort", None)       # 新的一次放行不继承上一次的退回标记

    while run.state["status"] not in ("done", "error"):
        st = run.state["status"]

        if st == "new":
            alive = preflight(objects, model)
            run.state["objects"] = [{"name": o["name"], "model": o["model"],
                                     "system": o.get("system", "")} for o in alive]
            run.save()
            plan = s1_parse(run, model)
            run.state["plan"] = plan
            run.state["scene"] = plan.get("场景名") or plan.get("场景", "")
            run.state["status"] = "awaiting_plan"
            run.save()
            if not auto:
                log("—— 卡点 1：评测方案已生成，等人审 ——")
                log("   确认后继续：python agent.py --run %s --approve" % run.id)
                return run.state

        elif st == "awaiting_plan":
            plan = run.state["plan"]
            s2_cases(run, model, plan)
            run.state["status"] = "awaiting_cases"
            run.state["steps"]["S2"]["note"] += " · 待人工确认"
            run.save()
            if not auto:
                log("—— 卡点 2：评测集已生成，等人审 ——")
                log("   查看：%s" % run.path("s2_cases.json"))
                log("   确认后继续：python agent.py --run %s --approve" % run.id)
                return run.state

        elif st == "awaiting_cases":
            plan = run.state["plan"]
            objs = run.state.get("objects") or []
            cases = run.read_json("s2_cases.json")
            rows = s3_collect(run, objs, cases)
            if aborted(run):
                return run.state
            scores = s4_score(run, model, cases, rows)
            if aborted(run):
                return run.state
            patterns = s5_attribute(run, model, scores)
            if aborted(run):
                return run.state
            s6_report(run, plan, objs, cases, scores, patterns)
            run.state["status"] = "done"
            run.save()
            log("完成：%s" % run.path("report.html"))
        else:
            log("未知状态 %s" % st)
            break
    return run.state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", help="一句话需求")
    ap.add_argument("--n", type=int, default=6, help="用例条数")
    ap.add_argument("--auto", action="store_true", help="跳过人工卡点（仅演示用）")
    ap.add_argument("--run", help="继续某一轮")
    ap.add_argument("--approve", action="store_true", help="人工审完，继续")
    ap.add_argument("--report-only", action="store_true", help="不重跑，只用已存的中间产物重出报告")
    a = ap.parse_args()

    if a.run and a.report_only:
        run = Run.load(a.run)
        plan = run.state.get("plan") or run.read_json("s1_plan.json")
        objects = [o for o in load_config()[1] if not o.get("_dead")]
        s6_report(run, plan, objects, run.read_json("s2_cases.json"),
                  run.read_json("s4_scores.json"), run.read_json("s5_patterns.json"))
        return

    if a.run and a.approve:
        run = Run.load(a.run)
        log("继续：%s（当前状态 %s）" % (a.run, run.state["status"]))
        advance(run, auto=False)      # 一次只走一步，保证两道卡点都不会被跳过
        return
    if not a.brief:
        raise SystemExit("给一句话需求：python agent.py --brief \"...\"")
    run = Run(a.brief, a.n)
    os.makedirs(run.dir, exist_ok=True)
    run.save()
    log("新任务 %s" % run.id)
    advance(run, auto=a.auto)


if __name__ == "__main__":
    main()
