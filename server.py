#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
竞品体检 Agent · 本地服务
========================
只做四件事：起任务、查状态、过卡点、看报告。业务逻辑全在 agent.py，这里不重复实现。

启动：python server.py      然后打开 http://127.0.0.1:8765
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 没设环境变量时，从用户注册表里取（Windows 上 setx 就是写这里）
if not os.environ.get("ARK_API_KEY"):
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
        for n in ("ARK_API_KEY", "ARK_MODEL"):
            try:
                os.environ[n] = winreg.QueryValueEx(k, n)[0]
            except FileNotFoundError:
                pass
    except Exception:
        pass

import agent  # noqa: E402

PORT = int(os.environ.get("PORT", "8765"))
WEB = os.path.join(BASE, "web")
LOCK = threading.Lock()


def kick(run, auto):
    def _run():
        with LOCK:
            try:
                agent.advance(run, auto=auto)
            except Exception as e:  # noqa: BLE001
                run.state["status"] = "error"
                run.state["error"] = str(e)[:300]
                run.save()
                print("任务失败：", e, flush=True)
    threading.Thread(target=_run, daemon=True).start()


def state_payload(run_id):
    run = agent.Run.load(run_id)
    d = dict(run.state)
    d["cases"] = None
    d["agg"] = None
    try:
        if os.path.exists(run.path("s2_cases.json")):
            d["cases"] = run.read_json("s2_cases.json")
        if os.path.exists(run.path("s3_raw.json")):
            d["rows"] = run.read_json("s3_raw.json")
        if os.path.exists(run.path("s4_scores.json")):
            sc = run.read_json("s4_scores.json")
            agg = {}
            for s in sc:
                a = agg.setdefault(s["object"], {"n": 0, "sum": 0})
                a["n"] += 1
                a["sum"] += s["总分"]
            d["agg"] = [{"name": k, "avg": round(v["sum"] / v["n"], 1)} for k, v in agg.items()]
            d["scores"] = [{"case": x["case"], "object": x["object"], "总分": x["总分"],
                            "理由": x.get("理由", ""), "answer": x.get("answer", "")}
                           for x in sc]
        if os.path.exists(run.path("s5_patterns.json")):
            d["patterns"] = run.read_json("s5_patterns.json")
    except Exception:
        pass
    d["has_report"] = os.path.exists(run.path("report.html"))
    return d


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(WEB, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            rid = (q.get("run") or [""])[0]
            if not rid:
                return self._json({"error": "缺少 run"}, 400)
            try:
                return self._json(state_payload(rid))
            except FileNotFoundError:
                return self._json({"error": "找不到这一轮"}, 404)
        if u.path == "/report":
            rid = (q.get("run") or [""])[0]
            p = os.path.join(agent.RUNS, rid, "report.html")
            if not os.path.exists(p):
                return self._send(404, "报告还没生成", "text/plain; charset=utf-8")
            with open(p, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if u.path == "/api/start":
            brief = (body.get("brief") or "").strip()
            if not brief:
                return self._json({"error": "先写一句需求"}, 400)
            run = agent.Run(brief, int(body.get("n") or 6))
            os.makedirs(run.dir, exist_ok=True)
            run.save()
            kick(run, auto=bool(body.get("auto")))
            return self._json({"run": run.id})
        if u.path == "/api/approve":
            rid = body.get("run") or ""
            try:
                run = agent.Run.load(rid)
            except FileNotFoundError:
                return self._json({"error": "找不到这一轮"}, 404)
            if run.state.get("status") not in ("awaiting_plan", "awaiting_cases"):
                return self._json({"error": "当前不在卡点状态：%s" % run.state.get("status")}, 400)
            kick(run, auto=False)          # 一次只放行一步，否则卡点 2 会被整段跳过
            return self._json({"ok": True})
        if u.path == "/api/edit_cases":
            rid = body.get("run") or ""
            try:
                run = agent.Run.load(rid)
            except FileNotFoundError:
                return self._json({"error": "找不到这一轮"}, 404)
            cases = body.get("cases")
            if not isinstance(cases, list) or not cases:
                return self._json({"error": "用例不能为空"}, 400)
            # 界面只回传可编辑的几个字段，其余（覆盖维度、错误代价）按 id 合并回来，
            # 否则人一改文字，「这条题测的是第几维度」就丢了，覆盖统计会凭空归零。
            old = {c.get("id"): c for c in run.read_json("s2_cases.json") or []}
            for i, c in enumerate(cases, 1):
                c.setdefault("id", "C%02d" % i)
                base = old.get(c["id"]) or {}
                for k, v in base.items():
                    c.setdefault(k, v)
            run.write_json("s2_cases.json", cases)
            plan = run.state.get("plan") or {}
            # 改完必须重算自检，不然人已经把「并」删了红框还挂着，
            # 下一次他就会学会无视这个框——自检一旦误报就再也不起作用。
            cov, warns = agent.refresh_case_warnings(run, cases, plan)
            run.save()
            return self._json({"ok": True, "n": len(cases), "coverage": cov, "warn_cases": warns})
        if u.path == "/api/edit_plan":
            rid = body.get("run") or ""
            try:
                run = agent.Run.load(rid)
            except FileNotFoundError:
                return self._json({"error": "找不到这一轮"}, 404)
            if run.state.get("status") != "awaiting_plan":
                return self._json({"error": "不在卡点 1，改不了方案"}, 400)
            plan = agent.apply_plan(run, body.get("plan") or {})
            return self._json({"ok": True, "plan": plan,
                               "warn_plan": run.state.get("warn_plan") or []})
        if u.path == "/api/back_cases":
            rid = body.get("run") or ""
            try:
                run = agent.Run.load(rid)
            except FileNotFoundError:
                return self._json({"error": "找不到这一轮"}, 404)
            st = run.state.get("status")
            if st not in ("awaiting_cases", "running", "done", "error"):
                return self._json({"error": "还没到卡点 2，没有可回退的用例"}, 400)
            agent.rewind_to_cases(run)
            return self._json({"ok": True, "status": run.state["status"]})
        if u.path == "/api/back":
            rid = body.get("run") or ""
            try:
                run = agent.Run.load(rid)
            except FileNotFoundError:
                return self._json({"error": "找不到这一轮"}, 404)
            if run.state.get("status") != "awaiting_cases":
                return self._json({"error": "只有卡点 2 能退回卡点 1"}, 400)
            agent.rewind(run)
            return self._json({"ok": True, "status": run.state["status"]})
        self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    print("竞品体检 Agent 已启动：http://127.0.0.1:%d" % PORT)
    print("Ctrl+C 停止")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
