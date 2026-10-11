import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ask import ask

qs = json.loads(Path(__file__).with_name("questions.json").read_text(encoding="utf-8"))
print(f"Chay {len(qs)} cau...")
ok = 0
for i, item in enumerate(qs, 1):
    try:
        ans, srcs = ask(item["q"])
        has_src = len(srcs) > 0 and any(s.metadata.get("paper") for s in srcs)
        refused_bad = "chua de cap" in ans and has_src
        good = has_src and len(ans) > 50
        ok += good
        print(f"[{i}/{len(qs)}] {'OK' if good else 'XEM LAI'}: {item['q'][:60]} -> {len(srcs)} nguon, {len(ans)} ky tu")
    except Exception as e:
        print(f"[{i}/{len(qs)}] LOI: {item['q'][:60]} — {e}")
print(f"\nKet qua: {ok}/{len(qs)} cau co nguon + tra loi day du.")
print("Tip: cai ragas de do faithfulness/context recall chinh xac hon (xem README).")
