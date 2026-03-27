"""Inspect real LoCoMo dataset from HuggingFace."""
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download("KimmoZZZ/locomo", "locomo10.json", repo_type="dataset")
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Samples: {len(data)}")
item = data[0]

# QA categories
cats = {}
for qa in item["qa"]:
    c = qa["category"]
    cats[c] = cats.get(c, 0) + 1
print(f"QA categories in sample 0: {cats}")
print(f"QA count sample 0: {len(item['qa'])}")

# QA examples by category
for cat_id in sorted(cats.keys()):
    qs = [q for q in item["qa"] if q["category"] == cat_id]
    print(f"\nCategory {cat_id} ({len(qs)} qs):")
    for q in qs[:2]:
        print(f"  Q: {q['question'][:100]}")
        ans = q.get('answer', q.get('answers', '???'))
        print(f"  A: {str(ans)[:100]}")
        print(f"  Evidence: {q.get('evidence', '???')}")
        print(f"  Keys: {list(q.keys())}")

# Conversation structure
conv = item["conversation"]
print(f"\nConversation keys: {list(conv.keys())[:12]}")
print(f"Speaker A: {conv['speaker_a']}")
print(f"Speaker B: {conv['speaker_b']}")

# Sessions
sess_keys = sorted([k for k in conv.keys() if k.startswith("session_") and not k.endswith("date_time")])
print(f"Sessions: {len(sess_keys)}")
for sk in sess_keys[:3]:
    turns = conv[sk]
    date_key = sk + "_date_time"
    date = conv.get(date_key, "?")
    print(f"  {sk} ({date}): {len(turns)} turns")
    if turns:
        print(f"    Turn 0: {str(turns[0])[:200]}")

# Total QA across all samples
total_qa = sum(len(s["qa"]) for s in data)
print(f"\nAll {len(data)} samples total QA: {total_qa}")
for s in data:
    print(f"  {s['sample_id']}: {len(s['qa'])} QA")
