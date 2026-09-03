import json
import os
import random

out_dir = "custom_dataset"

refclass = {}

with open(os.path.join(out_dir, "manifest.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["category"] not in refclass:
            refclass[item["category"]] = [item]
        else:
            refclass[item["category"]].append(item)

for key, items in refclass.items():
    if len(items) <= 1:
        continue
    for item in items:
        rname = item["ref"]
        rnum = item["id"]
        snum = rnum
        while snum == rnum:
            other = items[random.randint(0, len(items) - 1)]
            sname = other["scene"]
            snum = other["id"]
        with open(os.path.join(out_dir, 'replace.txt'), 'a+') as f:
            f.write(rname + ';' + sname + '\n')
