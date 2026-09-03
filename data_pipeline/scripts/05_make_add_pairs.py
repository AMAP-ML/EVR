import os

out_dir = "custom_dataset"

removelist = os.listdir(os.path.join(out_dir, 'remove'))
reflist = os.listdir(os.path.join(out_dir, 'ref'))
refdict = {}
for r in reflist:
    refdict[r.split('_')[0]] = r

for remove in removelist:
    sname = remove
    rnum = remove.split('_')[0]
    rname = refdict[rnum]
    with open(os.path.join(out_dir, 'add.txt'), 'a+') as f:
        f.write(rname + ';' + sname + '\n')
