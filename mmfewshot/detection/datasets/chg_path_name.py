import os



src_txt_path = r"/data/piaozhengquan/projects/FSOD/mmfewshot-main/data/few_shot_ann/voc/seed_src"
saved_path = r"/data/piaozhengquan/projects/FSOD/mmfewshot-main/data/few_shot_ann/voc/seed"


if not os.path.exists(saved_path):
    os.mkdir(saved_path)

all_sub_file = os.listdir(src_txt_path)

for a_sub_file in all_sub_file:
    all_txt = os.listdir(os.path.join(src_txt_path, a_sub_file))

    os.mkdir(os.path.join(saved_path, a_sub_file))

    for a_txt in all_txt:

        with open(os.path.join(saved_path, a_sub_file, a_txt), 'a+') as ww:
            with open(os.path.join(src_txt_path, a_sub_file, a_txt), "r") as f:
                for line in f.readlines():
                    line = line[9:]
                    ww.write(line)




