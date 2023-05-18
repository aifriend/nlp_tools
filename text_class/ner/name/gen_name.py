import random

from common.ClassFile import ClassFile

SOURCE = ".txt"
FILE_DUMP = ".txt"


def dump(doc):
    ClassFile.to_txtfile(
        data=doc + "\n",
        file_=FILE_DUMP,
        mode="a+")


if __name__ == '__main__':
    output = list()
    name_dic = ClassFile.file_to_list(".txt", binary=False, encoding="utf-8")
    name_list = list(map(lambda x: x.pop(), name_dic))
    del name_dic
    total_name = len(name_list)
    random.shuffle(name_list)

    for _ in range(0, 13622):
        output_name = list()
        for i in range(0, 2):
            r_id = random.randint(0, total_name)
            name = name_list[r_id]
            output_name.append(name)
        output.append(" ".join(output_name))
        output_name = list()
        for i in range(0, 3):
            r_id = random.randint(0, total_name)
            name = name_list[r_id]
            output_name.append(name)
        output.append(" ".join(output_name))
        output_name = list()
        for i in range(0, 4):
            r_id = random.randint(0, total_name)
            name = name_list[r_id]
            output_name.append(name)
        output.append(" ".join(output_name))
        output_name = list()
        for i in range(0, 5):
            r_id = random.randint(0, total_name)
            name = name_list[r_id]
            output_name.append(name)
        output.append(" ".join(output_name))
        output_name = list()
        for i in range(0, 6):
            r_id = random.randint(0, total_name)
            name = name_list[r_id]
            output_name.append(name)
        output.append(" ".join(output_name))

    total_output = len(output)
    for idx, out in enumerate(output):
        try:
            dump(f"\"{out}\"" + ",COMPANY")
        except:
            pass
        print(f"{idx}/{total_output}")
