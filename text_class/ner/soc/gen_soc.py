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
    soc_dic = ClassFile.file_to_list(SOURCE, binary=False, encoding="utf-8")
    soc_list = list(map(lambda x: x.pop(), soc_dic))
    del soc_dic
    total_soc = len(soc_list)
    random.shuffle(soc_list)

    total_output = len(soc_list)
    for idx, out in enumerate(soc_list):
        try:
            dump(f"\"{out}\"" + ",COMPANY")
        except:
            pass
        print(f"{idx}/{total_output}")
