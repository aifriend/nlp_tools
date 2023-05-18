import os
import random
import shutil

from common.ClassFile import ClassFile

if __name__ == '__main__':
    MAX_DOC = 800
    FILE_FORMAT = 'pdf'
    path = r""

    # Check whether the specified path exists or not
    path_empty = os.path.join(path, rf"./random")
    if not os.path.exists(path_empty):
        os.makedirs(path_empty)

    ext_list = ClassFile.list_files_ext(path, FILE_FORMAT)
    if len(ext_list) < MAX_DOC:
        print(f"Total:{len(ext_list)} docs less than MAX_DOC:{MAX_DOC}")
        exit()

    selection_list = random.sample(ext_list, k=MAX_DOC)
    total = len(selection_list)
    print(f"TOTAL FILE: {total}\nFROM: {path}")

    left = len(selection_list)
    for n, s_key in enumerate(selection_list, 1):
        try:
            f_path, f_name = os.path.split(s_key)
            shutil.copy(src=s_key,
                        dst=os.path.join(path_empty, f_name))
            print(f"{n}/{total} saved to {os.path.join(path_empty, f_name)}")
        except Exception as e:
            print(f"Error: {e}")
