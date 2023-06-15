import numpy as np
# flag = [0, 1, -1, 1, 1]
# flag = np.array(flag)
# mask = np.array(flag > 0, dtype = int)
# print(mask)


if __name__ == "__main__":
    with open("file_\\test.txt") as file:
        print(file)
        for line in file:
            print(line)
        content = file.read()
        c = dict(content)
        print(content + "\n" + "!")
        print(c)
    with open("file_\\res.txt", 'a') as file_res:
        file_res.write("dwadaw")
