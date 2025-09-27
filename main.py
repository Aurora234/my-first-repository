import numpy
import random


def print_hi(name):
    """
    自定义的print函数
    :param name: 一个字符串或其他变量和字面量
    :return: None
    """
    print(f'Hi, {name}')


if __name__ == '__main__':
    # name = input("please input a name : ")
    # age = random.randint(1, 100)
    # tel = 1234567
    # print(f'PyCharm！ {name} age:{age} tel:{tel}')
    # print("hello! %s " % type(name))
    # if age == 11:
    #     print("你的年龄是%d" % age)
    # else:
    #     print("you failed")
    age = int(input("请输入一个数字:"))
    count = 0
    while age <= 80:
        print("你猜的年龄是%s" % age)
        age = random.randint(1, 100)
        count += 1

    print("运行了%d次" % count)
    print("结束时的年龄是%d" % age)
    print_hi("你很骄傲")
