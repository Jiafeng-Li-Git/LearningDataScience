
import re


# split a string into several part, each part has its duplicated char
# eg. input: hello  ----->  output: ["h", "e", "ll", "o"]
def splitWord(str):
    i = 0
    partition = []
    while i < len(str):
        p_str = str[i]
        while (i < len(str)-1) and (str[i] == str[i+1]):
            p_str = p_str + str[i+1]
            i = i + 1
        i = i + 1
        partition.append(p_str)
    return partition


# compare splitWord(w1) and splitWord(w2)
def isStretchyWord(tar_s, queryWord):
    sol = 0
    com_list1 = splitWord(tar_s)
    for word in queryWord:
        flag = 0
        com_list2 = splitWord(word)
        if len(com_list1) != len(com_list2):
            continue
        for i in range(0, len(com_list1)):
            if com_list1[i] == com_list2[i]:
                continue
            if bool(re.search(com_list2[i], com_list1[i])) and len(com_list1[i]) >= 3:
                continue
            flag = 1
        if flag == 0:
            sol = sol + 1
    return sol


print(splitWord("heeellooo"))
print(isStretchyWord("heeellooo", ["hello", "helo", "hi"]))