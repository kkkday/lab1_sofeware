
with open("E:\\csnerwork\\软件工程lab1\\tmp.txt","r",encoding="utf-8") as file:
    sentence = file.read()

sentence = sentence.replace(".",",")
print(sentence)

