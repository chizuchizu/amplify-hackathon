import os
# print("--pwd--")
# os.system("pwd")
# print("--ls--")
# os.system("ls")
# print("--ls token--")
# os.system("ls token")
if os.path.isfile("token/token.json"):
    print("Your token file exists.")
else:
    print("Your token file doesn't exist.")
