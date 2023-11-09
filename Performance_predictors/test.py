import os

temp = "yo"
if not(os.path.exists("./testdir_{}".format(temp))):
    os.makedirs("./testdir_{}".format(temp))