import os
import subprocess
from hopp.keys import set_nrel_key_dot_env
from hopp.keys import get_developer_nrel_gov_key
set_nrel_key_dot_env()
print(get_developer_nrel_gov_key())
# os.system("hsconfigure")
# os.system("https://developer.nrel.gov/api/hsds")
# 
p = subprocess.Popen('hsconfigure', shell=True, stdin=subprocess.PIPE, 
                      stdout=subprocess.PIPE, universal_newlines=True)
newline = os.linesep # [1]
commands = ['https://developer.nrel.gov/api/hsds','','', "R3KA4NjP9stcbvKQWOUQEITf6gx357SLY8gZgSMp",'Y']
p.communicate(newline.join(commands))
