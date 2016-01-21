import os

INIT_FILE = 'module_init.txt'

if os.path.exists(INIT_FILE):
    os.remove(INIT_FILE)
assert not os.path.exists(INIT_FILE)


import module_init

assert os.path.exists(INIT_FILE)

with open(INIT_FILE, 'r') as f:
    print(f.read())
