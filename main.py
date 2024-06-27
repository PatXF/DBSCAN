import time
from sk import *

import funcs as f

print("Enter eps and minpoints:")  #checked

eps = float(input("\teps: "))
minpts = int(input("\tminpoints: "))
file = input("\tfile: ")

start = time.time()
f.dbs(eps, minpts, file)
end = time.time()
print(end - start)

start = time.time()
dbscan(eps, minpts, file)
end = time.time()
print(end - start)