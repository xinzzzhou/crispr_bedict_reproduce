import csv
import matplotlib.pyplot as plt

line = 0
nums = []
num0 = 0
plot = [0 for a in range(1,102)]
with open('/home/data/bedict_reproduce/data/41467_2021_25375_MOESM2_ESM.csv', 'r') as f:
     reader = csv.reader(f)
     for row in reader:
          line += 1
          if line == 1:
               continue
          nums += [float(a) for a in row[4:23]]
     for num in nums:
          if num == 0.0:
               num0 += 1
          else:
               # print(int(num))
               plot[int(num)] += 1
     plt.xlabel('ground truth')
     plt.ylabel('count')
     # plt.xlim(xmax=9,xmin=0)
     # plt.ylim(xmax=9,xmin=0)
     for a in range(100):
          plt.scatter(a,plot[a])
     plt.legend()
     plt.title(str(num0)+"|"+str(len(nums)))
     plt.savefig('/home/data/bedict_reproduce/data/count.png')
     plt.show()
     # print(num0)
     # print(len(nums))