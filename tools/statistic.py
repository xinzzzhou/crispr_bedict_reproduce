'''
Author: Xin Zhou
Date: 17 Sep, 2021
'''

import csv
import matplotlib.pyplot as plt

#计算平均数
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

#计算中位数
def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2)-1
        return listnum[i]
    else:
        i = int(lnum / 2)-1
        return (listnum[i] + listnum[i + 1]) / 2

#计算众数
def publicnum(num, d = 0):
    dictnum = {}
    for i in range(len(num)):
        if num[i] in dictnum.keys():
            dictnum[num[i]] += 1
        else:
            dictnum.setdefault(num[i], 1)
    maxnum = 0
    maxkey = 0
    for k, v in dictnum.items():
        if v > maxnum:
            maxnum = v
            maxkey = k
    return maxkey

def sum(list):
     s = 0
     for a in list:
          s+=a
     return s

editor='Target-AID'
line = 0
nums = []
num0 = 0
plot = [0 for a in range(101)]
plot_nozero = [0 for a in range(101)]
with open('/home/data/bedict_reproduce/data/test_data/'+editor+'/perbase.csv', 'r') as f:
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
               if int(num) != 0:
                    plot_nozero[int(num)] += 1
     plt.xlabel('ground truth')
     plt.ylabel('count')
     for a in range(100):
          plt.scatter(a,plot[a])
     plt.legend()
     # plt.title(("Total:%d|%d. Median:%d. Public:%d. Zero:%d.".format(len(nums)-num0, len(nums), mediannum(nums), publicnum(nums), num0)))
     # plt.title("Total As:"+str(len(nums)-num0)+"|"+str(len(nums)) +". Zero:"+str(num0) +". Probs>1:"+str(sum(plot_nozero))+". Median(no0):"+str(mediannum(plot_nozero))+". Public(no0):"+str(publicnum(plot_nozero)))
     plt.title(str(len(nums)-num0)+"|"+str(len(nums)) +". "+str(num0) +". "+str(sum(plot_nozero)) +". "+str(mediannum(plot_nozero))+". "+str(publicnum(plot_nozero)))
     plt.savefig('/home/data/bedict_reproduce/data/test_data/'+editor+'count.png')
     plt.show()
