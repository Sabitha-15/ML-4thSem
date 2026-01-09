import random
#functions to find mean median and mode
def mean(li1):
   val=sum(li1)
   mean=val/len(li1)
   return mean
def median(li1):
   list2=sorted(li1)
   length=len(list2)
   if length%2==0:
    median=list2[length//2 - 1]+list2[length//2 + 1]
   else:
    median=list2[length//2]
    return median
def mode(li1):
   maxcount=0
   for ele in li1:
    count=li1.count(ele)
    if count>maxcount:
        maxcount=count
        modee=ele
    return modee
list1=[random.randint(100, 150) for _ in range(100)]
mean1=mean(list1)
median1=median(list1)
mode1=mode(list1)
dict1={'Mean':mean1,'Median':median1,'Mode':mode1}
print(dict1)