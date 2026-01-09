list1=list(map(int,input("enter the numbers: ").split()))
list2=list(map(int,input("enter the numbers: ").split()))
def common_elements(li1,li2):
   common=[]
   for i in li1:
     if i in li2 and i not in common:
         common.append(i)
   return common
 #function calling to find common elements
common_elements=common_elements(list1,list2)
print(common_elements)
         
