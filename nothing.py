import sympy
def permu(str):
    cur = []
    concat = []
    if(len(str)==1):
        return str

    for i in range(len(str)):
        cur = permu(str[:i] + str[i+1:])
        for substr in cur:
            concat.insert(0,str[i]+substr)
    return concat
def compress(chain):
    count = 0
    streak = 0
    str2=""
    last = chain[0]
    for char in chain:
        if(char == last):
            streak+=1
            count+=1
        else:
           str2= str2 + last + str(count)
           streak -= 1
           count = 1
        last = char
        print(streak)
    str2 = str2 + last + str(count)
    if(streak>0):
        return str2
    return chain
import numpy as np
def rot(arrayn):
    temp = np.where(arrayn == 0)
    arrayn[temp[0],:] = 0
    arrayn[:,temp[1]] = 0
    return arrayn

class Qu:
    def __init__(self):
        self.content = []
    def is_empty(self):
        return len(self.content) == 0
    def add(self,data):
        self.content.insert(0,data)
    def remove(self):
        return self.content.pop()

class Node:
    def __init__(self,name):
        self.name = name
        self.childern = []
        self.parents = []
        self.has_parent = False
        self.visited = False
    def add_parent(self,parent):
        self.parents.append(parent)
    def add_child(self,child):
        self.childern.append(child)

def proj(projects, dependecies):
    proj_dict = {}
    for proj in projects:
        proj_dict[proj] = Node(proj)

    for dep in dependecies:
        potential_parent = proj_dict[dep[0]]
        potential_child = proj_dict[dep[1]]
        if(potential_child.name in potential_parent.parents or potential_parent.name in potential_child.childern):
            print("nope")
            return 0
        potential_child.add_parent(potential_parent.name)
        potential_child.has_parent = True
        potential_parent.add_child(potential_child.name)
    Q = Qu()
    for proj in projects:
        if len(proj_dict[proj].parents)== 0:
            Q.add(proj)
            proj_dict[proj].visited = True
    while(not Q.is_empty()):
        cur = Q.remove()
        if (len(proj_dict[cur].parents) != 0):
                print("nooooo")
                return False
        for child in proj_dict[cur].childern:
            proj_dict[child].parents.remove(cur)
        print(cur)
        for proj in proj_dict[cur].childern:
           if(proj_dict[proj].visited == False):

               proj_dict[proj].visited = True
               Q.add(proj)


def wave( A):
    A = sort(A)
    B = [0] * len(A)
    if (len(A) % 2 == 1):
        B[:len(A)-1:2] = A[1:len(A) - 1:2]
        B[1:len(A)-1:2] = A[:len(A) -1:2]
        B[len(A) - 1] = A[-1]
    else:
        B[::2] = A[1::2]
        B[1::2] = A[::2]
    return B

def sort(A):
        if len(A) <= 1:
            return A
        pivot = len(A) - 1
        wall = -1
        for i in range(len(A)):
            if (A[i] <= A[pivot]):
                temp = A[wall + 1]
                A[wall + 1] = A[i]
                A[i] = temp
                wall += 1
        return sort(A[:wall]) + sort(A[wall:])


def premu(mystr):
    if(len(mystr)==1):
        return mystr
    cur = []
    sent_list = []
    for index in range(len(mystr)):
        if(not mystr[index] in sent_list):
            sent_list.append(mystr[index])
            ret = premu(mystr[:index]  + mystr[index+1:])
            for item in ret:
                   cur.append(mystr[index] + item)
    return cur



#proj(['a','b','c','d'],[('d','a'),('a','b'),('b','c'),('c','a')])
def non_rec(n):
    paren(0,0,n,"")

def paren(open,closed, n, cur):
   if open == n:
       if closed == n:
           print(cur)
       else:
           paren(open,closed +1 ,n,cur+")")
   else:
       if(closed < open):
         paren(open + 1,closed,n,cur+"(")
         paren(open,closed+1,n,cur+")")
       elif(closed== open):
           paren(open +1,closed,n,cur+"(")

money_arry = [25,10,5,1]
def calc(cur_usage, sum_of_money):
    s1 = money_arry[0] * cur_usage[0]
    s2 = money_arry[1] * cur_usage[1]
    s3 = money_arry[2] * cur_usage[2]
    s4 = money_arry[3] * cur_usage[3]
    sum = s1+s2+s3+s4
    if(sum > sum_of_money):
        return -1
    if(sum == sum_of_money):
        return 0
    return 1

def change(sum_of_money,work_with):
     if(sum_of_money == 0):
         return 1
     if(sum_of_money < 0):
         return 0
     total = 0
     i = work_with
     for item in money_arry[work_with:]:
       cur_sum = sum_of_money
       i+=1
       while(cur_sum - item >= 0):
           cur_sum -= item
           total += change(cur_sum,i)

     return total

prime_arr = list(sympy.sieve.primerange(0, 100))

def get_hash_code(word):
    sum = 0
    for letter in word:
        val = ord(letter) - 97
        sum += prime_arr[val]
    return sum

def comp(word1,word2):

        if(get_hash_code(word1) <= get_hash_code(word2)):
            return 1


def anagram(list_of_words):
        if len(list_of_words) <= 1:
            return list_of_words
        pivot = len(list_of_words) - 1
        wall = -1
        for i in range(len(list_of_words)):
            if (comp(list_of_words[i],list_of_words[pivot])):
                temp = list_of_words[wall + 1]
                list_of_words[wall + 1] = list_of_words[i]
                list_of_words[i] = temp
                wall += 1
        return anagram(list_of_words[:wall]) + sort(list_of_words[wall:])

def search_rot(lst):
    start = 0
    mid =int(len(lst) / 2)
    end = len(lst) - 1
    while (end- mid > 0 and mid-start >0):
        if(lst[start] > lst[mid]):
            end = mid
            mid = int((start + end)/2)
        elif(lst[mid] > lst[end]):
            start = mid
            mid = int((start + end)/2)
        else:
            mid = 0
    return mid +1

#b[1:,::2] = a[1::2,1::2]
def largestNumber(A):
        B=[]
        for i in range(len(A)):
            B.append(str(A[i]))
        get_list = recursive_buckets(B,9)
        res =""
        for item in get_list:
           res += item[1:]
        return res


def recursive_buckets(A, cur_buck):
    if (len(A) == 1): return [str(cur_buck) + A[0]]
    key = cmp_to_key(lambda a, b: 1 if a + b >= b + a else -1)
    buckets = [[] for i in range(11)]
    ret = []
    for item in A:
        if (len(item) == 0):
            cur_first_digit = 10
        else:
            # we are working wit strings, so we convert to int and find the first digit
            cur_first_digit = int(int(item) / (10 ** (len(item) - 1)))
        # then we get the string that is after that digit and place it in a bucket
        # according to its first digit
        cur_rest_of_item = item[1:]
        buckets[cur_first_digit].append(cur_rest_of_item)
    ret_list = []
    # all buckets same or bigger then cur will be added more left, and others more right
    for i in range(9, cur_buck, -1):
        if (len(buckets[i]) > 0):
            ret_list += sorted(recursive_buckets(buckets[i], i), key=key, reverse=True)
    ret_list += [""] * len(buckets[10])
    for i in range(cur_buck, -1, -1):
        if (len(buckets[i]) > 0):
            ret_list += sorted(recursive_buckets(buckets[i], i), key=key, reverse=True)
    for i in range(len(ret_list)):
        ret_list[i] = str(cur_buck) + ret_list[i]
    return ret_list
from functools import cmp_to_key


def subUnsort(A):
    cur_min = min(A[0],A[1])
    cur_max = max(A[0],A[1])
    start_need_sort = 0
    still_good = True
    for i in range(len(A) - 1):
        if (cur_max > A[i + 1]):
            still_good = False
            if (A[i + 1]) >= cur_min:
                end_need_sort = i + 1
            else:
                end_need_sort = i + 1
                if(A[i+1]<cur_min):
                    while (A[i + 1] < A[start_need_sort] and start_need_sort > 0):
                        start_need_sort -= 1
                    if(start_need_sort > 0 or A[i+1] >= A[start_need_sort] ):
                        start_need_sort +=1
                    cur_min = A[i+1]
        else:
            cur_max = A[i+1]
        if (still_good):
            start_need_sort = i+1
            cur_min = A[start_need_sort]

    if (still_good): return -1
    return [start_need_sort,end_need_sort]

def firstMissingPositive( A):
        j = 0
        for item in A:
            if (item != 'good' and item > 0 and item <= len(A)):
                    bucket = item-1
                    temp = A[bucket]
                    A[bucket] = 'good'
                    while(temp != 'good' and temp>0 and temp< len(A)):
                        bucket =  temp - 1
                        temp = A[bucket]
                        A[bucket] = 'good'

        for i in range(len(A)):
            if (A[i] != 'good'):
                return i + 1
        return len(A)+1

def maxSpecialProduct(A):

        maxi = 0
        for i in range(len(A)):
            cur_right = -1
            cur_left = -1
            for j in range(max(i,len(A)-i)):
                left_index = i-j-1
                right_index = i+j+1
                if(left_index >= 0 and A[left_index]>A[i] and cur_left == -1 ):
                    cur_left = left_index
                if(right_index < len(A) and A[right_index] > A[i] and cur_right == -1):
                    cur_right = right_index
            if(cur_left == -1):
                cur_left = 0
            if(cur_right == -1):
                cur_right = 0
            if(cur_left*cur_right>maxi):
                maxi = cur_left*cur_right
        return maxi

#B= [ 699, 2, 690, 936, 319, 784, 562, 35, 151, 698, 126, 730, 587, 157, 201, 761, 956, 359, 198, 986, 915, 7, 703, 324, 814, 382, 294, 204, 120, 731, 615, 330, 486, 52, 223, 376, 649, 458, 564, 971, 72, 605, 177, 20, 461, 790, 872, 363, 916, 435, 991, 184, 410, 320, 16, 480, 768, 801, 117, 338, 650, 786, 17, 369, 979, 304, 445, 688, 862, 229, 311, 351, 985, 697, 135, 299, 310, 3, 643, 221, 831, 196, 887, 679, 484, 209, 824, 292, 588, 721, 140, 675, 827, 913, 271, 170, 812, 552, 334, 860, 981, 550, 308, 584, 442, 328, 251, 456, 976, 31, 507, 954, 982, 742, 45, 727, 794, 309, 527, 623, 56, 843, 436, 681, 143, 130, 689, 870, 362, 580, 560, 474, 385, 525, 881, 51, 890, 917, 820, 826, 139, 443, 978, 144, 512, 205, 682, 188, 344, 429, 497, 181, 749, 864, 664, 145, 621, 629, 886, 572, 89, 725, 945, 29, 553, 977, 783, 590, 236, 728, 125, 90, 492, 261, 543, 259, 662, 622, 285, 392, 561, 670, 200, 504, 246, 513, 910, 583, 460, 179, 207, 709, 127, 926, 816, 426, 520, 174, 464, 883, 780, 5, 268, 606, 1, 109, 704, 391, 661, 924, 516, 241, 477, 952, 405, 522, 247, 335, 356, 839, 423, 779, 4, 43, 720, 238, 965, 951, 914, 10, 496, 775, 651, 788, 373, 491, 746, 799, 518, 93, 86, 774, 652, 955, 494, 252, 781, 946, 412, 202, 741, 719, 612, 673, 896, 1000, 289, 554, 69, 424, 980, 506, 593, 889, 25, 959, 28, 736, 8, 969, 865, 657, 567, 434, 9, 167, 357, 929, 645, 250, 565, 94, 928, 473, 509, 823, 313, 762, -1, 208, 903, 922, 655, 948, 326, 485, 150, 73, 505, 225, 122, 129, 648, 838, 811, 972, 735, 78, 428, 740, 782, 632, 316, 440, 737, 297, 873, 281, 479, 654, 0, 633, 212, 152, 154, 470, 866, 79, 722, 958, 732, 900, 832, 278, 58, 842, 745, 540, 169, 347, 592, 438, 882, 462, 53, 34, 519, 489, 85, 757, 919, 701, 15, 211, 667, 637, 74, 573, 240, 559, -2, 472, 203, 112, 162, 776, -4, 155, 837, 99, 98, 64, 101, 983, 366, 853, 970, 482, 40, 921, 374, 758, 413, 339, 705, 771, 360, 734, 282, 219, 766, 535, 133, 532, 254 ]
#B = [699, 2, 690, 936, 319, 784, 562, 35, 151, 698, 126, 730, 587, 157, 201, 761, 956, 359, 198, 986, 915, 7, 703, 324, 814, 382, 294, 204, 120, 731, 615, 330, 486, 52, 223, 376, 649, 458, 564, 971, 72, 605, 177, 20, 461, 790, 872, 363, 916, 435, 991, 184, 410, 320, 16, 480, 768, 801, 117, 338, 650, 786, 17, 369, 979, 304, 445, 688, 862, 229, 311, 351, 985, 697, 135, 299, 310, 3, 643, 221, 831, 196, 887, 679, 484, 209, 824, 292, 588, 721, 140, 675, 827, 913, 271, 170, 812, 552, 334, 860, 981, 550, 308, 584, 442, 328, 251, 456, 976, 31, 507, 954, 982, 742, 45, 727, 794, 309, 527, 623, 56, 843, 436, 681, 143, 130, 689, 870, 362, 580, 560, 474, 385, 525, 881, 51, 890, 917, 820, 826, 139, 443, 978, 144, 512, 205, 682, 188, 344, 429, 497, 181, 749, 864, 664, 145, 621, 629, 886, 572, 89, 725, 945, 29, 553, 977, 783, 590, 236, 728, 125, 90, 492, 261, 543, 259, 662, 622, 285, 392, 561, 670, 200, 504, 246, 513, 910, 583, 460, 179, 207, 709, 127, 926, 816, 426, 520, 174, 464, 883, 780, 5, 268, 606, 1, 109, 704, 391, 661, 924, 516, 241, 477, 952, 405, 522, 247, 335, 356, 839, 423, 779, 4, 43, 720, 238, 965, 951, 914, 10, 496, 775, 651, 788, 373, 491, 746, 799, 518, 93, 86, 774, 652, 955, 494, 252, 781, 946, 412, 202, 741, 719, 612, 673, 896, 1000, 289, 554, 69, 424, 980, 506, 593, 889, 25, 959, 28, 736, 8, 969, 865, 657, 567, 434, 9, 167, 357, 929, 645, 250, 565, 94, 928, 473, 509, 823, 313, 762, -1, 208, 903, 922, 655, 948, 326, 485, 150, 73, 505, 225, 122, 129, 648, 838, 811, 972, 735, 78, 428, 740, 782, 632, 316, 440, 737, 297, 873, 281, 479, 654, 0, 633, 212, 152, 154, 470, 866, 79, 722, 958, 732, 900, 832, 278, 58, 842, 745, 540, 169, 347, 592, 438, 882, 462, 53, 34, 519, 489, 85, 757, 919, 701, 15, 211, 667, 637, 74, 573, 240, 559, -2, 472, 203, 112, 162, 776, -4, 155, 837, 99, 98, 64, 101, 983, 366, 853, 970, 482, 40, 921, 374, 758, 413, 339, 705, 771, 360, 734, 282, 219, 766, 535, 133, 532, 254 ]
#print(firstMissingPositive([1,4,3,2,5,7,6]))

def repeatedNumber( A):
            A = list(A)
            for j in range(len(A)):
                if (A[j] > 0):
                    first_time = True
                    bucket = A[j] - 1
                    temp = A[bucket]
                    A[j]=-10
                    while (temp > 0 or first_time):
                        first_time =False
                        temp = A[bucket]
                        if (A[bucket] == -1):
                            A[bucket] = -2
                        else:
                            A[bucket] = -1
                        bucket = temp - 1


            B = [0] * 2
            for i in range(len(A)):
                if (A[i] == -10):
                    B[1] = i+1
                if (A[i] == -2):
                    B[0] = i+1
            return B
#print(repeatedNumber([3,1,2,5,3]))
def subs(A):
    if(len(A)==1):
        return [A]
    cur_list = [[A[0]]]
    history = subs(A[1:])
    for p in history:
        cur_list.append([A[0]] + p)
    return cur_list + history

def subsets(mystr):
    if(len(mystr)== 1):
        return [mystr]
    cur_poss = [[mystr[0]]]
    prev = subsets(mystr[1:])
    for item in prev:
        cur_poss.append([mystr[0]]+item)
    return   cur_poss + prev



def sort_asc(A):
    if (len(A) <= 1):
        return A
    pivot = A[len(A) - 1]
    wall = -1
    for i in range(len(A)):
        if (A[i] >= pivot):
            wall += 1
            temp = A[i]
            A[i] = A[wall]
            A[wall] = temp
    return sort_asc(A[:wall]) + sort_asc(A[wall:])


import math


def product_of_divisors(num):
    sum = 1
    for i in range(num, int(math.sqrt(num)) - 1, -1):
        if (num % i) == 0:
            sum = sum * i
            rest = num /i
            if(rest != i):
                sum *= rest
    if(sum == 47829690000000 ):
        print("look")
    return int(sum)

def solve(A, B):
        G = []
        for i in range(len(A)):
            for j in range(len(A), i, -1):
                G.append(max(A[i:j]))
        for i in range(len(G)):
            G[i] = product_of_divisors(G[i])
        G.sort(reverse=True)
        return [G[i-1] for i in B]


def nextPermutation(A):
    for i in range(len(A) - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            if (A[i] > A[j]):
                temp = A[i]
                A[i] = A[j]
                A[j] = temp
                index = j
                return (sort_asc_from_index(A, index + 1))
    return (sort_asc_from_index(A, 0))


def sort_asc_from_index(A, index):
    replace = sort(A[index + 1:])
    for i in range(len(A) - index - 1):
        A[i + index] = replace[i]
    return A

def generateMatrix(A):
        count = 1
        start = 0
        end = A
        mat = [[0 for i in range(A)]for i in range(A)]
        while(end - start >=0):
            if(end-start == 0 and A%2 ==1):
                mat[start][start] = count
                return mat
            for i in range(start,end):
                mat[start][i] = count
                count+=1
            for i in range(start+1,end):
                mat[i][end-1] =count
                count+=1
            for i in range(end-2,start-1,-1):
                mat[end-1][i] = count
                count+=1
            for i in range(end-2,start-1,-1):
                if(mat[i][start]!=0):
                    break
                mat[i][start] = count
                count+=1
            end-=1
            start+=1
        return mat



def repeatedNumber(self, A):
    if (len(A) < 3):
        return A[0]
    item1 = 12434354436
    item2 = 421588584393
    count1 = 0
    count2 = 0
    for item in A:
        if (item == item1):
            count1 += 1
        elif (item == item2):
            count2 += 1
        elif (count1 == 0):
            item1 = item
            count1 = 1
        elif (count2 == 0):
            item2 = item
            count2 = 1
        else:
            count1 -= 1
            count2 -= 1

    count1 = 0;
    count2 = 0
    for item in A:
        if item == item1:
            count1 += 1
            if (count1 >= len(A) / 3): return item1
        if item == item1:
            count2 += 1
            if (count2 >= len(A) / 3): return item2

    return -1

def findPerm( A, B):
        sum_u = 0
        sum_d = 0
        cur_count = 0
        follow = []
        cur = A[B - 2]
        for s in  reversed(A):
            if (s == "I"):
                sum_u += 1
            else:
                sum_d += 1
            if (s != cur):
                follow.append((cur, cur_count))
                cur = s
                cur_count = 1
            else:
                cur_count += 1
        follow.append((cur, cur_count))
        ret = [0] * B
        pos = B - 1
        max = B - sum_d
        min = max
        ret[pos] = B - sum_d
        for item in follow:
            if (item[0] == "I"):
                start = min - item[1]
                end = min
                start_pos = pos - item[1]
                ret[start_pos:pos] = [i for i in range(start, end)]
                pos = start_pos
            else:
                start = max + item[1]
                end = max
                start_pos = pos - item[1]
                ret[start_pos:pos] = [i for i in range(start, end, -1)]
                pos = start_pos
        return ret
        # @return a list of integers


def primesum(A):
    for i in range(2, A, 1):
        if (is_prime(i) and is_prime(A - i)):
            return [i, A - i]


def is_prime(n):
    if (n == 2):
        return True
    if (n % 2 == 0):
        return False
    for i in range(2, int(math.sqrt(n)+1)):
        if (n % i) == 0:
            return False
    return True


def search(A, B):
        start = 0
        end = len(A) - 1
        mid = int(end / 2)
        if (A[start] > A[end - 1]):

            while (mid >= 0 and mid < len(A) - 1 and A[mid] < A[mid + 1]):
                if (A[mid] > A[start]):
                    start = mid
                else:
                    end = mid
                mid = int((start + end) / 2)

            if (B > A[len(A)-1]):

                start = 0
                end = mid

            else:
                end = len(A) - 1
                start = mid + 1

            mid = int((start + end) / 2)

        while (start != mid and end != mid):

            if (A[mid] == B):
                return mid

            if (A[mid] > B):

                end = mid

            else:
                start = mid

            mid = int((start + end) / 2)
        if(A[mid]==B):return mid
        return -1

import re
def isPalindrome( A):
    pattern = re.compile("[^A-Za-z\d]")
    end = len(A) - 1
    start = 0
    for i in range(len(A)):
        while pattern.match(A[i + start]):
            start = start + 1

        while pattern.match(A[end - i]):
            end = end - 1

        if (A[i + start] != A[end - i] and abs(ord(A[i+start]) - ord(A[end-i])) != (ord('a')-ord('A'))):
            return 0

        if (i + start >= end - i):
            return 1


class Solution:
    # @param A : integer
    # @return a list of list of integers
    def prettyPrint(self, A):
        if A == 1:
            return [[1]]
        arr = Solution.prettyPrint(self, A - 1)
        for i in range(len(arr)):
            arr[i].insert(0, A)
            arr[i].insert(len(arr[0]), A)
        arr.insert(0, [A for i in range(len(arr[0]))])
        arr.insert(len(arr), [A for i in range(len(arr[0]))])
        return arr

def temp():
    import tweepy
    import operator
    import webbrowser


    consumer_key        = "uPpV5uSZ1KsKUen46e8rE1gyT"
    consumer_secret     = "i2STWuWUK55z0YF4qzLyAMqxePJO7EyaM2rCSvE2abAotrKWjz"
  #  access_token        = "1169206450843000833-tLcDvyzDbMmhrMz3sf4krDYDJwzsFN"
   # access_token_secret = "9Fp9xyh7qSUUChg7N4eDUPS8n60BG8cgJSKpmPeBtVe33"
    access_token = "1169206450843000833-kWvRhMqnseBik3OYsxmrfsefmTJUxd"
    access_token_secret = "zQWWm38qHFrt3mkulMZqu0JcueCbAmPyiQbd8oAHWRMCa"


    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit_notify=False)
    print(1)
    target = "eyalio"
    try:
        gen_of_Tweets_by_requested_user = tweepy.Cursor(api.user_timeline,  count=100 ,id=target).items()
        lst = []
        print(2)
        i = 0
        for item in gen_of_Tweets_by_requested_user:

            lst.append(item._json['favorite_count'])
            i+=1
            if(i>100):break

        lst.sort(reverse=True)
        print(lst)
    except Exception as e:
            print(e.response.status_code)
    #a=[item._json["screen_name"] for item in list(tweepy.Cursor(api.friends).items())]
    #print(a)
# Get access token
#auth.get_access_token("verifier_value")

# Construct the API instance

#temp()

def minimize(A, B, C):
    cur_min_max_diff = 12434243
    for i in range(len(A)):
        expr = max(abs(A[i] - B[0]), abs(A[i] - C[0]), abs(B[0] - C[0]))
        if expr < cur_min_max_diff:
            cur_min_max_diff = expr

    for i in range(len(B)):
        expr = max(abs(A[len(A) - 1] - B[i]), abs(A[len(A) - 1] - C[0]), abs(B[i] - C[0]))
        if expr < cur_min_max_diff:
            cur_min_max_diff = expr
    for i in range(len(C)):
        expr = max(abs(A[len(A) - 1] - B[len(B) - 1]), abs(A[len(B) - 1] - C[i]), abs(B[len(B) - 1] - C[i]))
        if expr < cur_min_max_diff:
            cur_min_max_diff = expr
    return cur_min_max_diff

def numRange(A, B, C):

    return helper(A,C) - helper(A,B-1)


def helper(A, B):
    s = 0
    e = 1
    count = 0
    calc = 0
    while (s < len(A) and e < len(A) + 1):
        calc = sum(A[s:e])
        if (calc <= B):
            e += 1
            count += e - s - 1
        else:

            s += 1
            if (s == e):
                e += 1

    return count
def subsets1(A):
        A = sorted(A)
        return rec(A)
def k(item):
    if(item == []):
        return -12
    else:
        return item[0]
def rec(A):
        if(len(A)==0):
            return [[]]
        cur_sub = []
        res = rec(A[1:])
        for item in res:
            cur_sub.append(item)
        for item in res:
            cur_sub.append([A[0]]+item)
        return sorted(cur_sub,key=k)






def rec(A, cur_str, opened, closed):
    if (opened == A and closed == A):
        return [cur_str]
    cur = []
    o = []
    c = []
    if (opened < A):
        for item in rec(A, cur_str + "(", opened + 1, closed):
            o.append(item)
    if (closed < opened):
        for item in rec(A, cur_str + ")", opened, closed + 1):
            c.append(item)
    cur = o + c
    return cur

def permute( A):
        return recu([], A)


def recu(head, reminder):

    if (len(reminder) == 0):
        return [[head]]
    cur = []
    for i in range(len(reminder)):
        res=  recu(reminder[i], reminder[:i] + reminder[i + 1:])
        for item in res:
            if(head):
                cur.append([head]+ item )
            else:
                cur.append(item)
    return cur


# Definition for singly-linked list.
class ListNode:
	def __init__(self, x,nex):
		self.val = x
		self.next = nex


def reverseBetween(A, B, C):
    head = A
    cur = A
    for i in range(1, B-1 ):
        cur = cur.next
    if(B>1):
        staring_point = cur
        remember_for_last = staring_point.next
        rev_start,end_point = reverse(staring_point.next,C-B)
    else:
        staring_point = None
        remember_for_last = A
        rev_start,end_point = reverse(A,C-B)
    if(staring_point != None):
        staring_point.next = rev_start
    remember_for_last.next = end_point
    if(staring_point != None):
        return head
    else:
        return rev_start

def reverse(A,C):
    cur = A
    prev = None
    nex = A.next
    i=0
    while(i<C+1):
        cur.next = prev
        prev = cur
        cur  = nex
        i+=1
        if(cur!=None):
            nex  = cur.next
    return prev,cur
def make_array(N):
    a =[]
    while(N!=None):
        a.append(N.val)
        N=N.next
    return a

"""a1 = ListNode(12,None)
a2= ListNode(19,a1)
    a3= ListNode(18,a2)
    a31= ListNode(17,a3)
    a4= ListNode(16,a31)
    a5= ListNode(15,a4)
    a6= ListNode(14,a5)
    a7= ListNode(13,a6)
    a8= ListNode(12,a7)
a9= ListNode(11,a1)
a10= ListNode(10,a9)
a11= ListNode(9,a10)
a12= ListNode(8,a11)"""
a122= ListNode(7,None)
a13= ListNode(6,None)
a14= ListNode(5,a13)
a15= ListNode(4,a14)
a16= ListNode(3,a15)
a17= ListNode(2,a16)
a18= ListNode(1,a17)

print(make_array(a18))
res = reverseBetween(a18,4,4)
print(make_array(res))
#res2 = reverse(a18,4)

#print(make_array(res2[0]))
#print(make_array(res2[1]))

def braces( A):
    stack = []
    for item in A:
        if (item in "*/-+("):
            stack.append(item)
        elif (item == ")"):
            if (stack.pop() == '('):
                return 1
            while (stack.pop() != '('):
                if(len(stack) == 0):
                    break



    return 0


def simplifyPath( A):
    stack = []
    i = 0
    while (i < len(A) - 1):
        if A[i] == '.' and A[i + 1] == '.':
            i += 1
            if len(stack) > 0:
                stack.pop()

        else:
            cur = ""
            if (A[i] != '/' and A[i] != '.'):
                while (A[i] != '/' and i < len(A) - 1):
                    cur += A[i]
                    i + 1
            i += 1
            if (cur != ""):
                stack.append(cur)

    st = ""
    if (len(stack) == 0):
        return "/"
    for item in stack:
        st += "/" + item

    return st


def trap( A):
    stack = []
    sum1 = 0
    for index in range(len(A)):
        if (len(stack) == 0 or A[stack[-1]] > A[index]):
            stack.append(index)
        elif (A[stack[-1]] == A[index]):
            pass
        else:
            while (A[stack[-1]] <= A[index]):
                p = A[stack.pop()]
                if (len(stack) == 0):
                    break
                top_index = stack[-1]
                top_val = min(A[top_index],A[index])
                sum1 += (top_val - p) * (index - top_index - 1)
            stack.append(index)
    return sum1

def colorful(A):
        B = create_list(A)
        arr = [[] for i in range(46)]
        for i in range(len(B)):
            res = B[i]
            for j in range(i + 1, len(B)):
                if (res in arr[int(res % 45)]):
                    return 0
                arr[int(res % 45)].append(res)
                res *= B[j]

        return 1


def create_list(A):
    l = []
    while (A > 0):
        l.append(A % 10)
        A = A / 10
    return l
def lszero( A):
        maxi = 0
        s = 0
        e=0
        arr = [[] for i in range(100)]
        for i in range(len(A)+1):
            sum1 = sum(A[:i])
            arr[sum1%100].append((sum1,i))
        for i in range(100):
            if(len(arr[i])>1):
                f = arr[i][0]
                s = arr[i][-1]
                if (f[0] == s[0]):
                    temp = s[1] - f[1]
                    if (temp > maxi):
                        maxi = temp
                        start = f[1]
                        end = s[1]
        if (maxi == 0):
            return None
        return A[start:end ]


def my_i(A):
    arr = []
    res  =rec_help(A,arr)
    return arr
def rec_help(A,arr):
    if(len(A)==1):
        return [A]

    res = [[A[0]]]
    res1 =   rec_help(A[1:],arr)
    for item in res1:
            res.append(item)
            if(len (item) < 3):
                res.append([A[0]] + item)
            else:
                arr.append([A[0]] + item)
    return res

def findSubstring(A, B):
        dicta = {}
        count = len(B)
        goods = []
        window = len(B[0])
        cstart = 0
        for item in B:
            dicta[item] = 0
        for item in B:
            dicta[item] +=1
        end = len(A)
        for i in range(end):
            checking = A[cstart + i:cstart + i + window]
            while (checking in dicta and dicta[checking] >0
                   and count > 0 and cstart <= end):
                dicta[checking] -=1
                cstart += window
                count -= 1
                checking = A[cstart + i:cstart + i + window]
            if (count == 0):
                goods.append(i)
            count = len(B)
            cstart = 0
            for key in dicta.keys():
                dicta[key] = 0
            for item in B:
                dicta[item] +=1

        return goods
class TreeNode:
    def __init__(self, x,one,two):
        self.val = x
        self.left = one
        self.right = two





def isBalanced( A):
    if (helper(A) < 0):
        return 0
    return helper(A)


def helper( A):
    if (A == None):
        return 0

    rleft = helper(A.left) + 1
    rright = helper(A.right) + 1
    if (rleft < 0 or rright < 0 or abs(rleft - rright) > 1):
        return -2
    else:
        return max(rleft, rright)

def isSameTree( A, B):
        q1 = []
        q2 = []
        q1.append(A)
        q2.append(B)
        while(len(q1)>0 and len(q2)>0):
            one = q1.pop()
            two = q2.pop()
            if(one == None and two == None):
                pass
            elif(one== None or two== None or one.val!= two.val):
                return 0
            else:
                q1.insert(0,one.right)
                q1.insert(0,one.left)
                q2.insert(0,two.right)
                q2.insert(0,two.left)
        return 1
t6 = TreeNode(15,None,None)
t5 = TreeNode(6,None,None)
t4 = TreeNode(7,t5,None)
t3 = TreeNode(0,None,None)
t2 = TreeNode(2,None,None)
t1 = TreeNode(3,t2,None)
t00 = TreeNode(2,None,None)
t0 = TreeNode(2,None,None)

def trav(A,count,arr):
    if(A==None):
        pass
    else:
        count = trav(A.left,count,arr)
        count+=1
        if(count==2):
            arr.append(A.val)
        print(A.val,count)
        count = trav(A.right,count,arr)
    return count

def nchoc(A, B):
        count = 0
        h = create_h(B)
        for i in range(A):
            t = remove_top(h)
            count += t
            insert(h, int(t / 2))
        return count

def create_h(A):
    h = []
    for item in A:
        insert(h,item)
    return h
def heapfiy_up(A):
    index = len(A) - 1
    while (index > 0 and A[index] > A[int(index / 2)]):
        temp = A[index]
        A[index] = A[int(index / 2)]
        A[int(index / 2)] = temp


def insert(A, x):
    A.append(x)
    heapfiy_up(A)


def remove_top(A):
    t = A[0]
    A[0] = A[len(A) - 1]
    A.pop()
    heapify_down(A)
    return t


def heapify_down(A):
    index = 0
    while (index < len(A) / 2):
        if (index * 2 + 1 <= len(A) - 1):
            if (A[index * 2] > A[index * 2 + 1]):
                t = index * 2
            else:
                t = index * 2 + 1
        else:
            t = index * 2
        if (A[t] > A[index]):
            temp = A[index]
            A[index] = A[t]
            A[t] = temp
        else:
            break

print(nchoc(3,[1]))