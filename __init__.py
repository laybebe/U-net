# def pre_process(train, test, indexlist):
#     for index in indexlist:
#         temp = []
#         # print(len(train))
#         for i in range(len(train)):
#             temp.append(train[i][index])
#         tmax, tmin = max(temp), min(temp)
#         p1 = tmin + (tmax - tmin) / 3
#         p2 = tmax - (tmax - tmin) / 3
#         for i in range(len(train)):
#             if temp[i] < p1:
#                 train[i][index] = 0
#             elif temp[i] < p2:
#                 train[i][index] = 1
#             else:
#                 train[i][index] = 2
#         if test[index] < p1:
#             test[index] = 0
#         elif test[index] < p2:
#             test[index] = 1
#         else:
#             test[index] = 2
#     return train, test
# 4 5
# 0 1 0 0 0
# 0 2 3 0 0
# 0 0 4 5 0
# 0 0 7 6 0
# 0 1 3 2


# def dfs(start_x, start_y, end_x, end_y, migong_array):
#     next_step = [[0, 1],[1, 0],[0, -1],[-1, 0]]
#     if (start_x == end_x and start_y == end_y):
#         global count
#         count += 1
#         return
#
#     for i in range(len(next_step)):
#         last_x = start_x
#         last_y = start_y
#         next_x = start_x + next_step[i][0]
#         next_y = start_y + next_step[i][1]
#
#         if (next_x < 0 or next_y < 0 or next_x > len(migong_array) or next_y > len(migong_array[0])):
#             continue
#         elif (a[next_x][next_y] > a[last_x][last_y] and book[next_x][next_y] == 0):
#             book[next_x][next_y] = 1
#             dfs(next_x, next_y, end_x, end_y, migong_array)
#             book[next_x][next_y] = 0
#         else:
#             continue
#     return
#
#
# if __name__ == "__main__":
#     N, M = list(map(int, input().split()))
#     count = 0
#     a = [[0 for col in range(N)] for row in range(M)]  # map最大数组
#     book = [[0 for col in range(N)] for row in range(M)]  # 标记数组
#     # print(N)
#     map_array = []
#     for i in range(N):
#         map_array.append(list(map(int, input().split())))
#     node = list(map(int, input().split()))
#     start = node[0:2]
#     end = node[2:4]
#     if 0<=node[0]<N and 0<=node[1]<N and 0<=node[2]<M and 0<=node[0]<M:
#         for i in range(N):
#             for j in range(M):
#                 a[i][j] = map_array[i][j]
#         book[start[0]][start[1]] = 1  # 将第一步标记为1，证明走过了。避免重复走
#
#         dfs(start[0], start[1], end[0], end[1], map_array)
#         print((count % (10 ** 9)))
    # a=input()
    # b=""
    # string=""
    # for i in range(len(a)):
    #     if a[i]=="(" or  a[i]==")":
    #         continue
    #     else:
    #         b=b+a[i]
    # num=["0","1","2","3","4","5","6","7","8","9"]
    # while (i<len(b)-1):
    #     if (b[i] not in num) and (b[i+1] not in num):
    #         string=string+b[i]
    #         i=i+1
    #         continue
    #     elif (b[i] not in num) and b[i+1]  in num:
    #         j=i+1
    #         number=""
    #         while (j<len(b)):
    #             if b[j] in num:
    #                 number=number+b[j]
    #                 j=j+1
    #             else:
    #                 string = string + b[i]*int(number)
    #                 i=j
    #                 break
    #         continue
    #     else:
    #         break
    # print(string)




    # train = []
    # number_sample = int(input())
    # label = list(map(int, input().split()))
    # for i in range(number_sample):
    #     train.append(list(map(int, input().split())))
    # test = list(map(int, input().split()))
    # train, test = pre_process(train, test, [2, 3, 4])
    # f_num = len(test)
    # s_num = len(train)
    # p_y1 = sum(label) / s_num
    # p_y0 = 1 - p_y1
# n=int(input())
# num=[]
# for i in range(n):
#     num.append(int(input()))
# for i in range(n):
#     result=[]
#     temp=[(i+1) for i in range(num[i])]
#     for j in range(num[i]):
#         result.append(temp.pop(0))
#         if len(temp)!=0:
#             temp_num=temp.pop(0)
#             temp.append(temp_num)
#     print(result)
