from scipy import stats
import numpy as np

seed = 1505095

def generate_random_num(n):
    rand_num = []
    z = []
    for i in range(n):
        if i == 0:
            z.append(seed)
            u_0 = z[0]/(2**31)
            rand_num.append(u_0)
        else:
            z.append((65539*z[i-1]) % (2**31))
            u_n = z[i]/(2**31)
            rand_num.append(u_n) 
        # print(rand_num[i])
    return rand_num

def range_sub_interval(k):
    total_range = 1
    sub_range = 1/k
    sub_interval = []

    start = 0

    for i in range(k):
        # print(start, start+sub_range)
        sub_interval.append([start, start+sub_range])
        start += sub_range

    return sub_interval

def get_frequency(lst, len, rand_num_list, n):
    freq = 0
    for i in range(n):
        if (lst[0] <= rand_num_list[i] < lst[len-1]):
            # print(lst[0], rand_num_list[i], lst[len-1])
            freq += 1
    return freq

def uniformity_test(n, a, k):
    rand_list = generate_random_num(n)
    xk = stats.chi2.ppf(q=(1-a),df=(k-1))
    # print(xk)
    sub_range_list = range_sub_interval(k)
    fj_sum = 0
    for j in range(k):
        fj = get_frequency(sub_range_list[j], len(sub_range_list[j]), rand_list, n)
        # print(fj)
        fj_sum += (fj-(n/k))**2
    # print(sub_rand_list)
    # for j in range(k):
    
    x2 = (k/n)*fj_sum
    print("chi", xk)
    if x2 > xk:
        print(x2, "we reject this hypothesis")
    else:
        print(x2, "the null hypothesis of uniformity is not rejected")

print("\n\n**************uniformity test************\n\n")
n = [20, 500, 4000, 10000]
k_arr = [10, 20]
for i in n:
    for k in k_arr:
        print("\nnumber of generated random numbers ", i, " sub intervals ", k, "\n")
        uniformity_test(i, 0.1, k)


def get_freq_range(rand_num, sub_range_list):
    range_num = 0

    for i in range(len(sub_range_list)):
        if(sub_range_list[i][0] <= rand_num < sub_range_list[i][1]):
            range_num = i
    return range_num

def get_matrix(d, k):
    n_arr = np.zeros((k,) * d)
    return n_arr

def serial_test(n, k, alpha, d):
    print("serial test")
    rand_list = generate_random_num(n)
    chi = stats.chi2.ppf(q=(1-alpha),df=((k**d)-1))
    l = int(np.floor(n/d))
    sub_range_list = range_sub_interval(k)
    u_list = []
    range_num_list = []
    freq_list = []
    x2d = 0
    sum_f = 0
    for i in range(0, n, d):
        if (len(rand_list[i:i+d]) == d):
            u_list.append(rand_list[i:i+d])

    for i in range(len(u_list)):
        for j in range(len(u_list[i])):
            range_num_list.append(get_freq_range(u_list[i][j], sub_range_list))

    for i in range(0, n, d):
        if (len(range_num_list[i:i+d]) == d):
            freq_list.append(range_num_list[i:i+d])

    # print(freq_list)

    mat = get_matrix(d, k)
    # print(mat)
    
    if(d == 2):
        for i in range(len(freq_list)):
            for j in range(1, len(freq_list[i])):
                mat[freq_list[i][j-1]][freq_list[i][j]] += 1

        for i in range(k):
            for j in range(k):
                sum_f += (mat[i][j] - (n/(k**d)))**2
    
    if(d == 3):
        for i in range(len(freq_list)):
            for j in range(1, len(freq_list[i])-1):
                # print(freq_list[i][j-1], freq_list[i][j], freq_list[i][j+1]) 
                mat[freq_list[i][j-1]][freq_list[i][j]][freq_list[i][j+1]] += 1
        
        for i in range(k):
            for j in range(k):
                for m in range(k):
                    sum_f += (mat[i][j][m] - (n/(k**d)))**2

    # print(mat)

    x2d = ((k**d)/n)*(sum_f)
    if x2d > chi:
        print(x2d, chi, "we reject this hypothesis")
    else:
        print(x2d, chi, "the null hypothesis of uniformity is not rejected")

print("\n\n**************serial test************\n\n")
n = [20, 500, 4000, 10000]
d_arr = [2, 3]
k_arr = [4, 8]
for i in n:
    for d in d_arr:
        for k in k_arr:
            print("\nnumber of generated random numbers ", i, " sub intervals ", k, " tuples ", d, "\n")
            serial_test(i, k, 0.1, d)

def groupSequence(lst): 
    res = [[lst[0]]] 
    run_arr = [0 for i in range(6)] 
  
    for i in range(1, len(lst)): 
        if lst[i-1] < lst[i]: 
            res[-1].append(lst[i]) 
  
        else: 
            res.append([lst[i]]) 
    for i in range(len(res)):
        if(len(res[i]) == 1):
            run_arr[0] += 1 
            # print(1, res[i])
        if(len(res[i]) == 2):
            run_arr[1] += 1
            # print(2, res[i])
        if(len(res[i]) == 3):
            run_arr[2] += 1
            # print(3, res[i])
        if(len(res[i]) == 4):
            run_arr[3] += 1
            # print(4, res[i])
        if(len(res[i]) == 5):
            run_arr[4] += 1
            # print(5, res[i])
        if (len(res[i]) >= 6):
            run_arr[5] += 1
            # print(6, res[i])
    return run_arr 

# run = groupSequence(generate_random_num(20)) 
# print(run)

def run_test(n, alpha):
    sum_arnb = 0
    a = [[4529.4, 9044.9, 13568, 18091, 22615, 27892],
         [9044.9, 18097, 27139, 36187, 45234, 55789],
         [13568, 27139, 40721, 54281, 67852, 83685],
         [18091, 36187, 54281, 72414, 90470, 111580],
         [22615, 45234, 67852, 90470, 113262, 139476],
         [27892, 55789, 83685, 111580, 139476, 172860]
        ]
    b = [1 / 6, 5 / 24, 11 / 120, 19 / 720, 29 / 5040, 1 / 840]

    print("run test")
    xa = stats.chi2.ppf(q=(1-alpha),df=(6))
    rand_list = generate_random_num(n)
    run_up_list = []
    ri = groupSequence(rand_list)
    print("run ", ri)
    for i in range(6):
        for j in range(6):
            sum_arnb += a[i][j]*(ri[i]-(n*b[i]))*(ri[j]-(n*b[j]))
    result = (1/n)*sum_arnb

    if result >= xa:
        print(result, xa, "we reject this hypothesis")
    else:
        print(result, xa, "the null hypothesis of uniformity is not rejected")

n = [20, 500, 4000, 10000]
print("\n\n**************run test************\n\n")
for i in n:
    print("\nnumber of generated random numbers ", i, "\n")
    run_test(i, 0.1)

def get_sum_u(u_lst, h, j):
    sum_u = 0
    for k in range(h):
        sum_u += u_lst[k*j] * u_lst[(k+1)*j]
    # print(u_lst[0])
    return sum_u

def gen_var_pj(h):
    var_pj = ((13*h) + 7)/((h+1)**2)
    return var_pj

def correlation_test(j, a, n):
    u = generate_random_num(n)
    h = int(np.floor(((n-1)/j)-1))

    sum_u = get_sum_u(u, h, j)
    
    pj = ((12/(h+1))*sum_u) - 3
    var_pj = gen_var_pj(h)

    aj = pj/np.sqrt(var_pj)
    
    z_alpha = stats.norm.ppf(q=(1-(a/2)))
    print(np.abs(aj), z_alpha)
    if np.abs(aj) > z_alpha:
        print("reject this hypothesis")
    else:
        print("the hypothesis of correlation is not rejected")

j_lag = [1, 3, 5]
n = [20, 500, 4000, 10000]
print("\n\n***********correlation test****************\n\n")
for i in n:
    for j in j_lag:
        print("\nnumber of generated random numbers ", i, " j lags ", k, "\n")
        correlation_test(j, 0.1, i)
# j = 1, 3, 5



# rand_list = generate_random_num(20)
# sub_range_list = range_sub_interval(10)
# for j in range(10):
#     fj = get_frequency(sub_range_list[j], len(sub_range_list[j]), rand_list)
#     print("*******fj:", fj)
# print(range_sub_interval(10))


# def sub_freq(n_range, lst):
#     f = 0
#     for i in range(len(lst)):
#         if(n_range[0] < lst[i] < n_range[1]):
#             print("range", n_range, "value ", lst[i])
#             f += 1
#     return f

# def sub_freq_range(n_range, sub_lst):
#     f = 0
#     for i in range(1, len(n_range)):
#         for j in range(len(sub_lst)):
#             if(n_range[i-1][0] < sub_lst[j] < n_range[i-1][1])
#     return 0

# def freq_for_serial_test(range_list, sub_list):
#     freq_arr = []
#     for i in range(len(range_list)):
#         for j in range(len(sub_list)):
#             print(sub_freq(range_list[i], sub_list[j]))
#     for i in range(len(sub_list)):
#         print(sub_freq_range(range_list, sub_list[i]))
#     # for k in range(len(range_list)):
#     #     freq = 0
#     #     for i in range(len(sub_list)):
#     #         for j in range(1, len(sub_list[i])):
#     #             if(range_list[k][0] < sub_list[i][j-1] <range_list[k][1] and range_list[k][0] < sub_list[i][j] <range_list[k][1]):
#     #                 print("from same range", range_list[k][0], sub_list[i], range_list[k][1])
#     #                 freq += 1
#     #         for j in range(len(sub_list[i])-1):        
#     #             if((range_list[k-1][0] < sub_list[i][j] < range_list[k-1][1]) and (range_list[k][0] > sub_list[i][j+1] > range_list[k][1])):
#     #                 print(range_list[k-1][0], sub_list[i][j], sub_list[i], range_list[k][1])
#     #                 freq += 1
#     #     freq_arr.append(freq)            

#     return freq_arr

# def get_sum_j(d, k, range_lst, lst):
#     for i in range(k):
#         for j in range(len(lst)):
#             for m in range(d):
#                 if (range_lst[i][0] < lst[j][m] < range_lst[i][1]):
#                     print(range_lst[i][0], lst[j][m], range_lst[i][1], "element", m+1, "range", i+1)


# def demo(d, k, u_list):
#     mat = get_matrix(d, k)
#     sub_range_list = range_sub_interval(k)
#     for j in range(1, k+1):
#         for l in range(len(u_list)):
#             tuple_count = 0
#             for i in range(d):
#                 if(sub_range_list[j-1][0] < u_list[l][i] < sub_range_list[j-1][1]):
#                     tuple_count += 1
#                     print(u_list[l][i], sub_range_list[j-1])
#             if (tuple_count == d):
#                 print(tuple_count)
#                 mat[j-1][j-1] += 1
#     return mat

# rand_list = generate_random_num(20)
# u_list = []
# for i in range(0, 20, 2):
#     if (len(rand_list[i:i+2]) == 2):
#         u_list.append(rand_list[i:i+2])

# range_num_list = []
# sub_range_list = range_sub_interval(4)
# for i in range(10):
#     for j in range(2):
#         range_num_list.append(get_freq_range(u_list[i][j], sub_range_list))
#         print("range num for ", u_list[i][j])
# print(range_num_list, len(range_num_list))

# f_list = []

# for i in range(0, len(range_num_list)-2, 2):
#     print(range_num_list[i], range_num_list[i+1])

# for i in range(0, 20, 2):
#     if (len(range_num_list[i:i+2]) == 2):
#         f_list.append(range_num_list[i:i+2])

# print(f_list)
# print(demo(2, 4, u_list))
# print(u_list)
# get_sum_j(2, 4, range_sub_interval(4), u_list)

# lst = []
# sub_range_list = range_sub_interval(4)

# def count_range_in_list(li, min, max):
# 	ctr = 0
# 	for x in li:
# 		if min <= x <= max:
# 			ctr += 1
# 	return ctr

# mat = get_matrix(2, 4)

# for i in range(len(f_list)):
#     for j in range(1, len(f_list[i])):
#         print("tuple mat", f_list[i][j-1], f_list[i][j])
#         m = f_list[i][j-1]
#         n = f_list[i][j]
#         mat[m][n] += 1

# print(mat)

# for j in range(2):
#     for i in range(4):
#         lst.append(i)

# for i in range(4):
#     if(count_range_in_list(u_list[0], sub_range_list[i][0], sub_range_list[i][1])):
#         #lst.append(i)

# print(lst)