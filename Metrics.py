from pyinform.shannon import entropy
import matplotlib.pyplot as plt
import pyhrv.time_domain as td
from random import randrange
from random import sample
import pandas as pd
import numpy as np 
import statistics
import itertools
import pyinform
import random
import pyhrv
import sys
import os
import csv
import pywt

heartrate = []		#list of HR values
counter = 0
index = 0
##########################################################################
def embed(x,m):
    '''
    embeds a signal into dimension m
    '''
    X=[]
    for i in range(len(x)-m+1):
        t=[]
        for j in range(i,i+m):
            t.append(x[j])
        X.append(numpy.array(t))
    return X
##########################################################################
def norm(p1,p2,r):
    '''
    checks if p1 is similar to p2
    '''
    for i,j in zip(p1,p2):
        if numpy.abs(i-j)>r:
            return 0
    return 1
##########################################################################
def fm(timeseries,m,r):
    '''
    entropy in dimension m
    '''
    r=r*numpy.std(timeseries)
    X=embed(timeseries,m)
    N=len(X)
    cm = [1]*N # 1 is for self-matching

    for i,p in enumerate (itertools.permutations(X,2),0):
        cm[i//(N-1)] += norm(p[0],p[1],r)

    fm = 0
    for x in cm:
        fm += numpy.log(x/N)

    return fm/N
##########################################################################
def approximate_entropy_bucket(x ,m=2, r=0.2, rsplit=5):
    import ctypes
    lib = ctypes.cdll.LoadLibrary('./approximate_entropy_bucket_lib.so')

    N = len(x)
    array_type = ctypes.c_double * N
    lib.bucket.argtypes = [ctypes.POINTER(ctypes.c_double),
                           ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_double,
                           ctypes.c_int,
                           ]
    lib.bucket.restype = ctypes.c_double
    return lib.bucket(array_type(*x),N,m,r,rsplit)
##########################################################################
def approximate_entropy_straightforward(timeseries,m=2,r=0.2):

    '''
    a slow straightforward implementation of approximate entropy
    '''

    fm0 = fm(timeseries=timeseries,m=m,r=r)
    fm1 = fm(timeseries=timeseries,m=m+1,r=r)

    return fm0-fm1
##########################################################################
def approximate_entropy_bucket(x ,m=2, r=0.2, rsplit=5):

    import ctypes

    lib = ctypes.cdll.LoadLibrary('./approximate_entropy_bucket_lib.so')

    N = len(x)
    array_type = ctypes.c_double * N
    lib.bucket.argtypes = [ctypes.POINTER(ctypes.c_double),
                           ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_double,
                           ctypes.c_int,
                           ]
    lib.bucket.restype = ctypes.c_double
    return lib.bucket(array_type(*x),N,m,r,rsplit)
##########################################################################
def sample_entropy_bucket(x,m,r,rsplit):

    import ctypes

    #site_packages = site.getsitepackages()
    #suffix = sysconfig.get_config_var('EXT_SUFFIX')
    #lib_path = site_packages[0] + '/sample_entropy_bucket_lib'+suffix
    # print(lib_path)
    #lib = ctypes.cdll.LoadLibrary(lib_path)
    lib = ctypes.cdll.LoadLibrary('./sample_entropy_bucket_lib.so')

    N = len(x)
    array_type = ctypes.c_double * N
    lib.bucket.argtypes = [ctypes.POINTER(ctypes.c_double),
                           ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_double,
                           ctypes.c_int,
                           ]
    lib.bucket.restype = ctypes.c_double
    return lib.bucket(array_type(*x),N,m,r,rsplit)
#############################################################################
def entropy(x, order=1):
    """
    Shannon and Renyi entropy
    the size of the bin, bin size is set to 1
    values are supposed to be between -100 and 100  <-------
    :param x: the input series in beats/min
    :param order: order=1 is for Shannon, order>1 is for Renyi
    :return: metric
    """

  # check for valid parameters
    if not 1 <= order <=2:
        return 'error: 1<=order<=2'
    # initialization
    counter_p = [0] * 200
    n = 0
    # histogram
    for x in x:
        p = int(x) + 100
        counter_p[p] += 1
        n += 1
    # entropy computation
    if order == 1:
            r = 0
            for c in counter_p:
                    p = c / n
                    if p != 0:
                            r += p * log(p)
            result = -r
    if order == 2:
            r = 0
            for c in counter_p:
                    p = c / n
                    r += p ** 2
            result = -log(r)
    return result
#################################################
def embed(x, m):
    """
    embeds a signal into dimension m
    :param x: the input signal
    :type m: the dimension of the embedding space
    """
    X = []
    for i in range(len(x) - m + 1):
            t = []
            for j in range(i, i + m):
                    t.append(x[j])
            X.append(numpy.array(t))
    return X
##########################################################################
def similarity_check(p1, p2, r):
    """
    checks if p1 is similar to p2
    :param p1: point in m dim space
    :param p2: point in m dim space
    :param r: distance under which two points are coniderd as similar
    :return: 1 if p1 and p2 are similar, 0 othewise
    """
    for i, j in zip(p1, p2):
            if numpy.abs(i - j) > r:
                    return 0
    return 1
##########################################################################
def fm(x, m, r):
    """
    entropy in dimension m for given r
    :param x: the input signal
    :param m: the dimension of the embedding space
    :param r: distance under which two points are coniderd as similar
    :return: entropy
    """

    r = r * numpy.std(x)
    X = embed(x, m)

    N = len(X)
    cm = [1] * N  # 1 is for self-matching

    for i, p in enumerate(itertools.permutations(X, 2), 0):
            cm[i // (N - 1)] += similarity_check(p[0], p[1], r)

    entropy = 0
    for x in cm:
            entropy += numpy.log(x / N)

    return entropy / N


##########################################################################
def double_simiarity_check(p1, p2, r):
    """
    checks if p1 is similar to p2
    and also if p1[:-1] is similar to p2[:-1]
    :param p1: point in m dim space
    :param p2: point in m dim space
    :param r: distance under which two points are coniderd as similar
    :return: 1 or 0 for each case, depending on the similarity
    """

    for i in range(len(p1) - 1):
            if numpy.abs(p1[i] - p2[i]) > r:
                    return 0, 0
    if numpy.abs(p1[-1] - p2[-1]) > r:
            return 1, 0
    return 1, 1
#############################################################################
def bubble_count(x):
    """
    counts the number of swaps when sorting
    :param x: the input vector
    :return: the total number of swaps
    """
    y = 0
    for i in range(len(x) - 1, 0, -1):
            for j in range(i):
                    if x[j] > x[j + 1]:
                            x[j], x[j + 1] = x[j + 1], x[j]
                            y += 1
    return y
############################################
def complexity_count_fast(x, m):
    """
    :param x: the input series
    :param m: the dimension of the space
    :return: the series of complexities for total number of swaps
    """

    if len(x) < m:
            return []

    y = [bubble_count(x[:m])]
    v = sorted(x[:m])

    for i in range(m,len(x)):
            steps = y[i-m]
            steps -= v.index(x[i-m])
            v.pop(v.index(x[i-m]))
            v.append(x[i])
            j = m-1
            while j > 0 and v[j] < v[j-1]:
                    v[j], v[j-1] = v[j-1], v[j]
                    steps += 1
                    j -= 1
            y.append(steps)

    return y
############################################
def renyi_int(data):
    """
    returns renyi entropy (order 2) of an integer series and bin_size=1
    (specified for the needs of bubble entropy)
    :param data: the input series
    :return: metric
    """
    counter = [0] * (max(data) + 1)
    for x in data:
            counter[x] += 1
    r = 0
    for c in counter:
            p = c / len(data)
            r += p * p
    return -np.log(r)
########################################
def bubble_entropy(x, m=15):
    """
    computes bubble entropy following the definition
    :param x: the input signal
    :param m: the dimension of the embedding space
    :return: metric
    """
    X = embed(x, m)
    complexity = [bubble_count(v) for v in X]
    B = renyi_int(complexity)

    X = embed(x, m+1)
    complexity = [bubble_count(v) for v in X]
    A = renyi_int(complexity)

    return numpy.log((m+1)/(m-1)) * (A-B)
########################################
def bubble_entropy_fast(x, m=15):
    """
    computes bubble entropy following the definition
    :param x: the input signal
    :param m: the dimension of the embedding space
    :return: metric
    """
    complexity = complexity_count_fast(x, m)
    B = renyi_int(complexity)

    complexity = complexity_count_fast(x, m+1)
    A = renyi_int(complexity)

    return np.log((m+1)/(m-1)) * (A-B)
################################################################################
def average(l):
    return sum(l) / len(l)

def renyi(data):
	r = 0
	for c in data:
		p = c / len(data)
		r += p ** 2
	result = -np.log(r)
	return result

def shannon(data):
	r = 0
	for c in data:
		p = c / len(data)
		if p != 0:
			r += p * np.log(p)
	return -r
#############################################################################
def haarWavelet ( signal, level ):

    s = .5;                  # scaling -- try 1 or ( .5 ** .5 )
    h = [ 1,  1 ];           # lowpass filter
    g = [ 1, -1 ];           # highpass filter        
    f = len ( h );           # length of the filter

    t = signal;              # 'workspace' array
    l = len ( t );           # length of the current signal
    y = [0] * l;             # initialise output

    t = t + [ 0, 0 ];        # padding for the workspace

    for i in range ( level ):
        y [ 0:l ] = [0] * l; # initialise the next level 
        l2 = l // 2;         # half approximation, half detail
        for j in range ( l2 ):            
            for k in range ( f ):                
                y [j]    += t [ 2*j + k ] * h [ k ] * s;
                y [j+l2] += t [ 2*j + k ] * g [ k ] * s;

        l = l2;              # continue with the approximation
        t [ 0:l ] = y [ 0:l ] ;

    return y

#############################################################################
def save_var_latex(key, value):
    import csv
    import os

    dict_var = {}

    file_path = os.path.join(os.getcwd(), "latex_data.dat")

    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    dict_var[key] = value

    with open(file_path, "w") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")
#############################################################################
#find median element of list
def get_median(lst, index, n_prev_vals):
    medians = list()
    for i in range(index, 0, -1):
        if(lst[i] != 0):
            medians.append(lst[i])
        if(len(medians)==n_prev_vals):
            return statistics.median(medians)

    return statistics.median(lst)
################################################################################

#f_latex_data = open("latex_data.tex", "w")


#ARRAYS TO STORE MAX VALUES
max_AVG=[]
max_RMSSD=[]
max_SHANNON=[]
max_APEN=[]
max_SAMPEN=[]
max_BUBBLE=[]
#MAX VALUES
max_AVG_VALUE=0
max_RMSSD_VALUE=0
max_SHANNON_VALUE=0
max_APEN_VALUE=0
max_SAMPEN_VALUE=0
max_BUBBLE_VALUE=0
#AVG VALUES FOR EACH USER
AVG_HR=0
AVG_RMSSD=0
AVG_SHANNON=0
AVG_APEN=0
AVG_SAMPEN=0
AVG_BUBBLE=0


################## 1) NORMAL RR INTERVALS #############################
#number = sys.argv[1]
window_size = int(sys.argv[1])
window = [0]*window_size
library_name = 'cu-supraventricular-arrhythmia'
countlines = 0
tags = []

with open('RECORDS.txt') as fp:
	for line in fp:
		countlines += 1
		filename = line.strip()
		tags.append(line.strip())


for filename_record in tags:
	f_latex_data = open("latex_data" +"_" + str(filename_record) + ".tex", "w")
	with open('RR_differences' + str(filename_record) + '.txt') as file_RR:
		heartrate = [float(line.rstrip()) for line in file_RR]


	#filter out noise form initial heartrate
	heartrate_new = []#filtered values to be stored here
	prev_values_med = []
	n_prev_vals = 3
	thres = 0.25
	for i in range(0, len(heartrate)):
		prev_values_med = get_median(heartrate, i, n_prev_vals)	
		if(heartrate[i] > prev_values_med + prev_values_med * thres or heartrate[i] < prev_values_med - prev_values_med * thres ):
			heartrate_new.append(prev_values_med)
		else: 
			heartrate_new.append(heartrate[i])  

	amount_of_windows = int(len(heartrate_new)/window_size)
	RMSSD_array = []
	STD_array = []
	AVG_array = []
	Shannon_array = []
	approximate_array = []
	sample_array = []
	bubble_array = []
	haar_4 = []
	haar_6 = []
	haar_8 = []
	
	#Haar wavelets level 4, 6, 8
	haar_4 = pywt.wavedec(heartrate_new, 'haar', mode='zero', level=4)
	haar_4_list = [x.tolist() for x in haar_4]
	haar_4_flat = [item for haar_4_list in haar_4 for item in haar_4_list]
	haar_6 = pywt.wavedec(heartrate_new, 'haar', mode='zero', level=6)
	haar_6_list = [x.tolist() for x in haar_6]
	haar_6_flat = [item for haar_6_list in haar_6 for item in haar_6_list]
	haar_8 = pywt.wavedec(heartrate_new, 'haar', mode='zero', level=8)
	haar_8_list = [x.tolist() for x in haar_8]
	haar_8_flat = [item for haar_8_list in haar_8 for item in haar_8_list]

	for i in range(0, amount_of_windows):
		window = heartrate_new[window_size*i:window_size*(i+1)]	
		RMSSD = np.sqrt(np.mean(np.square(np.diff(window))))  #RMSSD 
		RMSSD_array.append(RMSSD)
		STD = np.std(window)							#STD
		STD_array.append(STD)
		AVG = average(window)							#AVG of window
		AVG_array.append(AVG) 
		Shannon = shannon(window)					#Shannon entropy
		Shannon_array.append(Shannon)
		approximate = approximate_entropy_bucket(window,m=2,r=0.2,rsplit=5)#approximate entropy
		approximate_array.append(approximate)
		sample = sample_entropy_bucket(window,m=2,r=0.2,rsplit=5)#sample entropy
		sample_array.append(sample)
		bubble = bubble_entropy_fast(window,m=15)		#bubble entropy
		bubble_array.append(bubble)
		
		
	AVG_HR = round(sum(AVG_array)/ len(AVG_array), 2)
	AVG_RMSSD = round(sum(RMSSD_array)/ len(RMSSD_array), 2)
	AVG_SHANNON = round(sum(Shannon_array)/ len(Shannon_array), 2)
	AVG_APEN = round(sum(approximate_array)/ len(approximate_array), 2)
	AVG_SAMPEN = round(sum(sample_array)/ len(sample_array), 2)
	AVG_BUBBLE = round(sum(bubble_array)/ len(bubble_array), 2)
	AVG_HAAR4 = round(sum(haar_4_flat)/ len(haar_4_flat), 2)
	AVG_HAAR6 = round(sum(haar_6_flat)/ len(haar_6_flat), 2)
	AVG_HAAR8 = round(sum(haar_8_flat)/ len(haar_8_flat), 2)
	


	#HEART RATE PLOT
	plt.figure()
	
	plt.yticks(fontsize=6)
	plt.xticks(fontsize=6)
	
	x = [i for i in range(1, len(heartrate_new)+1)]#length of samples/pulses
	y = heartrate_new

	rmssd_index = RMSSD_array.index(max(RMSSD_array))#index window of max RMSSD value
	x1 = [i for i in range(rmssd_index*window_size,(rmssd_index+1)*window_size)]
	y1 = heartrate_new[rmssd_index*window_size:(rmssd_index+1)*window_size]
	
	RMSSD_array.sort()
	
	rmssd_second_index = RMSSD_array.index(RMSSD_array[-2])
	x2 = [i for i in range(rmssd_second_index*window_size,(rmssd_second_index+1)*window_size)]
	y2 = heartrate_new[rmssd_second_index*window_size:(rmssd_second_index+1)*window_size]
	
	rmssd_third_index = bubble_array.index(bubble_array[-3])
	x3 = [i for i in range(rmssd_third_index*window_size,(rmssd_third_index+1)*window_size)]
	y3 = heartrate_new[rmssd_third_index*window_size:(rmssd_third_index+1)*window_size]

	bubble_index = bubble_array.index(max(bubble_array))#index window of max RMSSD value
	x1_ = [i for i in range(bubble_index*window_size,(bubble_index+1)*window_size)]
	y1_ = heartrate_new[bubble_index*window_size:(bubble_index+1)*window_size]
	
	bubble_array.sort()
	
	bubble_second_index = bubble_array.index(bubble_array[-2])
	x2_ = [i for i in range(bubble_second_index*window_size,(bubble_second_index+1)*window_size)]
	y2_ = heartrate_new[bubble_second_index*window_size:(bubble_second_index+1)*window_size]
	
	bubble_third_index = bubble_array.index(bubble_array[-3])
	x3_ = [i for i in range(bubble_third_index*window_size,(bubble_third_index+1)*window_size)]
	y3_ = heartrate_new[bubble_third_index*window_size:(bubble_third_index+1)*window_size]
			
	plt.title("HR PLOT")
	plt.xlabel("Duration (in seconds)")
	plt.ylabel("ECG (in mV)")
	plt.plot(x, y, color = 'blue', linewidth = 0.5, zorder = 1, label = 'Heart rate')
	
	plt.plot(x1, y1, color = 'red', linewidth = 0.5, zorder = 2, label = 'Detected areas')
	plt.plot(x2, y2, color = 'red', linewidth = 0.5, zorder = 2)
	plt.plot(x3, y3, color = 'red', linewidth = 0.5, zorder = 2)
	
	plt.plot(x1_, y1_, color = 'red', linewidth = 0.5, zorder = 2)
	plt.plot(x2_, y2_, color = 'red', linewidth = 0.5, zorder = 2)
	plt.plot(x3_, y3_, color = 'red', linewidth = 0.5, zorder = 2)
	
	plt.plot(haar_4_flat, color = 'green', linewidth = 0.5, zorder = 2, label = 'Haar scale=4')
	plt.plot(haar_6_flat, color = 'black', linewidth = 0.5, zorder = 2, label = 'Haar scale=6')
	plt.plot(haar_8_flat, color = 'orange', linewidth = 0.5, zorder = 2, label = 'Haar scale=8')
	
	plt.legend(loc = 'upper right')
	figurename =  library_name + "_" + str(filename_record) + "_" + str(window_size) + "sec.jpeg" 
	
	plt.savefig(figurename)
	


	f_latex_data.write("\\newcommand{\\mean}{"+str(max(AVG_array))+"}" + "\n")
	f_latex_data.write("\\newcommand{\\rmssd}{"+str(max(RMSSD_array))+"}" + "\n")
	f_latex_data.write("\\newcommand{\\Shannon}{"+str(max(Shannon_array))+"}" + "\n")
	f_latex_data.write("\\newcommand{\\ApEn}{"+str(max(approximate_array))+"}" + "\n")
	f_latex_data.write("\\newcommand{\\SampEn}{"+str(max(sample_array))+"}" + "\n")
	f_latex_data.write("\\newcommand{\\Bubble}{"+str(max(bubble_array))+"}" + "\n")

	f_latex_data.write("\\newcommand{\\meanmean}{"+str(AVG_HR)+"}" + "\n")
	f_latex_data.write("\\newcommand{\\meanrmssd}{"+str(AVG_RMSSD)+"}" + "\n")
	f_latex_data.write("\\newcommand{\\meanShannon}{"+str(AVG_SHANNON)+"}" + "\n")
	f_latex_data.write("\\newcommand{\\meanApEn}{"+str(AVG_APEN)+"}" + "\n")
	f_latex_data.write("\\newcommand{\\meanSampEn}{"+str(AVG_SAMPEN)+"}" + "\n")
	f_latex_data.write("\\newcommand{\\meanBubble}{"+str(AVG_BUBBLE)+"}" + "\n")
	f_latex_data.write("\\newcommand{\\meanHaar4}{"+str(AVG_HAAR4)+"}" + "\n")
	f_latex_data.write("\\newcommand{\\meanHaar4}{"+str(AVG_HAAR6)+"}" + "\n")
	f_latex_data.write("\\newcommand{\\meanHaar4}{"+str(AVG_HAAR8)+"}" + "\n")
	

	f_latex_data.write("\\newcommand{\\figurename}{"+figurename+"}" + "\n")
	f_latex_data.write("\\newcommand{\\file}{latex_data_"+ str(filename_record)+"}" + "\n")

	f_latex_data.close()
	file_RR.close()
	#os.system('xelatex document.tex')

