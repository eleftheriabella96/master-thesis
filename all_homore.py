from pyinform.shannon import entropy
import matplotlib.pyplot as plt
import pyhrv.time_domain as td
from random import randrange
from random import sample
import pandas as pd
import numpy as np 
import itertools
import pyinform
import random
import pyhrv
import sys
import os
import csv

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
def bubble_entropy(x, m=10):
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
def bubble_entropy_fast(x, m=10):
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

f = open("latex_data.tex", "w")


#ARRAYS TO STORE MAX VALUES
max_AVG=[]
max_RMSSD=[]
max_SDNN=[]
max_SDNNi=[]
max_PNN50=[]
max_SHANNON=[]
max_APEN=[]
max_SAMPEN=[]
max_BUBBLE=[]
#MAX VALUES
max_AVG_VALUE=0
max_RMSSD_VALUE=0
max_SDNN_VALUE=0
max_SDNNi_VALUE=0
max_PNN50_VALUE=0
max_SHANNON_VALUE=0
max_APEN_VALUE=0
max_SAMPEN_VALUE=0
max_BUBBLE_VALUE=0
#AVG VALUES FOR EACH USER
AVG_HR=0
AVG_RMSSD=0
AVG_SDNN=0
AVG_SDNNi=0
AVG_PNN50=0
AVG_SHANNON=0
AVG_APEN=0
AVG_SAMPEN=0
AVG_BUBBLE=0


################## 1) NORMAL RR INTERVALS #############################
user = sys.argv[1]	#User
#file_name = sys.argv[2]	#HR file

window_size = 60
#users = [1, 2, 3, 4, 5, 6, 7]
#names = ['Bro', 'Euh', 'Fanis_Balaskas','Geo', 'Greg', 'Ted', 'Vagelis']
experiments = ['zeros', 'higher', 'A_B+C', 'A_x_B_C', 'A_x_B-x_C', 'A_x_C']
colors = ['palevioletred', 'powderblue', 'salmon', 'springgreen', 'tomato', 'greenyellow', 'teal']
window = [0]*window_size

#hr = pd.read_csv(str(os.path.dirname(os.path.abspath(__file__))) +"/"+ str(user) +'_HeartRate.csv', delimiter =',')

with open('RR_differences.txt') as file:#read RR differences from file
    heartrate = [line.rstrip() for line in file]
    
#heartrate = hr['heartrate'].values.tolist()
#heartrate_sec = heartrate

#for i in range(len(heartrate)):	    #convert bpm to msec
	#if heartrate[i] != 0:
		#heartrate[i] = int(60000/heartrate[i])  # 60,000/bpm = msec
	#else:
		#heartrate[i] = 0


amount_of_windows = int(len(heartrate)/window_size)
RMSSD_array = []
STD_array = []
AVG_array = []
SDNN_array = []
SDNNi_array = []
pnn50_array = []
Shannon_array = []
approximate_array = []
sample_array = []
bubble_array = []

for i in range(0, amount_of_windows):
	window = heartrate[window_size*i:window_size*(i+1)]	
	RMSSD = np.sqrt(np.mean(np.square(np.diff(window))))  #RMSSD 
	RMSSD_array.append(RMSSD)
	STD = np.std(window)							#STD
	STD_array.append(STD)
	AVG = average(window)							#AVG of window
	AVG_array.append(AVG) 
	SDNN = td.sdnn(window)
	SDNN_array.append(SDNN[0])				#SDNN 							
	#SDNNi = td.sdann(window)					#SDNNi
	#SDNNi_array.append(np.nanmean(SDNN[0]))
	SDNNi = td.sdnn_index(window)					#SDNNi
	SDNNi_array.append(SDNNi['sdnn_index'])
	pnn50 = td.nn50(window)
	pnn50_array.append(pnn50[0])			#pnn50
	Shannon = shannon(window)					#Shannon entropy
	Shannon_array.append(Shannon)
	approximate = approximate_entropy_bucket(window,m=2,r=0.2,rsplit=5)#approximate entropy
	approximate_array.append(approximate)
	sample = sample_entropy_bucket(window,m=2,r=0.2,rsplit=5)#sample entropy
	sample_array.append(sample)
	bubble = bubble_entropy_fast(window,m=10)		#bubble entropy
	bubble_array.append(bubble)
	
AVG_HR = round(sum(AVG_array)/ len(AVG_array), 2)
AVG_SDNN = round(sum(SDNN_array)/ len(SDNN_array), 2)
AVG_RMSSD = round(sum(RMSSD_array)/ len(RMSSD_array), 2)
AVG_SDNNi = round(sum(SDNNi_array)/ len(SDNNi_array), 2)
AVG_PNN50 = round(sum(pnn50)/ len(pnn50), 2)
AVG_SHANNON = round(sum(Shannon_array)/ len(Shannon_array), 2)
AVG_APEN = round(sum(approximate_array)/ len(approximate_array), 2)
AVG_SAMPEN = round(sum(sample_array)/ len(sample_array), 2)
AVG_BUBBLE = round(sum(bubble_array)/ len(bubble_array), 2)


f.write("\\newcommand{\\meanmean}{"+str(AVG_HR)+"}" + "\n")
f.write("\\newcommand{\\meansdnn}{"+str(AVG_SDNN)+"}" + "\n")
#f.write("\\newcommand{\\AVG_SDNNi}{"+str(AVG_SDNNi)+"}" + "\n")
f.write("\\newcommand{\\meanrmssd}{"+str(AVG_RMSSD)+"}" + "\n")
f.write("\\newcommand{\\meanpnn}{"+str(AVG_PNN50)+"}" + "\n")
f.write("\\newcommand{\\meanShannon}{"+str(AVG_SHANNON)+"}" + "\n")
f.write("\\newcommand{\\meanApEn}{"+str(AVG_APEN)+"}" + "\n")
f.write("\\newcommand{\\meanSampEn}{"+str(AVG_SAMPEN)+"}" + "\n")
f.write("\\newcommand{\\meanBubble}{"+str(AVG_BUBBLE)+"}" + "\n")

################## 2) ZERO RR INTERVALS #############################
window = [0]*window_size

heartrate = hr['heartrate'].values.tolist()
heartrate_sec = heartrate


for i in range(len(heartrate)):	    #convert bpm to msec
	if heartrate[i] != 0:
		heartrate[i] = int(60000/heartrate[i])  # 60,000/bpm = msec
	else:
		heartrate[i] = 0

amount_of_windows = int(len(heartrate)/window_size)
RMSSD_array = []
AVG_array = []
SDNN_array = []
SDNNi_array = []
pnn50_array = []
Shannon_array = []
approximate_array = []
sample_array = []
bubble_array = []


#first 1/3 of the windows: 20 random values become 0 (up to)
for i in range(0, int(amount_of_windows/3)):
	zeros = heartrate[i*window_size:(i+1)*window_size]
	for i in range(0, 20):
		r = random.randrange(0, 60)
		zeros[r] = 0
	heartrate[i*window_size:i*window_size+60] = zeros

for i in range(0, amount_of_windows):
	window = heartrate[window_size*i:window_size*(i+1)]	
	window_sec = heartrate_sec[window_size*i:window_size*(i+1)]	
	RMSSD = np.sqrt(np.mean(np.square(np.diff(window))))  #RMSSD 
	RMSSD_array.append(RMSSD)
	AVG = average(window)																	#AVG of window
	AVG_array.append(AVG) 
	SDNN = td.sdnn(window)																#SDNN 
	SDNN_array.append(SDNN[0])															
	SDNNi = td.sdnn_index(window)					#SDNNi
	SDNNi_array.append(SDNNi['sdnn_index'])		
	pnn50 = td.nn50(window)
	pnn50_array.append(pnn50[0])			#pnn50											
	Shannon = shannon(window)															#Shannon entropy
	Shannon_array.append(Shannon)
	approximate = approximate_entropy_bucket(window,m=2,r=0.2,rsplit=5)#approximate entropy
	approximate_array.append(approximate)
	sample = sample_entropy_bucket(window,m=2,r=0.2,rsplit=5)#sample entropy
	sample_array.append(sample)
	bubble = bubble_entropy_fast(window,m=10)							#bubble entropy
	bubble_array.append(bubble)

max_AVG.append(max(AVG_array))
max_RMSSD.append(max(RMSSD_array))
max_SDNN.append(max(SDNN_array))
max_SDNNi.append(max(SDNNi_array))
max_PNN50.append(max(pnn50_array))
max_SHANNON.append(max(Shannon_array))
max_APEN.append(max(approximate_array))
max_SAMPEN.append(max(sample_array))
max_BUBBLE.append(max(bubble_array))

#HEART RATE PLOT
plt.figure(1)
x = [i for i in range(1, len(heartrate)+1)]
y = heartrate

#window with highest RMSSD value
RMSSD_index = RMSSD_array.index(max(RMSSD_array))#index window of max RMSSD value
x1 = [i for i in range(RMSSD_index*window_size,(RMSSD_index+1)*window_size)]
y1 = heartrate[RMSSD_index*window_size:(RMSSD_index+1)*window_size]#window of max RMSSD value

#window with highest ApEn value
APEN_index = approximate_array.index(max(approximate_array))#index window of max ApEn value
x2 = [i for i in range(APEN_index*window_size,(APEN_index+1)*window_size)]
y2 = heartrate[APEN_index*window_size:(APEN_index+1)*window_size]#window of max ApEn value


plt.title("HR PLOT")
plt.xlabel("Sample number")
plt.ylabel("RR intervals [ms]")
plt.plot(x, y, color = 'blue', linewidth = 0.5, zorder = 1)
plt.plot(x1, y1, color = 'red', linewidth = 0.5, zorder = 2)
plt.plot(x2, y2, color = 'red', linewidth = 0.5, zorder = 3)
figurename1 = "Usr" + str(user)+"0.jpeg"
plt.savefig(figurename1)
#plt.show()

################## 3) HIGHER RR INTERVALS #############################
window = [0]*window_size

heartrate = hr['heartrate'].values.tolist()
heartrate_sec = heartrate

for i in range(len(heartrate)):	    #convert bpm to msec
	if heartrate[i] != 0:
		heartrate[i] = int(60000/heartrate[i])  # 60,000/bpm = msec
	else:
		heartrate[i] = 0

amount_of_windows = int(len(heartrate)/window_size)
RMSSD_array = []
STD_array = []
AVG_array = []
SDNN_array = []
SDNNi_array = []
pnn50_array = []
Shannon_array = []
approximate_array = []
sample_array = []
bubble_array = []
	
for i in range(0, int(amount_of_windows/2)):
	zeros = []
	zeros = heartrate[i*window_size:(i+1)*window_size]
	zeros =  [element * 2 for element in zeros]
	heartrate[i*window_size:(i+1)*window_size] = zeros
	
for i in range(0, amount_of_windows):
	window = heartrate[window_size*i:window_size*(i+1)]	
	RMSSD = np.sqrt(np.mean(np.square(np.diff(window))))  #RMSSD 
	RMSSD_array.append(RMSSD)
	AVG = average(window)																	#AVG of window
	AVG_array.append(AVG) 
	SDNN = td.sdnn(window)																#SDNN 
	SDNN_array.append(SDNN[0])															
	SDNNi = td.sdnn_index(window)					#SDNNi
	SDNNi_array.append(SDNNi['sdnn_index'])
	pnn50 = td.nn50(window)
	pnn50_array.append(pnn50[0])			#pnn50												
	Shannon = shannon(window)															#Shannon entropy
	Shannon_array.append(Shannon)
	approximate = approximate_entropy_bucket(window,m=2,r=0.2,rsplit=5)#approximate entropy
	approximate_array.append(approximate)
	sample = sample_entropy_bucket(window,m=2,r=0.2,rsplit=5)#sample entropy
	sample_array.append(sample)
	bubble = bubble_entropy_fast(window,m=10)							#bubble entropy
	bubble_array.append(bubble)
		
max_AVG.append(max(AVG_array))
max_RMSSD.append(max(RMSSD_array))
max_SDNN.append(max(SDNN_array))
max_SDNNi.append(max(SDNNi_array))
max_PNN50.append(max(pnn50_array))
max_SHANNON.append(max(Shannon_array))
max_APEN.append(max(approximate_array))
max_SAMPEN.append(max(sample_array))
max_BUBBLE.append(max(bubble_array))


#HEART RATE PLOT
plt.figure(2)
x = [i for i in range(1, len(heartrate)+1)]
y = heartrate

#window with highest RMSSD value
RMSSD_index = RMSSD_array.index(max(RMSSD_array))#index window of max RMSSD value
x1 = [i for i in range(RMSSD_index*window_size,(RMSSD_index+1)*window_size)]
y1 = heartrate[RMSSD_index*window_size:(RMSSD_index+1)*window_size]#window of max RMSSD value

#window with highest ApEn value
APEN_index = approximate_array.index(max(approximate_array))#index window of max ApEn value
x2 = [i for i in range(APEN_index*window_size,(APEN_index+1)*window_size)]
y2 = heartrate[APEN_index*window_size:(APEN_index+1)*window_size]#window of max ApEn value


plt.title("HR PLOT")
plt.xlabel("Sample number")
plt.ylabel("RR intervals [ms]")
plt.plot(x, y, color = 'blue', linewidth = 0.5, zorder = 1)
plt.plot(x1, y1, color = 'red', linewidth = 0.5, zorder = 2)
plt.plot(x2, y2, color = 'red', linewidth = 0.5, zorder = 3)
figurename2 = "Usr" + str(user)+"1.jpeg"
plt.savefig(figurename2)
#plt.show()
	
	
################## 4) A_B+C #############################
heartrate = hr['heartrate'].values.tolist()

for i in range(len(heartrate)):	    #convert bpm to msec
	if heartrate[i] != 0:
		heartrate[i] = int(60000/heartrate[i])  # 60,000/bpm = msec
	else:
		heartrate[i] = 0

amount_of_windows = int(len(heartrate)/window_size)
RMSSD_array = []
STD_array = []
AVG_array = []
SDNN_array = []
SDNNi_array = []
pnn50_array = []
Shannon_array = []
approximate_array = []
sample_array = []
bubble_array = []

for i in range(0, 5*60, 20):	#one every 20 seconds for the first 5 finutes
	heartrate[i] += heartrate[i+1]
	heartrate = [heartrate[i] for i in range(len(heartrate)) if i != i+1]
		
for i in range(0, amount_of_windows):
	window = heartrate[window_size*i:window_size*(i+1)]	
	RMSSD = np.sqrt(np.mean(np.square(np.diff(window))))  #RMSSD 
	RMSSD_array.append(RMSSD)
	AVG = average(window)																	#AVG of window
	AVG_array.append(AVG) 
	SDNN = td.sdnn(window)																#SDNN 
	SDNN_array.append(SDNN[0])															
	SDNNi = td.sdnn_index(window)					#SDNNi
	SDNNi_array.append(SDNNi['sdnn_index'])									
	pnn50 = td.nn50(window)
	pnn50_array.append(pnn50[0])													#pnn50			
	Shannon = shannon(window)															#Shannon entropy
	Shannon_array.append(Shannon)
	approximate = approximate_entropy_bucket(window,m=2,r=0.2,rsplit=5)#approximate entropy
	approximate_array.append(approximate)
	sample = sample_entropy_bucket(window,m=2,r=0.2,rsplit=5)#sample entropy
	sample_array.append(sample)
	bubble = bubble_entropy_fast(window,m=10)							#bubble entropy
	bubble_array.append(bubble)

max_AVG.append(max(AVG_array))		
max_RMSSD.append(max(RMSSD_array))
max_SDNN.append(max(SDNN_array))
max_SDNNi.append(max(SDNNi_array))
max_PNN50.append(max(pnn50_array))
max_SHANNON.append(max(Shannon_array))
max_APEN.append(max(approximate_array))
max_SAMPEN.append(max(sample_array))
max_BUBBLE.append(max(bubble_array))


#HEART RATE PLOT
plt.figure(3)
x = [i for i in range(1, len(heartrate)+1)]
y = heartrate

#window with highest RMSSD value
RMSSD_index = RMSSD_array.index(max(RMSSD_array))#index window of max RMSSD value
x1 = [i for i in range(RMSSD_index*window_size,(RMSSD_index+1)*window_size)]
y1 = heartrate[RMSSD_index*window_size:(RMSSD_index+1)*window_size]#window of max RMSSD value

#window with highest ApEn value
APEN_index = approximate_array.index(max(approximate_array))#index window of max ApEn value
x2 = [i for i in range(APEN_index*window_size,(APEN_index+1)*window_size)]
y2 = heartrate[APEN_index*window_size:(APEN_index+1)*window_size]#window of max ApEn value


plt.title("HR PLOT")
plt.xlabel("Sample number")
plt.ylabel("RR intervals [ms]")
plt.plot(x, y, color = 'blue', linewidth = 0.5, zorder = 1)
plt.plot(x1, y1, color = 'red', linewidth = 0.5, zorder = 2)
plt.plot(x2, y2, color = 'red', linewidth = 0.5, zorder = 3)
figurename3 = "Usr" + str(user)+"2.jpeg"
plt.savefig(figurename3)
#plt.show()

################## 5) A_x_B_C #############################
heartrate = hr['heartrate'].values.tolist()

for i in range(len(heartrate)):	    #convert bpm to msec
	if heartrate[i] != 0:
		heartrate[i] = int(60000/heartrate[i])  # 60,000/bpm = msec
	else:
		heartrate[i] = 0

amount_of_windows = int(len(heartrate)/window_size)
RMSSD_array = []
STD_array = []
AVG_array = []
SDNN_array = []
SDNNi_array = []
pnn50_array = []
Shannon_array = []
approximate_array = []
sample_array = []
bubble_array = []
	
#print(len(heartrate))
for i in range(0, 5*60, 5):#one every five seconds for the first 5 finutes
	heartrate.insert(i, randrange(5)*0.1)
#print(len(heartrate))

for i in range(0, amount_of_windows):
	window = heartrate[window_size*i:window_size*(i+1)]	
	RMSSD = np.sqrt(np.mean(np.square(np.diff(window))))  #RMSSD 
	RMSSD_array.append(RMSSD)
	AVG = average(window)																	#AVG of window
	AVG_array.append(AVG) 
	SDNN = td.sdnn(window)																#SDNN 
	SDNN_array.append(SDNN[0])														
	pnn50 = td.nn50(window)
	pnn50_array.append(pnn50[0])			#pnn50													
	SDNNi = td.sdnn_index(window)					#SDNNi
	SDNNi_array.append(SDNNi['sdnn_index'])													
	Shannon = shannon(window)															#Shannon entropy
	Shannon_array.append(Shannon)
	approximate = approximate_entropy_bucket(window,m=2,r=0.2,rsplit=5)#approximate entropy
	approximate_array.append(approximate)
	sample = sample_entropy_bucket(window,m=2,r=0.2,rsplit=5)#sample entropy
	sample_array.append(sample)
	bubble = bubble_entropy_fast(window,m=10)							#bubble entropy
	bubble_array.append(bubble)
		
max_AVG.append(max(AVG_array))
max_RMSSD.append(max(RMSSD_array))
max_SDNN.append(max(SDNN_array))
max_SDNNi.append(max(SDNNi_array))
max_PNN50.append(max(pnn50_array))
max_SHANNON.append(max(Shannon_array))
max_APEN.append(max(approximate_array))
max_SAMPEN.append(max(sample_array))
max_BUBBLE.append(max(bubble_array))

#HEART RATE PLOT
plt.figure(4)
x = [i for i in range(1, len(heartrate)+1)]
y = heartrate

#window with highest RMSSD value
RMSSD_index = RMSSD_array.index(max(RMSSD_array))#index window of max RMSSD value
x1 = [i for i in range(RMSSD_index*window_size,(RMSSD_index+1)*window_size)]
y1 = heartrate[RMSSD_index*window_size:(RMSSD_index+1)*window_size]#window of max RMSSD value

#window with highest ApEn value
APEN_index = approximate_array.index(max(approximate_array))#index window of max ApEn value
x2 = [i for i in range(APEN_index*window_size,(APEN_index+1)*window_size)]
y2 = heartrate[APEN_index*window_size:(APEN_index+1)*window_size]#window of max ApEn value


plt.title("HR PLOT")
plt.xlabel("Sample number")
plt.ylabel("RR intervals [ms]")
plt.plot(x, y, color = 'blue', linewidth = 0.5, zorder = 1)
plt.plot(x1, y1, color = 'red', linewidth = 0.5, zorder = 2)
plt.plot(x2, y2, color = 'red', linewidth = 0.5, zorder = 3)
figurename4 = "Usr" + str(user)+"3.jpeg"
plt.savefig(figurename4)
#plt.show()

################## 6) A_x_B-x_C #############################
heartrate = hr['heartrate'].values.tolist()

for i in range(len(heartrate)):	    #convert bpm to msec
	if heartrate[i] != 0:
		heartrate[i] = int(60000/heartrate[i])  # 60,000/bpm = msec
	else:
		heartrate[i] = 0

amount_of_windows = int(len(heartrate)/window_size)
RMSSD_array = []
STD_array = []
AVG_array = []
SDNN_array = []
SDNNi_array = []
pnn50_array = []
Shannon_array = []
approximate_array = []
sample_array = []
bubble_array = []

#print(len(heartrate))
for i in range(0, 5*60, 5):#one every five seconds for the first 5 finutes
	x = randrange(5)*0.1
	heartrate.insert(i, x)
	heartrate[i+1] -= x
#print(len(heartrate))

for i in range(0, amount_of_windows):
	window = heartrate[window_size*i:window_size*(i+1)]	
	RMSSD = np.sqrt(np.mean(np.square(np.diff(window))))  #RMSSD 
	RMSSD_array.append(RMSSD)
	AVG = average(window)																	#AVG of window
	AVG_array.append(AVG) 
	SDNN = td.sdnn(window)																#SDNN 
	SDNN_array.append(SDNN[0])														
	SDNNi = td.sdnn_index(window)					#SDNNi
	SDNNi_array.append(SDNNi['sdnn_index'])
	pnn50 = td.nn50(window)
	pnn50_array.append(pnn50[0])			#pnn50													
	Shannon = shannon(window)															#Shannon entropy
	Shannon_array.append(Shannon)
	approximate = approximate_entropy_bucket(window,m=2,r=0.2,rsplit=5)#approximate entropy
	approximate_array.append(approximate)
	sample = sample_entropy_bucket(window,m=2,r=0.2,rsplit=5)#sample entropy
	sample_array.append(sample)
	bubble = bubble_entropy_fast(window,m=10)							#bubble entropy
	bubble_array.append(bubble)
		
max_AVG.append(max(AVG_array))
max_RMSSD.append(max(RMSSD_array))
max_SDNN.append(max(SDNN_array))
max_SDNNi.append(max(SDNNi_array))
max_PNN50.append(max(pnn50_array))
max_SHANNON.append(max(Shannon_array))
max_APEN.append(max(approximate_array))
max_SAMPEN.append(max(sample_array))
max_BUBBLE.append(max(bubble_array))

#HEART RATE PLOT
plt.figure(5)
x = [i for i in range(1, len(heartrate)+1)]
y = heartrate

#window with highest RMSSD value
RMSSD_index = RMSSD_array.index(max(RMSSD_array))#index window of max RMSSD value
x1 = [i for i in range(RMSSD_index*window_size,(RMSSD_index+1)*window_size)]
y1 = heartrate[RMSSD_index*window_size:(RMSSD_index+1)*window_size]#window of max RMSSD value

#window with highest ApEn value
APEN_index = approximate_array.index(max(approximate_array))#index window of max ApEn value
x2 = [i for i in range(APEN_index*window_size,(APEN_index+1)*window_size)]
y2 = heartrate[APEN_index*window_size:(APEN_index+1)*window_size]#window of max ApEn value


plt.title("HR PLOT")
plt.xlabel("Sample number")
plt.ylabel("RR intervals [ms]")
plt.plot(x, y, color = 'blue', linewidth = 0.5, zorder = 1)
plt.plot(x1, y1, color = 'red', linewidth = 0.5, zorder = 2)
plt.plot(x2, y2, color = 'red', linewidth = 0.5, zorder = 3)
figurename5 = "Usr" + str(user)+"4.jpeg"
plt.savefig(figurename5)
#plt.show()


################## 7) A_x_C #############################
heartrate = hr['heartrate'].values.tolist()

for i in range(len(heartrate)):	    #convert bpm to msec
	if heartrate[i] != 0:
		heartrate[i] = int(60000/heartrate[i])  # 60,000/bpm = msec
	else:
		heartrate[i] = 0

amount_of_windows = int(len(heartrate)/window_size)
RMSSD_array = []
STD_array = []
AVG_array = []
SDNN_array = []
SDNNi_array = []
pnn50_array = []
Shannon_array = []
approximate_array = []
sample_array = []
bubble_array = []
	
for i in range(0, 5*60, 10):#one every 10 seconds for the first five finutes
	heartrate[i] /= 10
		
for i in range(0, amount_of_windows):
	window = heartrate[window_size*i:window_size*(i+1)]	
	RMSSD = np.sqrt(np.mean(np.square(np.diff(window))))  #RMSSD 
	RMSSD_array.append(RMSSD)
	AVG = average(window)																	#AVG of window
	AVG_array.append(AVG) 
	SDNN = td.sdnn(window)																#SDNN 
	SDNN_array.append(SDNN[0])															
	SDNNi = td.sdnn_index(window)					#SDNNi
	SDNNi_array.append(SDNNi['sdnn_index'])
	pnn50 = td.nn50(window)
	pnn50_array.append(pnn50[0])			#pnn50													
	Shannon = shannon(window)															#Shannon entropy
	Shannon_array.append(Shannon)
	approximate = approximate_entropy_bucket(window,m=2,r=0.2,rsplit=5)#approximate entropy
	approximate_array.append(approximate)
	sample = sample_entropy_bucket(window,m=2,r=0.2,rsplit=5)#sample entropy
	sample_array.append(sample)
	bubble = bubble_entropy_fast(window,m=10)							#bubble entropy
	bubble_array.append(bubble)


max_AVG.append(max(AVG_array))
max_RMSSD.append(max(RMSSD_array))
max_SDNN.append(max(SDNN_array))
max_SDNNi.append(max(SDNNi_array))
max_PNN50.append(max(pnn50_array))
max_SHANNON.append(max(Shannon_array))
max_APEN.append(max(approximate_array))
max_SAMPEN.append(max(sample_array))
max_BUBBLE.append(max(bubble_array))

#HEART RATE PLOT
plt.figure(6)
x = [i for i in range(1, len(heartrate)+1)]
y = heartrate

#window with highest RMSSD value
RMSSD_index = RMSSD_array.index(max(RMSSD_array))#index window of max RMSSD value
x1 = [i for i in range(RMSSD_index*window_size,(RMSSD_index+1)*window_size)]
y1 = heartrate[RMSSD_index*window_size:(RMSSD_index+1)*window_size]#window of max RMSSD value

#window with highest ApEn value
APEN_index = approximate_array.index(max(approximate_array))#index window of max ApEn value
x2 = [i for i in range(APEN_index*window_size,(APEN_index+1)*window_size)]
y2 = heartrate[APEN_index*window_size:(APEN_index+1)*window_size]#window of max ApEn value


plt.title("HR PLOT")
plt.xlabel("Sample number")
plt.ylabel("RR intervals [ms]")
plt.plot(x, y, color = 'blue', linewidth = 0.5, zorder = 1)
plt.plot(x1, y1, color = 'red', linewidth = 0.5, zorder = 2)
plt.plot(x2, y2, color = 'red', linewidth = 0.5, zorder = 3)
figurename6 = "Usr" + str(user)+"5.jpeg"
plt.savefig(figurename6)
#plt.show()


#FIND MAX VALUE (AND ITS INDEX) FOR EACH METRIC
max_AVG_VALUE = round(max(max_AVG), 2)
max_AVG_VALUE_INDEX = max_AVG.index(max(max_AVG))

max_RMSSD_VALUE = round(max(max_RMSSD), 2)
max_RMSSD_VALUE_INDEX = max_RMSSD.index(max(max_RMSSD))

max_SDNN_VALUE = round(max(max_SDNN), 2)
max_SDNN_VALUE_INDEX = max_SDNN.index(max(max_SDNN))

max_SDNNi_VALUE = round(max(max_SDNNi), 2)
max_SDNNi_VALUE_INDEX = max_SDNNi.index(max(max_SDNNi))

max_PNN50_VALUE = round(max(max_PNN50), 2)
max_PNN50_VALUE_INDEX = max_PNN50.index(max(max_PNN50))

max_SHANNON_VALUE = round(max(max_SHANNON), 2)
max_SHANNON_VALUE_INDEX = max_SHANNON.index(max(max_SHANNON))

max_APEN_VALUE = round(max(max_APEN), 2)
max_APEN_VALUE_INDEX = max_APEN.index(max(max_APEN))

max_SAMPEN_VALUE = round(max(max_SAMPEN), 2)
max_SAMPEN_VALUE_INDEX = max_SAMPEN.index(max(max_SAMPEN))

max_BUBBLE_VALUE = round(max(max_BUBBLE), 2)
max_BUBBLE_VALUE_INDEX = max_BUBBLE.index(max(max_BUBBLE))


f.write("\\newcommand{\\mean}{"+str(max_AVG_VALUE)+"}" + "\n")
f.write("\\newcommand{\\rmssd}{"+str(max_RMSSD_VALUE)+"}" + "\n")
f.write("\\newcommand{\\sdnn}{"+str(max_SDNN_VALUE)+"}" + "\n")
#f.write("\\newcommand{\\SDNNi}{"+str(max_SDNNi_VALUE)+"}" + "\n")
f.write("\\newcommand{\\pnn}{"+str(max_PNN50_VALUE)+"}" + "\n")
f.write("\\newcommand{\\Shannon}{"+str(max_SHANNON_VALUE)+"}" + "\n")
f.write("\\newcommand{\\ApEn}{"+str(max_APEN_VALUE)+"}" + "\n")
f.write("\\newcommand{\\SampEn}{"+str(max_SAMPEN_VALUE)+"}" + "\n")
f.write("\\newcommand{\\Bubble}{"+str(max_BUBBLE_VALUE)+"}" + "\n")


#figurename = "Usr"+ str(user)+str(max_AVG_VALUE_INDEX)+".jpeg"
f.write("\\newcommand{\\first}{"+figurename1+"}" + "\n")
f.write("\\newcommand{\\second}{"+figurename2+"}" + "\n")
f.write("\\newcommand{\\third}{"+figurename3+"}" + "\n")
f.write("\\newcommand{\\fourth}{"+figurename4+"}" + "\n")
f.write("\\newcommand{\\fifth}{"+figurename5+"}" + "\n")
f.write("\\newcommand{\\sixth}{"+figurename6+"}" + "\n")

f.close()
#os.system('xelatex document.tex')



