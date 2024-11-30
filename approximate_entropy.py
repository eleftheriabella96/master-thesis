
import itertools
import numpy
import sys
import time

import time
import random
from random import sample
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


def approximate_entropy_lightweight(x ,m=2, r=0.2):

    import ctypes

    lib = ctypes.cdll.LoadLibrary('./approximate_entropy_lightweight_lib.so')

    N = len(x)
    array_type = ctypes.c_double * N
    lib.lightweight.argtypes = [ctypes.POINTER(ctypes.c_double),
								ctypes.c_int,
								ctypes.c_int,
								ctypes.c_double
								]
    lib.lightweight.restype = ctypes.c_double
    return lib.lightweight(array_type(*x),N,m,r)


##########################################################################


def approximate_entropy(timeseries,m,r,algorithm,rsplit):

    if algorithm=='straightforward':
        return approximate_entropy_straightforward(timeseries,m,r)
    if algorithm=='bucket':
        return approximate_entropy_bucket(timeseries,m,r,rsplit)
    if algorithm=='lightweight':
        return approximate_entropy_lightweight(timeseries,m,r)


##########################################################################
##########################################################################
def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False

def main():

    n = int(sys.argv[1])
    file = sys.argv[2]
    limit  = sys.argv[3]
    not_normal_indexes = []
    random_sample_ = []
    random_indexes = []
    N_index = []
    filelines = 0
    normal = 0

    filenames = ['f1y01', 'f1y02', 'f1y03', 'f1y04', 'f1y05', 'f1y06', 'f1y07', 'f1y08', 'f1y09', 'f1y10']


    #for i in range(len(filenames)):
    for i in range(0, int(limit)):
        #print(i+1)
        #x = [line.rstrip('\n') for line in open(filenames[i] + '.txt')]
        x = [line.rstrip('\n') for line in open('f1y' + file + '.txt')]
        x1 = [line.split() for line in x]
        x2 = []
        N = []
        zipped = []

        for line in x1:
            filelines += 1
            N.append(line[1])
            x2.append(float(line[2]))
            if(line[1] != 'N'):
                normal += 1
        N[0] = 'N'

        for c in range(len(N)):
            if N[c] != 'N':
                not_normal_indexes.append(c)#indexes of not N beats

        sample_lengths_1 = [100, len(x2)//32, len(x2)//16, len(x2)//8]
        sample_lengths_2 = [len(x2)//4, len(x2)//2, len(x2)]


        for j in sample_lengths_1:
            random_sample = sample(x2, j)
            index = x2.index(random_sample[0])#index of start of random sample
            N_index = list(range(index, index + j))#indexes of random sample

            while(common_member(not_normal_indexes, N_index)):
                random_sample = sample(x2, j)
                index = x2.index(random_sample[0])#index of start of random sample
                N_index = list(range(index, index + j))#indexes of random sample

            #copy = random_sample
            zero_ectopic.write(str(approximate_entropy(random_sample, n, 0.2, 'bucket', rsplit=5)) + '\n')


			#2 consecutive
            random_index = random.randint(0, len(random_sample)-2)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*2

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 2] = consecutives
            two_cons_ectopic.write(str(approximate_entropy(random_sample_, n, 0.2, 'bucket', rsplit=5)) + '\n')


			#5 consecutive
            random_index = random.randint(0, len(random_sample)-5)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*5

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 5] = consecutives
            five_cons_ectopic.write(str(approximate_entropy(random_sample_, n, 0.2, 'bucket', rsplit=5)) + '\n')


            #10 consecutive
            '''for s in range(0, 3):
                random_index = random.randint(0, len(random_sample)-2)
                x_RR = round(random.uniform(0.1, 0.9), 5)
                random_sample_ = random_sample[:random_index]
                random_sample_.append(x_RR)
                random_sample_ = random_sample_ + random_sample[random_index:]
                #final = apply_ectopic(random_sample)

            #ten_cons_ectopic.write('sample length ' + str(j) + '\n')
'''

            random_index = random.randint(0, len(random_sample)-10)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*10

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 10] = consecutives
            ten_cons_ectopic.write(str(approximate_entropy(random_sample_, n, 0.2, 'bucket', rsplit=5)) + '\n')


			#tenpercent
            size_ = j//10
            random_index = random.randint(0, len(random_sample)- j//10)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]* int(j//10)

            random_sample_ = random_sample
            random_sample_[random_index:random_index + j//10] = consecutives
            #tenpercent_cons_ectopic.write('sample length ' + str(j) + '\n')
            tenpercent_cons_ectopic.write(str(approximate_entropy(random_sample_, n, 0.2, 'bucket', rsplit=5)) + '\n')


        for j in sample_lengths_2:
            random_sample_ = sample(x2, j)
            index = x2.index(random_sample_[0])#index of start of random sample
            N_index = list(range(index, index + j))#indexes of random sample
            to_be_deleted = list(set(not_normal_indexes) & set(N_index))

            random_sample = [i for i in random_sample_ if i not in to_be_deleted]



			#no skipped beats
            #a0.write('sample length ' + str(j) + '\n')
            zero_ectopic.write(str(approximate_entropy(random_sample, n, 0.2, 'bucket', rsplit=5)) + '\n')


			#2
            random_index = random.randint(0, len(random_sample)-2)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*2

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 2] = consecutives

            #two_cons_ectopic.write('sample length ' + str(j) + '\n')
            two_cons_ectopic.write(str(approximate_entropy(random_sample_, n, 0.2, 'bucket', rsplit=5)) + '\n')


			#5
            random_index = random.randint(0, len(random_sample)-5)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*5

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 5] = consecutives
            #five_cons_ectopic.write('sample length ' + str(j) + '\n')
            five_cons_ectopic.write(str(approximate_entropy(random_sample_, n, 0.2, 'bucket', rsplit=5)) + '\n')


			#5
            random_index = random.randint(0, len(random_sample)-10)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*10

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 10] = consecutives
            ten_cons_ectopic.write(str(approximate_entropy(random_sample_, n, 0.2, 'bucket', rsplit=5)) + '\n')


			#tenpercent
            random_index = random.randint(0, len(random_sample)-j//10)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]* int(j//10)

            random_sample_ = random_sample
            random_sample_[random_index:random_index + j//10] = consecutives
            tenpercent_cons_ectopic.write(str(approximate_entropy(random_sample_, n, 0.2, 'bucket', rsplit=5)) + '\n')



if __name__ == '__main__':

    zero_ectopic = open('0_consecutive_zeros_approximate.txt', 'a')
    two_cons_ectopic = open('2_consecutive_zeros_approximate.txt', 'a')
    five_cons_ectopic = open('5_consecutive_zeros_approximate.txt', 'a')
    ten_cons_ectopic = open('10_consecutive_zeros_approximate.txt', 'a')
    tenpercent_cons_ectopic = open('10percent_consecutive_zeros_approximate.txt', 'a')
    #start_time = time.time()
    #for i in range(1):
    main()
    #print("--- %s seconds ---" % (time.time() - start_time))
    zero_ectopic.close()
    two_cons_ectopic.close()
    five_cons_ectopic.close()
    ten_cons_ectopic.close()
    tenpercent_cons_ectopic.close()
