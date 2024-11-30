import itertools
import numpy
import time
import random
from random import sample
from numpy import log
import sys
import os

##########################################################################

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
    return -numpy.log(r)


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

    return numpy.log((m+1)/(m-1)) * (A-B)

################################################################################
################################################################################
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

			#no skipped beats
            copy = random_sample
			#no ectopic beats
            #zero_ectopic.write('sample length ' + str(j) + '\n')
            zero_ectopic.write(str(bubble_entropy_fast(random_sample,n)) + '\n')


			#2 consecutive
            random_index = random.randint(0, len(random_sample)-2)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*2

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 2] = consecutives
            two_cons_ectopic.write(str(bubble_entropy_fast(random_sample_,n)) + '\n')


			#5 consecutive
            random_index = random.randint(0, len(random_sample)-5)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*5

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 5] = consecutives
            five_cons_ectopic.write(str(bubble_entropy_fast(random_sample_,n)) + '\n')


			#10 consecutive
            random_index = random.randint(0, len(random_sample)-10)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*10

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 10] = consecutives
            ten_cons_ectopic.write(str(bubble_entropy_fast(random_sample_,n)) + '\n')


			#10percent consecutive
            random_index = random.randint(0, len(random_sample)-j//10)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*int(j//10)

            random_sample_ = random_sample
            random_sample_[random_index:random_index + j//10] = consecutives
            tenpercent_cons_ectopic.write(str(bubble_entropy_fast(random_sample_,n)) + '\n')


        for j in sample_lengths_2:
            random_sample_ = sample(x2, j)
            index = x2.index(random_sample_[0])#index of start of random sample
            N_index = list(range(index, index + j))#indexes of random sample
            to_be_deleted = list(set(not_normal_indexes) & set(N_index))

            random_sample = [i for i in random_sample_ if i not in to_be_deleted]
            #for k in random_sample_:
                #for l in range(len(to_be_deleted)):
                    #if k != to_be_deleted[l]:
                        #random_sample.append(k)

			#no skipped beats
            #a0.write('sample length ' + str(j) + '\n')
            zero_ectopic.write(str(bubble_entropy_fast(random_sample,n)) + '\n')


			#2 consecutive
            random_index = random.randint(0, len(random_sample)-2)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*2

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 2] = consecutives
            two_cons_ectopic.write(str(bubble_entropy_fast(random_sample_,n)) + '\n')


			#5 consecutive
            random_index = random.randint(0, len(random_sample)-5)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*5

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 5] = consecutives
            five_cons_ectopic.write(str(bubble_entropy_fast(random_sample_,n)) + '\n')


			#10 consecutive
            random_index = random.randint(0, len(random_sample)-10)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*10

            random_sample_ = random_sample
            random_sample_[random_index:random_index + 10] = consecutives
            ten_cons_ectopic.write(str(bubble_entropy_fast(random_sample_,n)) + '\n')


			#10percent consecutive
            random_index = random.randint(0, len(random_sample)-j//10)
            #random_consecutive = random.randint(0, 30)
            consecutives = [0.0]*int(j//10)

            random_sample_ = random_sample
            random_sample_[random_index:random_index + j//10] = consecutives
            tenpercent_cons_ectopic.write(str(bubble_entropy_fast(random_sample_,n)) + '\n')


if __name__ == '__main__':

    zero_ectopic = open('0_consecutive_zeros_bubble.txt', 'a')
    two_cons_ectopic = open('2_consecutive_zeros_bubble.txt', 'a')
    five_cons_ectopic = open('5_consecutive_zeros_bubble.txt', 'a')
    ten_cons_ectopic = open('10_consecutive_zeros_bubble.txt', 'a')
    tenpercent_cons_ectopic = open('10percent_consecutive_zeros_bubble.txt', 'a')
    #start_time = time.time()
    main()
    #print("--- %s seconds ---" % (time.time() - start_time))
    zero_ectopic.close()
    two_cons_ectopic.close()
    five_cons_ectopic.close()
    ten_cons_ectopic.close()
    tenpercent_cons_ectopic.close()
