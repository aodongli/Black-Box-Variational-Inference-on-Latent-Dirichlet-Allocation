import os
import ast
import numpy as np
import matplotlib.pyplot as plt

RESULTS_FOLDERS = ['./data/bbvi_test_likelihood',  './data/bbvi_test_likelihood_adagrad', './data/test_inf-lda-lhood']

SAVE_PATH = './plots/'

COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired

def main():
    # create result directory
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    num_test_samples = 100

    bbvi = []
    bbvi_adagrad = []
    vi = []

    with open(RESULTS_FOLDERS[0], 'rb') as resfile:
        cnt = 0
        for line in resfile.readlines():
            bbvi.append(float(line))
            cnt += 1
            if cnt > 100:
                break

    with open(RESULTS_FOLDERS[1], 'rb') as resfile:
        cnt = 0
        for line in resfile.readlines():
            bbvi_adagrad.append(float(line))
            cnt += 1
            if cnt > 100:
                break

    with open(RESULTS_FOLDERS[2], 'rb') as resfile:
        cnt = 0
        for line in resfile.readlines():
            vi.append(float(line))
            cnt += 1
            if cnt > 100:
                break


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(bbvi)
    ax.plot(bbvi_adagrad)
    ax.plot(vi)

    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i,j in enumerate(ax.lines):
        j.set_color(colors[i])

    # print plt.ylim()
    # plt.ylim(-45000, 4000) 

    ax.legend(['bbvi', 'bbvi_adagrad', 'vi'])
    
    plt.ylabel('log likelihood')
    plt.xlabel('text index')
    #plt.show()
    plt.savefig(SAVE_PATH + 'results_ori_scale.pdf')




if __name__ == '__main__':
    main()