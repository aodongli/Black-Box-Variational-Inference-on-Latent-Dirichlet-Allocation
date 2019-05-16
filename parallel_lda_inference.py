import lda_model
import read_data
import numpy as np
from scipy.special import digamma
from math import lgamma
from math import isnan
from scipy.stats import dirichlet

from optparse import OptionParser
# multiprocessing
from multiprocessing import Pool, cpu_count
import threading
import Queue

np.random.seed(123)

def parse_args():
    parser = OptionParser()
    parser.set_defaults(numthreads=1)
    parser.add_option("--numthreads", type="int", dest="numthreads",
                    help="number of threads to use for inference algorithms")
    parser.add_option("--thread", dest="isthread", action="store_true",
                    default=False,
                    help="use thread")
    (options, args) = parser.parse_args()
    return options

def wrapper_lda_inference(item):
    return lda_inference(item[0], item[1], item[2])

def lda_inference(doc, lda_model, adagrad=True):
    S = 10 # samples
    converged = 100.0
    rho = 1e-4 # learning rate
    if adagrad:
        epsilon = 1e-6 # fudge factor
        g_phi = np.zeros([doc.length, lda_model.num_topics])
        g_var_gamma = np.zeros([lda_model.num_topics])

    # variational parameters

    phi = np.ones([doc.length, lda_model.num_topics]) \
            / lda_model.num_topics  # N * k matrix 
    var_gamma = np.ones([lda_model.num_topics]) * lda_model.alpha \
             + doc.total / float(lda_model.num_topics)

    likelihood_old = 0

    var_ite = 0
    while (converged > 1e-3 and var_ite < 1e3):
        var_ite += 1

        # sample S theta
        sample_theta = np.random.dirichlet(var_gamma, S)

        # sample S z for each word n
        sample_zs = np.zeros([doc.length, S], dtype=np.int32)
        for n in range(doc.length):
            # sample S z for each word
            sample_z = np.random.multinomial(1, phi[n,:], S) # S * k matrix
            which_j = np.argmax(sample_z, 1) # S length vector
            sample_zs[n,:] = which_j

        # compute gamma gradient

        dig = digamma(var_gamma)
        var_gamma_sum = np.sum(var_gamma)
        digsum = digamma(var_gamma_sum)

        ln_theta = np.log(sample_theta) # S * k matrix

        dqdg = ln_theta - dig + digsum # S * k matrix

        ln_p_theta = dirichlet.logpdf(np.transpose(sample_theta), \
                                [lda_model.alpha] * lda_model.num_topics) 
                                # S length vector
        ln_q_theta = dirichlet.logpdf(np.transpose(sample_theta), var_gamma)
                                # S length vector

        # explicitly evaluate expectation
        # E_p_z = np.sum(ln_theta * np.sum(phi, 0), 1) # S length vector

        # monte-carlo estimated expectation
        E_p_z = np.zeros(S) # S length vector
        for sample_id in range(S):
            cur_ln_theta = ln_theta[sample_id,:]
            sampled_ln_theta = []
            for n in range(doc.length):
                which_j = sample_zs[n,:]
                sampled_ln_theta += list(cur_ln_theta[which_j]) # (doc.counts[n] * list(cur_ln_theta[which_j]))
            E_p_z[sample_id] = np.average(sampled_ln_theta)

        grad_gamma = np.average(dqdg * np.reshape(ln_p_theta - ln_q_theta + E_p_z, (S, 1)), 0)

        # update
        if adagrad:
            g_var_gamma += grad_gamma ** 2
            grad_gamma = grad_gamma / (np.sqrt(g_var_gamma) + epsilon)
        var_gamma = var_gamma + rho * grad_gamma 
        

        # for phi

        # for explicit evaluation of expectation
        # dig = digamma(var_gamma)
        # var_gamma_sum = np.sum(var_gamma)
        # digsum = digamma(var_gamma_sum)

        # resample from updated gamma
        sample_theta = np.random.dirichlet(var_gamma, S)
        ln_theta = np.log(sample_theta) # S * k matrix

        for n in range(doc.length):

            # compute phi gradient
            which_j = sample_zs[n,:]

            dqdphi = 1 / phi[n][which_j] # S length vector

            ln_p_w = lda_model.log_prob_w[which_j][:,doc.words[n]] # S length vector

            ln_q_phi = np.log(phi[n][which_j]) # S length vector

            # explicitly evaluate expectation
            # E_p_z_theta = dig[which_j] - digsum # S length vector

            # monte-carlo estimated expectation
            E_p_z_theta = np.zeros(S) # S length vector
            for sample_id in range(S):
                cur_ln_theta = ln_theta[sample_id,:]
                E_p_z_theta += cur_ln_theta[which_j]
            E_p_z_theta = E_p_z_theta / S

            # print( dqdphi.shape, ln_p_w.shape, ln_q_phi.shape, E_p_z_theta.shape)
            # print (lda_model.log_prob_w[which_j][:,doc.words[n]])
            # print ln_p_w,ln_q_phi,E_p_z_theta
            grad_phi = doc.counts[n] * dqdphi * (ln_p_w - ln_q_phi + E_p_z_theta)

            # update phi

            for i,j in enumerate(which_j): 
                if adagrad:
                    g_phi[n][j] += grad_phi[i] ** 2
                    grad_phi[i] = grad_phi[i] / (np.sqrt(g_phi[n][j]) + epsilon)
                # print grad_phi[i]
                phi[n][j] = phi[n][j] + rho * grad_phi[i]
                if phi[n][j] < 0: # bound phi
                    phi[n][j] = 0 
                phi[n] /= np.sum(phi[n]) # normalization


        # compute likelihood

        likelihood = compute_likelihood(doc, lda_model, phi, var_gamma)
        assert(not isnan(likelihood))
        converged = abs((likelihood_old - likelihood) / likelihood_old)
        likelihood_old = likelihood
        # print likelihood, converged
    return likelihood

def compute_likelihood(doc, lda_model, phi, var_gamma):
    # an implementation reproducing lda-c code
    # var_gamma is a vector
    likelihood = 0
    digsum = 0
    var_gamma_sum = 0

    dig = digamma(var_gamma)
    var_gamma_sum = np.sum(var_gamma)
    digsum = digamma(var_gamma_sum)

    likelihood = lgamma(lda_model.alpha * lda_model.num_topics) \
                - lda_model.num_topics * lgamma(lda_model.alpha) \
                - (lgamma(var_gamma_sum))

    for k in range(lda_model.num_topics):

        likelihood += (lda_model.alpha - 1)*(dig[k] - digsum) \
                    + lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum)

        for n in range(doc.length):
            if phi[n][k] > 0:
                likelihood += doc.counts[n] * \
                    (phi[n][k]*((dig[k] - digsum) - np.log(phi[n][k]) \
                                + lda_model.log_prob_w[k,doc.words[n]]))

    return likelihood

class ThreadClass(threading.Thread):
    def __init__(self, queue, ofile=None):
        threading.Thread.__init__(self)
        self.queue = queue
        self.ofile = ofile

    def run(self):
        while True:
            item = self.queue.get()
            # work
            likelihood = lda_inference(item[0], item[1], item[2])
            if self.ofile:
                self.ofile.write(str(likelihood) + '\n')
            self.queue.task_done()


def main():
    corpus = read_data.read_corpus('./data/tmp.data')
    model = lda_model.load_model('./model/final')

    options = parse_args()
    print(options)

    numthreads = options.numthreads
    ofile = open('./data/parallel_bbvi_test_likelihood_all_sample' + str(numthreads) + str(options.isthread), 'a')

    if options.isthread:
        queue = Queue.Queue()
        for i in range(numthreads):
            t = ThreadClass(queue, ofile)
            t.setDaemon(True)
            t.start()

        for i in range(len(corpus)):
            # print(i)
            queue.put((corpus[i], model, False))

        queue.join()
    else:
        pool = Pool(min(cpu_count(), numthreads))
        likelihood = pool.map(wrapper_lda_inference, [(doc, model, False) for doc in corpus])
        ofile.write('\n'.join(map(str, likelihood)))

    return

if __name__ == '__main__':
    main()