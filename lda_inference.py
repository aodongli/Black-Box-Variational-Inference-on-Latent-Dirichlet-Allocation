import lda_model
import read_data
import numpy as np
from scipy.special import digamma
from math import lgamma
from math import isnan
from scipy.stats import dirichlet

np.random.seed(123)

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

        # compute gamma gradient

        dig = digamma(var_gamma)
        var_gamma_sum = np.sum(var_gamma)
        digsum = digamma(var_gamma_sum)

        ln_theta = np.log(sample_theta)

        dqdg = ln_theta - dig + digsum # S * k matrix

        ln_p_theta = dirichlet.logpdf(np.transpose(sample_theta), \
                                [lda_model.alpha] * lda_model.num_topics) 
                                # S length vector
        ln_q_theta = dirichlet.logpdf(np.transpose(sample_theta), var_gamma)
                                # S length vector

        E_p_z = np.sum(ln_theta * np.sum(phi, 0), 1) # S length vector

        grad_gamma = np.average(dqdg * np.reshape(ln_p_theta - ln_q_theta + E_p_z, (S, 1)), 0)

        if adagrad:
            g_var_gamma += grad_gamma ** 2
            grad_gamma = grad_gamma / (np.sqrt(g_var_gamma) + epsilon)
        var_gamma = var_gamma + rho * grad_gamma # update
        

        # for phi
        dig = digamma(var_gamma)
        var_gamma_sum = np.sum(var_gamma)
        digsum = digamma(var_gamma_sum)

        for n in range(doc.length):
            # sample S z for each word
            # print phi[n,:]
            sample_z = np.random.multinomial(1, phi[n,:], S) # S * k matrix

            # compute phi gradient
            which_j = np.argmax(sample_z, 1)

            dqdphi = 1 / phi[n][which_j] # S length vector

            ln_p_w = lda_model.log_prob_w[which_j][:,doc.words[n]] # S length vector

            ln_q_phi = np.log(phi[n][which_j]) # S length vector

            E_p_z_theta = dig[which_j] - digsum # S length vector

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


def main():
    corpus = read_data.read_corpus('./data/tmp.data')
    model = lda_model.load_model('./model/final')
    with open('./data/bbvi_test_likelihood', 'a') as ofile:
        for i in range(len(corpus)):
            i = i + 41
            print i
            likelihood = lda_inference(corpus[i], model, False)
            ofile.write(str(likelihood) + '\n')
            ofile.flush()
    return

if __name__ == '__main__':
    main()