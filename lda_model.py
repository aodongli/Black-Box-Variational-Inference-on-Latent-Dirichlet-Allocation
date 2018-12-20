import numpy as np

class lda_model:

    def __init__(self, num_topics, num_terms):
        self.num_topics = num_topics
        self.num_terms = num_terms
        self.alpha = 1.0
        self.log_prob_w = np.zeros([num_topics, num_terms])

def load_model(model_root):
    # model_root is a string
    with open(model_root + '.other', 'rb') as other_file:
        parser = other_file.readline().split()
        num_topics = int(parser[1])
        parser = other_file.readline().split()
        num_terms = int(parser[1])
        parser = other_file.readline().split()
        alpha = float(parser[1])

    model = lda_model(num_topics, num_terms)
    model.alpha = alpha

    with open(model_root + '.beta', 'rb') as beta_file:
        for i in range(num_topics):
            parser = beta_file.readline()
            parser = np.array(map(float, parser.split()))
            model.log_prob_w[i] = parser

    return model


def main():
    lda = load_model('./model/final')
    # print lda.log_prob_w[1][1]

if __name__ == '__main__':
    main()
