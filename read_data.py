# read documents data

class document:
    def __init__(self, length):
        self.length = length
        self.counts = []
        self.words = []
        self.total = 0

def read_corpus(data_file):
    corpus = []
    with open(data_file, 'rb') as dfile:
        for doc_abs in dfile.readlines():
            parser = doc_abs.split()
            doc = document(int(parser[0]))
            for pair in parser[1:]:
                wid, wcnt = map(int, pair.split(':'))
                doc.words.append(wid)
                doc.counts.append(wcnt)
                doc.total += wcnt
            assert(len(doc.words) == doc.length)
            corpus.append(doc)
    return corpus


def main():
    corpus = read_corpus('./data/test.data')
    print corpus[0].length


if __name__ == '__main__':
    main()