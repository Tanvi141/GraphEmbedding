import numpy as np
import sys
from ge.classify import read_node_label, Classifier
from ge import LINE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings(embeddings, tan_infile, dirname):
    of = open(dirname+"/outfile0.txt", "w")
    count = 0
    file_num = 0
    for i in embeddings:
        count += 1    
        for ele in (embeddings[i]):
            of.write(str(ele)+" ")
        of.write("\n")
        if count >= 10000:
            count = 0
            file_num += 1
            of.close()
            of = open(dirname+"/outfile%s.txt"%(file_num), "w")
    of.close()	


if __name__ == "__main__":
    #tan_infile = "/home2/sp504/tanvi/test_ip.txt"
    tan_infile = "/home2/sp504/tanvi/al_songs_ints.txt"
    dirname = sys.argv[1]
    #G = nx.read_edgelist(tan_infile,
                         create_using=nx.DiGraph(), nodetype=int, data=[('weight', int)])
    print("done reading")
    model = LINE(tan_infile, embedding_size=9, order='second')
    model.train(batch_size=1024, epochs=50, verbose=2)
    embeddings = model.get_embeddings()

    evaluate_embeddings(embeddings, tan_infile, dirname)
    #plot_embeddings(embeddings, tan_infile)
