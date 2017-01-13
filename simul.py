# coding=utf-8
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as LA


def g_wrapper(p, alpha=0):
    def get_fri(G):
        flag = True

        if flag:
            deg = np.ones(shape=(len(G.nodes()),))
            for i in range(len(G.nodes())):
                deg[i] += G.degree(i)
            deg += alpha
            deg = deg / (1.0 * np.sum(deg, axis=0))
            mu = np.random.binomial(1, p, size=(len(deg),))
            n_deg = np.asarray([np.random.binomial(1, deg[i]) for i in range(len(deg))])
            mkfri = mu * n_deg
            friend = [f for f in range(len(deg)) if mkfri[f] == 1]
        else:
            friend = []

            for i in range(len(G.nodes())):
                rand = np.random.binomial(1, p)
                if rand == 1:
                    friend.append(i)

        return friend

    return get_fri


def f_wrapper(q):
    def get_comm(G, fris, comm_assign):
        rand = np.random.binomial(1, q)
        nb_comm = len(np.unique(comm_assign.values()))
        if rand == 1:
            comm_hot = np.random.multinomial(1, [1 / (1.0 * nb_comm)] * nb_comm)
        else:
            fri_in_comm = np.zeros(shape=(nb_comm,))
            for friend in fris:
                assert len(comm_assign) == len(G.nodes()) - 1, 'haha'
                fri_in_comm[comm_assign[friend]] += 1

            nb_fris = len(fris)
            if nb_fris == 0:
                comm_hot = np.random.multinomial(1, [1 / (1.0 * nb_comm)] * nb_comm)
            else:
                comm_hot = np.random.multinomial(1, fri_in_comm / (1.0 * nb_fris))

        comm = [f for f in range(len(comm_hot)) if comm_hot[f] == 1]
        assert len(comm) == 1
        return comm[0]

    return get_comm


def load_data(mode='top'):
    if mode == 'top':
        url = 'com-lj.top5000.cmty.txt'
    elif mode == 'all':
        url = 'com-lj.all.cmty.txt'
    else:
        return

    comm_data = []
    f = open(url, 'r')
    for i, l in enumerate(f.readlines()):
        l.decode('utf-8')
        mem = l.split('\t')
        comm_data.append(len(mem))

    comm_data = sorted(comm_data, reverse=True)

    return comm_data


def whole_model(get_fri, get_comm, c0=3, n0=2, p=0.02, Iter=50000):
    G = nx.Graph()

    # record the community of one person
    # comm[person_id] = comm_id
    comm = dict()
    nb_comm = 0
    nb_mem = 0


    # initailize a group of communities and people
    # param: c0: community no.
    #       n0: people no. in one community

    for comm_id in range(nb_comm, nb_comm + c0):
        for mem_id in range(nb_mem, nb_mem + n0):
            # print comm_id, mem_id
            G.add_node(mem_id)
            # print 'nodes ', G.nodes()
            for pre_mem in range(nb_mem, mem_id):
                # print 'idx: ', nb_mem, mem_id, pre_mem
                G.add_edge(mem_id, pre_mem)
            # print 'nodes ', G.nodes()
            comm[mem_id] = comm_id
        nb_mem += n0
    nb_comm += c0

    print 'initial: '
    # print G.nodes()
    # print comm


    for iter in range(Iter):
        incre_p = np.random.binomial(1, p)

        if iter % 500 == 0:
            print 'iter {}  nb_comm: '.format(iter), len(np.unique(comm.values())),
            print 'nb_nodes for unique: ', len(comm),
            print 'nb_nodes: ', len(G.nodes())

        # introducing one new person into system
        # param: p: the change to introduce one person
        if incre_p == 0:

            # randomly build relationship with others which can be sophisticated
            # according to the degree of one note
            # param: m: friends no.
            #       g: the way to choose friends, return a m-list
            cur_id = nb_mem
            nb_mem += 1
            # stime_fris = timeit.default_timer()

            fris = get_fri(G)

            # etime_fris = timeit.default_timer()
            # if iter%100==0:
            #     print etime_fris - stime_fris
            G.add_node(cur_id)
            for friend in fris:
                G.add_edge(friend, cur_id)

            # assign one community to him which can also be sophisticated
            # according to friends no. in community
            # param: f: the way to choose community

            # print '\ninside ',
            # print 'nb_comm: ', len(np.unique(comm.values())),
            # print 'nb_nodes for unique: ', len(comm),
            # print 'nb_nodes: ', len(G.nodes())

            # stime_comm = timeit.default_timer()
            in_comm = get_comm(G, fris, comm)
            # etime_comm = timeit.default_timer()
            # if iter % 100 == 0:
            #     print etime_comm - stime_comm
            comm[cur_id] = in_comm

        # OR introducing one full-connected community into system
        # param: 1-p: the change to introduce one community
        #       n0: people no. in one commuinty
        else:
            for mem_id in range(nb_mem, nb_mem + n0):
                G.add_node(mem_id)
                for pre_mem in range(nb_mem, mem_id):
                    G.add_edge(mem_id, pre_mem)
                comm[mem_id] = nb_comm
            nb_mem += n0
            nb_comm += 1

    return G, comm


def visualize(G, comm):
    # numbers: ordered community id on the size
    comm_size = dict()
    for mem in comm:
        if comm[mem] in comm_size:
            comm_size[comm[mem]] += 1
        else:
            comm_size[comm[mem]] = 1
    temp_sum = 1.0*sum(comm_size.values())
    # print comm_size
    comm_size = sorted(comm_size.iteritems(), key=lambda d: d[1], reverse=True)
    # print comm_size
    # comm_proportion = [(k, v/temp_sum) for k, v in comm_size]
    # print comm_proportion

    # number: nb_mem, nb_comm

    # graph: community size v.s. community size rank

    rank = [i for i in range(1, len(comm_size)+1)]
    size = [v for k, v in comm_size]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter, = ax.loglog(rank, size, "o")

    lnrank = [math.log(f, math.e) for f in rank]
    lnsize = [math.log(f, math.e) for f in size]
    x = np.mat([lnrank, [1] * len(rank)]).T
    y = np.mat(lnsize).T

    lens = len(x)
    trainlen = int(0.75*lens)
    (t, res, rank_, s) = LA.lstsq(x[:trainlen], y[:trainlen])
    r = t[0][0]
    c = t[1][0]
    y_ = [float(np.exp(r * a + c)[0][0]) for a in lnrank]
    # print len(x)
    # print len(y)
    # print len(rank)
    # print len(y_)
    # print y_
    linreg, = ax.loglog(rank, y_, "r-")
    ax.text(20, 90, 'lny = %.3f * ln(r) + %.3f' % (r, c), fontsize=15)
    ax.text(20, 60, 'y = %.3f / ( r ^ %.3f )' % (float(np.exp(c)[0][0]), r), fontsize=15)

    plt.legend([scatter, linreg], [r'loglog-scatter', r'linear-regression'], loc=3)
    # print r, c

    plt.show()
    return r


def visual_test(comm_Data):
    rank = [i for i in range(1, len(comm_Data) + 1)]
    size = comm_Data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter, = ax.loglog(rank, comm_Data, "o")
    lnrank = [math.log(f, math.e) for f in rank]
    lnsize = [math.log(f, math.e) for f in size]
    x = np.mat([lnrank, [1] * len(rank)]).T
    y = np.mat(lnsize).T
    (t, res, rank_, s) = LA.lstsq(x[:4000], y[:4000])
    r = t[0][0]
    c = t[1][0]
    y_ = [float(np.exp(r * a + c)[0][0]) for a in lnrank]
    linreg, = ax.loglog(rank, y_, "r-")
    ax.text(100, 900, 'lny = %.3f * ln(r) + %.3f' % (r, c), fontsize=15)
    ax.text(100, 600, 'y = %.3f / ( r ^ %.3f )' % (float(np.exp(c)[0][0]), r), fontsize=15)

    plt.legend([scatter, linreg], [r'loglog-scatter', r'linear-regression'], loc=3)
    print r, c

    plt.show()


def simulation():
    c0 = 3
    n0 = 3
    p = 0.12  # add person or community probability
    mu = 1  # add relationship by mu for one person
    q = 0.1  # chance of choose random community
    alpha = 5  # smooth parameter
    Iter = 2000

    G, comm = whole_model(g_wrapper(mu, alpha), f_wrapper(q), c0, n0, p, Iter)
    r = visualize(G, comm)



# simulation()



comm_data = load_data()
visual_test(comm_data)