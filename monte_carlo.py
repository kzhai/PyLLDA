#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Ke Zhai (zhaike@cs.umd.edu)
"""
import time;
import numpy;
import scipy;
import sys;

from inferencer import compute_dirichlet_expectation
from inferencer import Inferencer;

"""
This is a python implementation of labeled lda, based on gibbs sampling, with hyper parameter updating.

References:
[1] Danie Ramage, David Hall, Ramesh Nallapati, Christopher Manning, Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora. In EMNLP, 2009
"""

class MonteCarlo(Inferencer):
    def __init__(self,
                 hyper_parameter_optimize_interval=10,
                 symmetric_alpha_alpha=True,
                 symmetric_alpha_beta=True,
                 ):
        Inferencer.__init__(self, hyper_parameter_optimize_interval);

        self._symmetric_alpha_alpha = symmetric_alpha_alpha
        self._symmetric_alpha_beta = symmetric_alpha_beta

    """
    @param num_topics: desired number of topics
    @param data: a dict data type, indexed by document id, value is a list of words in that document, not necessarily be unique
    """
    def _initialize(self, corpus, vocab, labels, alpha_alpha, alpha_beta):
        Inferencer._initialize(self, vocab, labels, alpha_alpha, alpha_beta);
        
        self._number_of_topics = len(self._label_to_index)
        
        self._parsed_corpus, self._parsed_labels = self.parse_data(corpus);
        
        # define the total number of document
        self._number_of_documents = len(self._parsed_corpus);
        
        # define the counts over different topics for all documents, first indexed by doc_id id, the indexed by topic id
        self._n_dk = numpy.zeros((self._number_of_documents, self._number_of_topics), dtype=int)
        # define the counts over words for all topics, first indexed by topic id, then indexed by token id
        self._n_kv = numpy.zeros((self._number_of_topics, self._number_of_types), dtype=int)
        self._n_k = numpy.zeros(self._number_of_topics, dtype=int)
        # define the topic assignment for every word in every document, first indexed by doc_id id, then indexed by word word_pos
        self._k_dn = {};
        
        self.random_initialize();

    def random_initialize(self, parsed_corpus_response=None):
        if parsed_corpus_response == None:
            word_idss = self._parsed_corpus;
            labels = self._parsed_labels
        else:
            word_idss, labels = parsed_corpus_response;

        # topic assignments
        self._k_dn = {}

        for doc_id in xrange(len(word_idss)):
            self._k_dn[doc_id] = numpy.zeros(len(self._parsed_corpus[doc_id]));
            for word_pos in xrange(len(word_idss[doc_id])):
                type_index = self._parsed_corpus[doc_id][word_pos];
                topic_index = numpy.random.randint(self._number_of_topics);
                
                self._k_dn[doc_id][word_pos] = topic_index;
                self._n_dk[doc_id, topic_index] += 1;
                self._n_kv[topic_index, type_index] += 1;
                self._n_k[topic_index] += 1;
    
    def sample_document(self, doc_id, local_parameter_iteration=1):
        parsed_labels = self._parsed_labels[doc_id, :];
        
        for iter in xrange(local_parameter_iteration):
            for word_pos in xrange(len(self._parsed_corpus[doc_id])):
                assert word_pos >= 0 and word_pos < len(self._parsed_corpus[doc_id])
                
                # retrieve the word_id
                word_id = self._parsed_corpus[doc_id][word_pos]
                old_topic = self._k_dn[doc_id][word_pos]
                
                self._n_kv[old_topic, word_id] -= 1
                self._n_k[old_topic] -= 1
                self._n_dk[doc_id, old_topic] -= 1
                
                posterior_probability = parsed_labels.copy()
                posterior_probability *= self._n_dk[doc_id, :] + self._alpha_alpha
                # posterior_probability /= numpy.sum(self._n_dk[doc_id, :] + self._alpha_alpha)
                posterior_probability *= self._n_kv[:, word_id] + self._alpha_beta[word_id]
                posterior_probability /= self._n_k + numpy.sum(self._alpha_beta)
                
                # sample a new topic out of a distribution according to old_log_probability
                temp_probability = posterior_probability / numpy.sum(posterior_probability)
                temp_topic_probability = numpy.random.multinomial(1, temp_probability)[numpy.newaxis, :]
                # print numpy.nonzero(temp_topic_probability == 1)
                new_topic = numpy.nonzero(temp_topic_probability == 1)[1][0];

                # after we draw a new topic for that word_id, we will change the topic|doc_id counts and word_id|topic counts, i.e., add the counts back
                self._n_dk[doc_id, new_topic] += 1
                self._n_kv[new_topic, word_id] += 1;
                self._n_k[new_topic] += 1;
                # assign the topic for the word_id of current document at current position
                self._k_dn[doc_id][word_pos] = new_topic
                
    """
    sample the corpus to train the parameters
    @param hyper_delay: defines the delay in updating they hyper parameters, i.e., start updating hyper parameter only after hyper_delay number of gibbs sampling iterations. Usually, it specifies a burn-in period.
    """
    def learning(self):
        # sample the total corpus
        # for iter1 in xrange(number_of_iterations):
        self._counter += 1;
        
        processing_time = time.time();

        # sample every document
        for doc_id in xrange(self._number_of_documents):
            self.sample_document(doc_id)

            if (doc_id + 1) % 1000 == 0:
                print "successfully sampled %d documents" % (doc_id + 1)

        if self._counter % self._hyper_parameter_optimize_interval == 0:
            self.optimize_hyperparameters()

        processing_time = time.time() - processing_time;
        print("iteration %i finished in %d seconds with log-likelihood %g" % (self._counter, processing_time, self.log_likelihood(self._alpha_alpha, self._alpha_beta)))

    def log_likelihood(self, alpha_alpha, alpha_beta):
        """
        likelihood function
        """
        assert self._n_dk.shape == (self._number_of_documents, self._number_of_topics);
        assert alpha_alpha.shape == (self._number_of_topics,);
        assert alpha_beta.shape == (self._number_of_types,);
        
        alpha_sum = numpy.sum(alpha_alpha);
        beta_sum = numpy.sum(alpha_beta);
        
        log_likelihood = 0
        
        # compute the log posterior of the document
        log_likelihood += len(self._parsed_corpus) * scipy.special.gammaln(numpy.sum(alpha_alpha))
        log_likelihood -= len(self._parsed_corpus) * numpy.sum(scipy.special.gammaln(alpha_alpha))
        
        for doc_id in xrange(len(self._parsed_corpus)):
            log_likelihood += numpy.sum(scipy.special.gammaln(self._n_dk[doc_id, :] + alpha_alpha) * self._parsed_labels[doc_id, :]);
            # log_likelihood -= scipy.special.gammaln(numpy.sum(self._n_dk[doc_id, :]) + alpha_sum);
            log_likelihood -= scipy.special.gammaln(len(self._parsed_corpus[doc_id]) + alpha_sum);
        
        # compute the log posterior of the topic
        log_likelihood += self._number_of_topics * scipy.special.gammaln(numpy.sum(alpha_beta)) 
        log_likelihood -= self._number_of_topics * numpy.sum(scipy.special.gammaln(alpha_beta))
        
        for topic_id in xrange(self._number_of_topics):
            log_likelihood += numpy.sum(scipy.special.gammaln(self._n_kv[topic_id, :] + alpha_beta));
            log_likelihood -= scipy.special.gammaln(self._n_k[topic_id] + beta_sum);
        
        return log_likelihood
    
    '''
    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = (self._n_kv + self._alpha_beta[numpy.newaxis, :]) / (self._n_k[:, numpy.newaxis] + numpy.sum(self._alpha_beta))
        thetas = (self._n_dk + self._parsed_labels * self._alpha_alpha[numpy.newaxis, :]) / (self._n_dk + self._parsed_labels * self._alpha_alpha[numpy.newaxis, :]).sum(axis=1)[:, numpy.newaxis]

        log_per = N = 0
        for doc, theta in zip(docs, thetas):
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:, w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)
    '''
    
    """
    """
    def optimize_hyperparameters(self, hyper_parameter_samples=10, hyper_parameter_step=1.0, hyper_parameter_iteration=50):
        # old_hyper_parameters = [numpy.log(self._alpha_alpha), numpy.log(self._alpha_beta)]
        # old_hyper_parameters = numpy.hstack((numpy.log(self._alpha_alpha), numpy.log(self._alpha_beta)));
        # assert old_hyper_parameters.shape==(self._number_of_topics+self._number_of_types,);
        
        old_log_alpha_alpha = numpy.log(self._alpha_alpha);
        old_log_alpha_beta = numpy.log(self._alpha_beta);
        
        for ii in xrange(hyper_parameter_samples):
            log_likelihood_old = self.log_likelihood(self._alpha_alpha, self._alpha_beta)
            log_likelihood_new = numpy.log(numpy.random.random()) + log_likelihood_old
            
            l_log_alpha_alpha = old_log_alpha_alpha;
            if self._symmetric_alpha_alpha:
                l_log_alpha_alpha -= numpy.random.random() * hyper_parameter_step
            else:
                l_log_alpha_alpha -= numpy.random.random(old_log_alpha_alpha.shape) * hyper_parameter_step
            r_log_alpha_alpha = old_log_alpha_alpha + hyper_parameter_step;
            assert numpy.all(l_log_alpha_alpha <= old_log_alpha_alpha), (l_log_alpha_alpha, old_log_alpha_alpha);
            assert numpy.all(r_log_alpha_alpha >= old_log_alpha_alpha), (r_log_alpha_alpha, old_log_alpha_alpha);
            
            l_log_alpha_beta = old_log_alpha_beta;
            if self._symmetric_alpha_beta:
                l_log_alpha_beta -= numpy.random.random() * hyper_parameter_step
            else:
                l_log_alpha_beta -= numpy.random.random(old_log_alpha_beta.shape) * hyper_parameter_step
            r_log_alpha_beta = old_log_alpha_beta + hyper_parameter_step;
            assert numpy.all(l_log_alpha_beta <= old_log_alpha_beta), (l_log_alpha_beta, old_log_alpha_beta);
            assert numpy.all(r_log_alpha_beta >= old_log_alpha_beta), (r_log_alpha_beta, old_log_alpha_beta);
            
            # l = old_hyper_parameters - numpy.random.random(old_hyper_parameters.shape) * hyper_parameter_step
            # r = old_hyper_parameters + hyper_parameter_step;

            # l = [x - numpy.random.random() * hyper_parameter_step for x in old_hyper_parameters]
            # r = [x + hyper_parameter_step for x in old_hyper_parameters]

            for jj in xrange(hyper_parameter_iteration):
                # new_hyper_parameters = l + numpy.random.random(old_hyper_parameters.shape) * (r - l);
                # new_alpha_alpha = numpy.exp(new_hyper_parameters[:self._number_of_topics]);
                # new_alpha_beta = numpy.exp(new_hyper_parameters[self._number_of_topics:]);
                
                new_log_alpha_alpha = l_log_alpha_alpha;
                if self._symmetric_alpha_alpha:
                    new_log_alpha_alpha += numpy.random.random() * (r_log_alpha_alpha - l_log_alpha_alpha);
                else:
                    new_log_alpha_alpha += numpy.random.random(new_log_alpha_alpha.shape) * (r_log_alpha_alpha - l_log_alpha_alpha);
                new_alpha_alpha = numpy.exp(new_log_alpha_alpha);
                
                new_log_alpha_beta = l_log_alpha_beta;
                if self._symmetric_alpha_beta:
                    new_log_alpha_beta += numpy.random.random() * (r_log_alpha_beta - l_log_alpha_beta);
                else:
                    new_log_alpha_beta += numpy.random.random(new_log_alpha_beta.shape) * (r_log_alpha_beta - l_log_alpha_beta);                    
                new_alpha_beta = numpy.exp(new_log_alpha_beta);
                
                assert new_alpha_alpha.shape == (self._number_of_topics,)
                assert new_alpha_beta.shape == (self._number_of_types,)
                # new_hyper_parameters = [l[x] + numpy.random.random() * (r[x] - l[x]) for x in xrange(len(old_hyper_parameters))]
                # new_alpha_alpha, new_alpha_beta = [numpy.exp(x) for x in new_hyper_parameters]
                lp_test = self.log_likelihood(new_alpha_alpha, new_alpha_beta)

                if lp_test > log_likelihood_new:
                    # self._alpha_sum = self._alpha_alpha * self._number_of_topics
                    # self._beta_sum = self._alpha_beta * self._number_of_types
                    # old_hyper_parameters = [numpy.log(self._alpha_alpha), numpy.log(self._alpha_beta)]
                    # old_hyper_parameters = new_hyper_parameters;
                    
                    old_log_alpha_alpha = new_log_alpha_alpha
                    old_log_alpha_beta = new_log_alpha_beta
                    
                    self._alpha_alpha = new_alpha_alpha;
                    self._alpha_beta = new_alpha_beta;
                    
                    assert numpy.all(l_log_alpha_alpha <= old_log_alpha_alpha), (l_log_alpha_alpha, old_log_alpha_alpha);
                    assert numpy.all(r_log_alpha_alpha >= old_log_alpha_alpha), (r_log_alpha_alpha, old_log_alpha_alpha);
                    
                    break;
                else:
                    assert numpy.all(l_log_alpha_alpha <= old_log_alpha_alpha), (l_log_alpha_alpha, old_log_alpha_alpha);
                    assert numpy.all(r_log_alpha_alpha >= old_log_alpha_alpha), (r_log_alpha_alpha, old_log_alpha_alpha);
                    for dd in xrange(len(new_log_alpha_alpha)):
                        if new_log_alpha_alpha[dd] < old_log_alpha_alpha[dd]:
                            l_log_alpha_alpha[dd] = new_log_alpha_alpha[dd]
                        else:
                            r_log_alpha_alpha[dd] = new_log_alpha_alpha[dd]
                    assert numpy.all(l_log_alpha_alpha <= old_log_alpha_alpha), (l_log_alpha_alpha, old_log_alpha_alpha);
                    assert numpy.all(r_log_alpha_alpha >= old_log_alpha_alpha), (r_log_alpha_alpha, old_log_alpha_alpha);
                    
                    for dd in xrange(len(new_log_alpha_beta)):
                        if new_log_alpha_beta[dd] < old_log_alpha_beta[dd]:
                            l_log_alpha_beta[dd] = new_log_alpha_beta[dd]
                        else:
                            r_log_alpha_beta[dd] = new_log_alpha_beta[dd]
                    assert numpy.all(l_log_alpha_beta <= old_log_alpha_beta)
                    assert numpy.all(r_log_alpha_beta >= old_log_alpha_beta)
                    
                    '''
                    for dd in xrange(len(new_hyper_parameters)):
                        if new_hyper_parameters[dd] < old_hyper_parameters[dd]:
                            l[dd] = new_hyper_parameters[dd]
                        else:
                            r[dd] = new_hyper_parameters[dd]
                        assert l[dd] <= old_hyper_parameters[dd]
                        assert r[dd] >= old_hyper_parameters[dd]
                    '''

            # print("update hyper-parameters (%d, %d) to: %s %s" % (ii, jj, self._alpha_alpha, self._alpha_beta))

    def export_beta(self, exp_beta_path, top_display=-1):
        output = open(exp_beta_path, 'w');
        for topic_index in xrange(self._number_of_topics):
            output.write("==========\t%d\t%s\t==========\n" % (topic_index, self._index_to_label[topic_index]));
            
            beta_probability = self._n_kv[topic_index, :] + self._alpha_beta;
            beta_probability /= numpy.sum(beta_probability);
            
            i = 0;
            for type_index in reversed(numpy.argsort(beta_probability)):
                i += 1;
                output.write("%s\t%g\n" % (self._index_to_type[type_index], beta_probability[type_index]));
                if top_display > 0 and i >= top_display:
                    break;
                
        output.close();
        
if __name__ == "__main__":
    print "not implemented"
