
import os
# from os import walk

import json
import unidecode
import re
import spacy

import math
import matplotlib.pyplot as plt

from datetime import datetime
import collections
# --------------------------->



def frequency_table(items):
    fTable = dict()

    for item in items:
        fTable[item] = fTable[item] + 1 if item in fTable else 1

    return fTable

def count_text(text, ys):
    '''
    Input:
        text: a list of words
        ys: a list corresponding to the class of each text
    Output:
        result: a dictionary mapping each pair to its frequency
    '''

    result = {}
    for y, text in zip(ys, text):
        for word in text:
            # define the key, which is the word and label tuple
            pair = (word,y)
            result[pair] = result[pair] + 1 if pair in result else 1
            # if the key exists in the dictionary, increment the count
            # else, if the key is new, add it to the dictionary and set the count to 1
    return result
def sort_dict_by_values(dict, descent = True):
    return {k: v for k, v in sorted(dict.items(),reverse=descent, key=lambda item: item[1])}

def categorizeByUrl(url, dictionary):
    '''
    Input:
        url: url that will be mapped to the dictionary\n
        dictionary: ( category : possible names, ... )
    Output:
        list : all categories found in url

    Info:
        url starts after website ( 3 : '/') : standard format https://jisho.org/search/ \n
        last ocurrence of word in url will be returned


    '''

    url = url.split('/')[3:]
    l = []
    for w in url:
        for k, v in dictionary.items():
            if w in v:
                l.append(k)
    return l

def addWords(dataset,category,words,bias):
    '''
    Add words n times (bias) to each article of selected category of dataset
    Input:
        dataset: json load of the file containing array of objects (article)
        category: category to be biased with words
        words: list of words
        bias: number of times the word is appended to 'textBody' of all articles of selected category
    Output:
        result: modified dataset
    '''
    for idx, article in enumerate(dataset):
        if article['category'] == category:
            for word in words:
                  dataset[idx]['textBody'] = article['textBody'] +(' ' +word)*bias
    return dataset


class NaiveNty:
    def __init__(self,class_field='category',text_field='textBody',path = '', spacy_model = 'es_core_news_lg', posTypes = ['NOUN', 'PROPN'], stop_words_path = './assets/data/stopwords.json'):
        '''
        Args:
            - class_field: field of the objects containing the classes
            - text_field: field of the objects containing the text
            - path: path to the json file of the dataset of structure [{self.class_field : 'futbol', self.text_field : 'el real madrid gana el premio al equipo con menos pelo.'}, ...]\n
            - spacy_model: name of the spacy model to be available in install_libs and will be used to lemmatize\n
            - posTypes: a list of 'Part of Speech' acronym names from Spacy.
                - Ex: ADJ, ADV, AUX, CONJ, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X, SPACE
                - See: [about pos](https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/)
            - stop_words_path: path of the json file containing the stop word list
            - install_libs: install all necessaary libraries being used in NaiveNty\n
            - update_dataset: extract all naive == true articles from database and update the database file --> path = 'dataset.json'\n
        '''

        self.path = path
        self.posTypes = posTypes

        self.class_field = class_field
        self.text_field = text_field
        self.stop_words = []
        self.spacy_model = spacy_model

        #load spacy model
        self.nlp = spacy.load(self.spacy_model)

        #load stop words from file
        if stop_words_path:
            f = open(stop_words_path, encoding='UTF-8')
            stop_words = json.load(f)
            stop_words = ' '.join(stop_words)
            self.stop_words = self.process_text(stop_words)

        #load dataset if path exists
        if self.path:
            f = open(self.path, encoding='UTF-8')
            self.dataset = json.load(f)

        #trained dataset
        self.X = []
        self.y = []

        #model
        self.V = 0
        self.D = 0
        self.N_categories = None
        self.D_categories = None
        self.D_categories_log_probability = None

        #data
        self.train_data = []
        self.test_data = []

        #split and lemmatize dataset if path exists
        self.split_train_test() if self.path else None

     # --------------------------->


    def process_text(self, text):
        tokens = [re.split(r"([\s]+)", unidecode.unidecode(token.lemma_.lower()))[0] for token in list(filter(lambda y: y.is_alpha and y.pos_ in self.posTypes, self.nlp(text)))]
        tokens = [x for x in tokens if x not in self.stop_words] if self.stop_words else tokens
        return tokens


    def split_train_test(self, ratio_split = 8/9):
        '''
        Input:
            ratio_split: proportion of instances of each category for training data, the rest will be used to test the accuracy
        Info:
            Prune training and test data.
            Process training data text (nlp) and categories (remove punctuation, strip and lower case).

        '''
        # Opening JSON file (datasetfrompy in get dataset)
        print('dataset length ------------------> ',len(self.dataset))
        #discard texts with 10 or less chars and without text_field
        pruned_docs = list(filter(lambda x: x.get(self.text_field)!=None and len(x.get(self.text_field))>10 ,self.dataset[:]))
        print('Documentos con texto: ', len(pruned_docs))

        train_data = []
        test_data = []
        #split each category in train_data and test_data by ratio
        [train_data.append(x) for x in pruned_docs if len(list(filter(lambda y: y[self.class_field] == x[self.class_field],train_data))) < len(list(filter(lambda z: z[self.class_field] == x[self.class_field],pruned_docs)))*ratio_split]
        [test_data.append(x) for x in pruned_docs if x not in train_data]

        print('len(train_data) --->',len(train_data))
        print('len(test_data) --->',len(test_data))

        self.y = [unidecode.unidecode(a.get(self.class_field)).lower().strip() for a in train_data]
        self.X = [self.process_text(a.get(self.text_field)) for a in train_data]

        self.train_data = train_data
        self.test_data = test_data

        print('self.test_data ---> ',self.test_data[0])
        print('self.X ---> ',self.X[0])


    ## Naive Bayes ------------------------------------------------------------------------------------------------------------------------------------->

    def train(self):
        '''
        Input:
            freqs: dictionary from (word, label) to how often the word appears
            train_x: a list of texts
            train_y: a list of categories correponding to the texts ('deportes','economia')
        Output:
            V: Number of unique words in the vocabulary
            N_categories: (freqTable) a list containing a dict of key = category and value a dict containing the frequency of each word associated with it
            D: Number of documents
            D_categories: dictionary containing the number of texts of each category
            D_categories_log_probability: dictionary containing the log prior probability of each category ---> log(number of texts of each category / total texts) or
        '''

        train_y = self.y
        freqs = count_text(self.X, train_y)
        # train_y = self.y

        # calculate V, the number of unique words in the vocabulary
        vocab = set([pair[0] for pair in freqs.keys()])

        V = len(vocab)
        # calculate n, the frequency of words per category
        N_categories = dict()

        for pair in freqs.keys():
            if pair[1] in N_categories:
                if pair[0] in N_categories[pair[1]]:
                    N_categories[pair[1]][pair[0]] += freqs[pair]
                else:
                    N_categories[pair[1]].update({pair[0]:freqs[pair]})
            else:
                N_categories.update({pair[1]:{pair[0]:freqs[pair]}})

        # Calculate D, the number of documents
        D = len(train_y)

        #create dictionary containing the number of texts of each category
        D_categories = frequency_table(train_y)

        #create dictionary containing the log prior probability of each category
        D_categories_log_probability = dict()
        for cat in D_categories:
            D_categories_log_probability[cat] = math.log(D_categories[cat]) - math.log(D)

        self.V = V
        self.N_categories = N_categories
        self.D = D
        self.D_categories = sort_dict_by_values(D_categories)
        self.D_categories_log_probability = sort_dict_by_values(D_categories_log_probability)

        for c in N_categories:
            N_categories[c] = sort_dict_by_values(N_categories[c])

        return V, N_categories, D, D_categories, D_categories_log_probability




    # Compute token probability for a given a category, laplace smoothing is used to avoid the 'black swan', consider lidstone < 1
    def tokenProbability(self,token, category):

    #number of times this word appeared in documents mapped to this category

            wordFrequencyCount = 0
            try:
                category = self.N_categories[category]
                wordFrequencyCount = category[token]
            except:
                wordFrequencyCount = 0

    #count of all words mapped to this category
            wordCount = 0
            try:
                wordCount = len(category)
                # wordCount = 0
            except:
                wordCount = 0
            # print('sum(len(category.keys()) ---->',len(category.keys()),'wordFrequencyCount ---> ',wordFrequencyCount,'wordCount -----> ',wordCount)
    #use laplace Add-1 Smoothing equation

            # print('wordFrequencyCount -->',wordFrequencyCount,'wordCount --->', wordCount)
            # print('tokenProb wordPrior --->',(wordFrequencyCount + 1) / (wordCount + self.V),'tokenProb no wrod prior --->',(wordFrequencyCount + 1) / (self.V))
            return (wordFrequencyCount + 1) / (wordCount + self.V)




    def categorize(self,text,safe_categorization = False,similarity_threshold = 0.08):
        '''
        Input:
            text: str
            safe_categorization: restrict categorization results based on similarity_threshold
            similarity_threshold: minimum % of 'similarity' of a text on its highest scored category
        Output:
            result: an descendent ordered dictionary containing keys as category name and value as the percentage of 'similarity' of the category
            safe: boolean value indicating if the restrictions of safe_categorization were met
        '''
        safe = False
        tokens = self.process_text(text)
        maxProbability = float('-inf')
        categoryProbs = dict()
        fTable = frequency_table(tokens)
        std = lambda x,ddof = 1: ((sum([abs( (y - (sum(x)/len(x))))**2 for y in x]) / (len(x)-ddof) ))**0.5


        for cat in self.D_categories:
            #prior probability
            #category_probability = D_categories[cat]/ D
            #since log is a monotonic function we use log probabilities to avoid underflow
            category_probability_log = self.D_categories_log_probability[cat]
            #precalculation of D_categories_log_probability dictionary

            #compute probability of each class
            for token in fTable:
                tokenFreq = fTable[token]
                tokenProb = self.tokenProbability(token, cat)
                category_probability_log += tokenFreq * math.log(tokenProb)
                # print('category -->',cat,'token -->',token,'tokens len -->',len(tokens),'category_probability_log_word_sum 1 freq all words --->',(math.log(1/(len(self.N_categories[cat]) + self.V))*len(tokens))+category_probability_log,'token freq -->',tokenFreq,'tokenProb -->',tokenProb,'tokenProb log -->',math.log(tokenProb),'category_probability_log_word_sum --->',category_probability_log)
            # print('category -->',cat,'all words min log sum--->',((math.log(1/(len(self.N_categories[cat]) + self.V))*len(tokens))),'all words log sum --->',category_probability_log)
            categoryProbs[cat] = category_probability_log

            if (category_probability_log > maxProbability):
                maxProbability = category_probability_log
            result = sort_dict_by_values(categoryProbs)
        if safe_categorization:
            for cat in result:
                result[cat] = [result[cat],(math.log(1/(len(self.N_categories[cat]) + self.V))*len(tokens))+self.D_categories_log_probability[cat]]
            distance_all = sort_dict_by_values({cat : (abs(result[cat][0] - result[cat][1])) / abs(result[cat][0]) for cat in result})
            first_item = distance_all[next(iter(distance_all))]

            standard_dev_all = std([distance_all[cat] for cat in distance_all])
            # print('first item --->',distance_all,'minimum_threshold --->',similarity_threshold,'distance_all[internacional] -->',distance_all['internacional'],'std distance all --> ',standard_dev_all,'scaled --> ',[distance_all[x] / standard_dev_all for x in distance_all],'first item -->',first_item)
            safe = False if first_item < similarity_threshold else True

            result = {cat: distance_all[cat] for cat in distance_all if (first_item-distance_all[cat])<standard_dev_all}

        return result,safe



    # def computeSimilarity(self,text,result,minimum_threshold = 0.1,category_threshold = 0.1):
    #     tokens = self.process_text(text)
    #     safe = None
    #     #append the minimum probability to each category in result
    #     for cat in result:
    #         result[cat] = [result[cat],(math.log(1/(len(self.N_categories[cat]) + self.V))*len(tokens))+self.D_categories_log_probability[cat]]
    #     #distance of each category from the minimum probability of the tokens
    #     distance_all = sort_dict_by_values({cat : (abs(result[cat][0] - result[cat][1])) / abs(result[cat][0]) for cat in result})
    #     std = lambda x,ddof = 0: ((sum([abs( (y - (sum(x)/len(x))))**2 for y in x]) / (len(x)-ddof) ))**0.5
    #     first_item = distance_all[next(iter(distance_all))]

    #     standard_dev_all = std([distance_all[cat] for cat in distance_all])

    #     print('first item --->',distance_all,'minimum_threshold --->',minimum_threshold,'category_threshold --->',category_threshold,'distance_all[internacional] -->',distance_all['internacional'],'std distance all --> ',standard_dev_all,'scaled --> ',[distance_all[x] / standard_dev_all for x in distance_all],'first item -->',first_item)
    #     safe = False if first_item < minimum_threshold else True
    #     return {cat: distance_all[cat] for cat in distance_all if distance_all[cat] > first_item-(category_threshold*first_item)}

        #select items taking into account a fraction of the first item
        # selected_items = {cat: result[cat][0] for cat in result if first_item }



        # distance_all = sort_dict_by_values({cat : abs(result[cat][0] - result[cat][1]) for cat in result})
        # first_item = distance_all[next(iter(distance_all))]
        # threshold = threshold*first_item
        # print('first item --->',first_item,'threshold --->',threshold,'distance_all[internacional] -->',distance_all['internacional'])
        # #select items taking into account a fraction of the first item
        # selected_items = {cat: result[cat][0] for cat in result if distance_all[cat] > first_item - threshold}
        return distance_all


        # priorSimilarity = True if
        # (wordFrequencyCount + 1) / (wordCount + self.V)
        pass
    ##accuracy tests --------------------------------------------------------->

    def plot_freqs(self,plot_path = './naive_categories_plot_freq/'+str(datetime.now().isoformat())+'/',type = 'images', max_words = 30):

        '''
        Exports graphics representing the weights some labels of each category in the current trained dataset.
        *Do not use this method without training or importing your model in this naive instance*

            - `plot_path = './naive_categories_plot_freq/'`: string path to use to export the graphs
            - `type = 'images' || 'json'`: specify the kind of output to generate
            - `max_words = 30`: The number of labels to be represented in each graph for each category

        Returns: A dictionary whose keys are the different category names and the value consists of an array of ordered wheights of words for the category in the trained model
        '''

        dt = self.N_categories

        def plot_freq(v,l,title,save_folder = './graphs/'):
            plt.gcf().set_size_inches(30, 5)
            plt.bar(range(l), list(v.values())[:l])
            plt.xticks(range(l), list(v.keys())[:l])
            plt.savefig(str(save_folder + title + '.png'))
            plt.close()

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        cats_wordfreq_ordered = {}

        for c in dt.keys():
            values = sorted(dt[c].items(), key=lambda x: x[1], reverse=True)
            weights_ordered = {key: value for (key, value) in values}
            if type == 'images':
                plot_freq(weights_ordered,max_words,c,plot_path)

            cats_wordfreq_ordered[c] = weights_ordered

        if(type == 'json'):
            serialized = json.dumps(cats_wordfreq_ordered, indent=1)

            with open(str(plot_path+'wordFreqs.json'), mode='w', encoding='UTF-8') as file_:
                file_.write(serialized)
                file_.close()

        return cats_wordfreq_ordered


    def test(self, test_name, export = True, n_categories=2, keywords=None, keywords_bias=None , text_field=None):
        '''
        Test the model. Perform a complete test with the already trained model.
        Input:
            - ``test_ name``: The name of the test to be created, this will be used when exporting the model test info.
            - ``export``: Flag to indicate wether or not to export the test
            - ``n_categories``: Number of classes with more probability ordered DESC accepted
            - ``keywords``: a document indicating each categories keywords in one of the following formats:
                dict[str, list[str]] - An array of same wheigthed keywords
                dict[str, list[dict[str, Number]]] - An array of dictionaries conaining each keyword string and wheight value '{name: wheight}'
        '''

        if keywords:
            print('updating keywords')
            self.update_weight_categories(keywords, keywords_bias)

        acc = self.compute_accuracy(n_categories,text_field)['total_accuracy']
        today = datetime.now().isoformat(timespec='minutes')

        folder = test_name if test_name else datetime.now().isoformat(timespec='seconds')
        folder = folder if folder.endswith('/') else folder + '/'
        folder = 'tests/' + folder + str(acc) + '_' + str(today) + '/'

        models_path = folder+'model/'
        graphs_path = folder+'graphs/'

        if not os.path.exists(models_path):
            os.makedirs(models_path)

        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)
        print('testing done.')
        self.export(models_path) and self.plot_freqs(graphs_path) if export else None


    def compute_accuracy(self,n_categories, text_field=None,X = False,y = False, verbose=False):

        '''
        Test the model accuracy against self test or external data
        Input:
            n_categories: Number of classes with more probability ordered DESC accepted
            text_field (optional): Name of the field containing the text in X object array
            X (optional): Array of documents
            y (optional): Classes
        '''

        categories_naive_accuracy_test = []
        categories_accuracy_test = []
        categories_prob_naive_accuracy_test = []
        correct = 0
        hit = []
        missed = []

        if not text_field:
            text_field = self.text_field

        if self.train_data and X and y:
            X = [x for x in X if x not in self.train_data]

        if not X and not y:
            y = [unidecode.unidecode(a.get(self.class_field)).lower().strip() for a in self.test_data]
            # text_field = self.text_field
            # processing not needed (already in categorize)
            X = self.test_data


        for idx, article in enumerate(X):
            result = self.categorize(article[text_field])

            current_categories_prob_naive = {k: result[k] for k in list(result.keys())[:n_categories]}
            print('current_categories_prob_naive', current_categories_prob_naive) if verbose else None
            # for idx, cat in enumerate(current_categories_prob_naive):
            #     if idx == n_categories:
            #         break
            categories_prob_naive_accuracy_test.append(current_categories_prob_naive)
            # print('categories_prob_naive_accuracy_test ------>', categories_prob_naive_accuracy_test)
            current_categories_naive = [cat for cat in current_categories_prob_naive]
            categories_naive_accuracy_test.append(current_categories_naive)

            current_category = y[idx]
            categories_accuracy_test.append(current_category)
            print('current_category_naive ------------------>',current_categories_naive) if verbose else None
            print('current_category -------------------->',current_category, '\r\n') if verbose else None
            # print('article id ---->',article['_id'],'\r\n')
            print('text hint -------------------->',article[text_field][:50].strip(), '...\r\n') if verbose else None

            if current_category in current_categories_naive:
                correct += 1
                hit.append(current_category)
            else:
                missed.append(current_category)
            print(correct,'<----------------- \r\n',idx, '<------------------- total\r\n') if verbose else None

        hit = sort_dict_by_values(frequency_table(hit))
        missed = sort_dict_by_values(frequency_table(missed))
        print('categories_accuracy_test ---->', categories_accuracy_test,' len categories_accuracy_test ---->', len(categories_accuracy_test)) if verbose else None
        print('categories_prob_naive_accuracy_test ---->', categories_prob_naive_accuracy_test,'len categories_prob_naive_accuracy_test ---->', len(categories_prob_naive_accuracy_test)) if verbose else None

        acc_by_category = sort_dict_by_values({k : (str(hit[k])+'/'+str(hit[k] + missed[k])) for k in hit if k in missed} | {k : (str(hit[k])+'/'+str(hit[k])) for k in hit if k not in missed} | {k : ('0'+'/'+str(missed[k])) for k in missed if k not in hit})
        ground_predicted = dict()
        for idx in range(len(categories_accuracy_test)):
            print('idx -->',idx,'categories_accuracy_test[idx] -->',categories_accuracy_test[idx],'categories_prob_naive_accuracy_test[idx] -->',categories_prob_naive_accuracy_test[idx])
            ground_predicted.update({categories_accuracy_test[idx] : categories_prob_naive_accuracy_test[idx]})
        # ground_predicted = {categories_accuracy_test[idx] : categories_prob_naive_accuracy_test[idx] for idx in range(len(list(missed)+list(hit)))}
        print('len ground_predicted',ground_predicted)
        print('ground_predicted ---->', ground_predicted,'len ground_predicted ---->', len(ground_predicted)) if verbose else None
        accuracy_test = {'total_accuracy': correct / idx, 'total_accuracy_str': str(correct) + '/' + str(idx) ,'accuracy_by_categories': acc_by_category, 'ground_predicted' : ground_predicted,'hit': hit, 'missed': missed}
        self.accuracy_test = accuracy_test

        return accuracy_test


    ## --------------------------------------------------------->

    def update_word_weight(self, word,category,bias = 0.5, verbose=True):

        print('Updating \''+word+'\' in ' + category) if verbose else None

        if(len(word) == 0):
            return

        word = self.process_text(word)

        if(word is None or len(word) == 0):
            return
        else:
            word = word[0]

        n_cat = self.N_categories
        if category in n_cat:
            freq_dict = n_cat[category]
        elif category.lower() in n_cat:
            category = category.lower()
            freq_dict = n_cat[category]
        else:
            return

        if(freq_dict is None or len(freq_dict) == 0):
            return


        found = word in freq_dict

        old_weight = freq_dict[word] if word in freq_dict else 'None'

        #if word is in dictionary distance is the interval from the word to max, else from mean to max
        distance = max(freq_dict.values()) - freq_dict[word] if word in freq_dict else max(freq_dict.values()) - (sum(freq_dict.values())/len(freq_dict.values()))
        #if word is in dictionary new weight is the former word freq value + bias * distance, else mean + bias * distance
        new_weight = freq_dict[word] + (bias * distance) if word in freq_dict else (sum(freq_dict.values())/len(freq_dict.values())) + (bias * distance)
        #update the word in the model
        self.N_categories[category][word] = new_weight

        print(('Modified \'' if found else 'Added \'') + word + '\' distance = ['+str(distance)+'] weight = [ old = '+ str(old_weight) +';  new = '+str(new_weight) + ']') if verbose else None


    def update_weight_categories(self,dictionary,default_bias = 0.5):

        '''
        Input:
            n_categories: a document indicating each categories keywords in one of the following formats:
                dict[str, list[str]] - An array of same wheigthed keywords
                dict[str, list[dict[str, Number]]] - An array of dictionaries conaining each keyword string and wheight value '{name: wheight}'
        '''

        if type(list(dictionary.values())[0]) is dict:
            for c in dictionary:
                for i in dictionary[c]:
                    self.update_word_weight(i,c,dictionary[c][i])
        else:
            for c in dictionary:
                print(str(len(dictionary[c])) + ' keywords for category: ' + str(c))
                for i in dictionary[c]:
                    self.update_word_weight(i,c,default_bias)

    def export(self, export_path = ''):
        # serialize
        serialize_naive = json.dumps({
            'N_categories' : self.N_categories,
            'V' : self.V,
            'D' : self.D,
            'D_categories' : self.D_categories,
            'D_categories_log_probability' : self.D_categories_log_probability,
            'Accuracy_test' : self.accuracy_test,
            'data' : [self.train_data,self.test_data]
            # 'test_data' : self.test_data
            })
        if export_path:
            with open(str(export_path+str(datetime.now().isoformat())+'_'+str(self.accuracy_test['total_accuracy'])+'.json'), mode='w', encoding='UTF-8') as file_:
                file_.write(serialize_naive)
                file_.close()
        return serialize_naive

    def import_model(self, import_path = ''):
        if import_path:
            with open(import_path) as json_file:
                data = json.load(json_file)
                # print('data: ', data)
                self.N_categories = data['N_categories']
                self.V = data['V']
                self.D = data['D']
                self.D_categories = data['D_categories']
                self.D_categories_log_probability = data['D_categories_log_probability']
                self.accuracy_test = data['Accuracy_test']
                self.train_data = data['data'][0]
                self.test_data = data['data'][1]
                # self.test_data = data['test_data']

                # Print the type of data variable
                # print("Model imported.")
