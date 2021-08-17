# Retrieve mathBERT.
#####################################
from transformers import BertModel, BertTokenizer,AutoTokenizer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from scipy.spatial import distance
import torch
import tensorflow as tf
import pandas as pd
import compress_pickle
from os.path import join, dirname, realpath
mathbert_model = BertModel.from_pretrained("tbs17/MathBERT",output_hidden_states=True)
mathbert_tokenizer = BertTokenizer.from_pretrained("tbs17/MathBERT")

### CHECK THIS
default_grading_model_file = '{}_{}.pkl'.format('WordCountModel','base')
default_grading_model = None

# get the row in the table corresponding with the given row id
def get_by_id(table, id, as_dict=False):
    db = get_db()
    query = 'SELECT * FROM {} WHERE id = {};'.format(table, id)
    row, col = du.db_query(db, query, return_column_names=True)

    if len(row) == 0:
        return None

    if as_dict:
        res = dict()
        for c in range(len(col)):
            res[col[c]] = row[0][c]
    else:
        res = row

    return res
# returns the field names of the given table
def get_field_names(table):
    db = get_db()
    query = 'SELECT * FROM {} WHERE id = 0;'.format(table)
    _, col = du.db_query(db, query, return_column_names=True)
    return np.array(col).ravel()
# get all rows from a table
def get_all(table, as_dict=False):
    db = get_db()
    query = 'SELECT * FROM {};'.format(table)
    row, col = du.db_query(db, query, return_column_names=True)

    if len(row) == 0:
        return None

    if as_dict:
        res = dict()
        for c in range(len(col)):
            res[col[c]] = []
            for r in range(len(row)):
                res[col[c]].append(row[r][c])
    else:
        res = row

    return res
# get first row from a table where the column matches a given value
def get_first_by_column(table, column, value, as_dict=False):
    db = get_db()
    if isinstance(value,(list,)) and len(value) > 0:
        v = '('
        v += '\'{}\''.format(value[0]) if isinstance(value[0],str) else str(value[0])
        for i in range(1,len(value)):
            v += ',' + '\'{}\''.format(value[i]) if isinstance(value[i],str) else str(value[i])
        value = v + ')'
    else:
        value = '\'{}\''.format(value) if isinstance(value,str) else str(value)

    query = 'SELECT * FROM {} WHERE {} = {};'.format(table, column, value)
    try:
        row, col = du.db_query(db, query, return_column_names=True)
    except ValueError:
        return None

    if len(row) == 0:
        return None

    if as_dict:
        res = dict()
        for c in range(len(col)):
            res[col[c]] = row[0][c]
    else:
        res = row

    return res
# get all rows from a table where the column matches a given value
def get_all_by_column(table, column, value, as_dict=False):
    db = get_db()
    if isinstance(value, str):
        value = '\'{}\''.format(value)
    query = 'SELECT * FROM {} WHERE {} = {};'.format(table, column, value)
    row, col = du.db_query(db, query, return_column_names=True)

    if len(row) == 0:
        return None

    if as_dict:
        res = dict()
        for c in range(len(col)):
            res[col[c]] = []
            for r in range(len(row)):
                res[col[c]].append(row[r][c])
    else:
        res = row

    return res
# get a list of the distinct problem ids in the student_responses table
def get_distinct_problems():
    db = get_db()
    query = 'SELECT DISTINCT problem_id FROM student_responses'
    return du.db_query(db, query)
# build student_responses table using a csv file
def build_responses_table(filename=None):
    filename = 'resources/full_connected_responses.csv' if filename is None else filename
    du.db_write_from_csv(filename, get_db(), du.TableBuilder('student_responses'))
def get_directory():
    return dirname(realpath(__file__))
# get the student response data from the student_responses table (can filter by problem_id if desired)
def get_response_data(problem_id=None):
    if problem_id is None:
        pr_data = get_all('student_responses')
    else:
        pr_data = get_all_by_column('student_responses', 'problem_id', int(problem_id))

    pr_headers = get_field_names('student_responses')

    return np.array(pr_data),np.array(problem_id)

    array(pr_data), np.array(pr_headers)
class Model:
    @staticmethod
    def evaluate(modelclass, problem_id):
        data, headers = get_response_data(problem_id)

        fold_column = np.argwhere(np.array(headers) == 'folds').ravel()[0]
        folds = np.unique(data[:, fold_column])
        nfolds = len(folds)
        ret = {'auc': 0, 'rmse': 0, 'kappa': 0}

        for f in folds:
            training = np.argwhere(data[:, fold_column] != f).ravel()
            testing = np.argwhere(data[:, fold_column] == f).ravel()

            # TODO: handle case where there are not enough test cases
            # TODO: handle case where there is not enough variation in test labels
            # TODO: handle case where there are not enough training samples
            # TODO: handle case where there is not enough variation in training labels

            mod = modelclass()

            mod.train(data[training,:], headers)
            res = mod.test(data[testing,:], headers)

            # TODO: ensure that the test function returns a dictionary object with matching keys

            for k in ret.keys():
                ret[k] += res[k]

        for k in ret.keys():
            ret[k] /= nfolds

        mod = modelclass()
        mod.train(data, headers)

        ret['mod'] = modelclass.pack(mod)

        return ret

    @staticmethod
    def pack(mod, filename=None):
        return du.pickle_save(mod, filename)

    @staticmethod
    def unpack(obj):
        try:
            return du.pickle_load(obj)
        except (ValueError, AttributeError, TypeError):
            return du.pickle_load_from_file(obj)

    def load(self):
        return self

    def train(self, data, headers):
        raise NotImplementedError()

    def test(self, data, headers):
        raise NotImplementedError()

    def predict(self, data, headers):
        raise NotImplementedError()

    def classify(self, data, headers):
        raise NotImplementedError()

class Persistent:
    BERT_TRANSFORMER = BertModel.from_pretrained("tbs17/MathBERT",output_hidden_states=True)
class SBERTCanberraScoringModel(Model):
    def __init__(self, problem_id, data,embeddingList):
        """
        comment models are saved initially in this state, so they must be loaded after they are saved
        :param problem_id: problem id
        """
        self.is_loaded = False
        self.problem_id = problem_id
        self.stu_answers = None
        self.teacher_comments = None
        self.answer_scores = None
        self.ans_com_mapping = None
        self.ans_grade_mapping = None
        self.embedder = Persistent.BERT_TRANSFORMER
        self.stu_ans_embeddings = None
        self.stu_answers, unique_ind = np.unique(data['raw_answer_text'], True)
        # self.teacher_comments = data['teacher_comment'][unique_ind]
        print(self.stu_answers)
        print("data..")
        print(data)
        print()
        print("scores")
        print(data['score'])
        print()
        # print(data.loc[unique_ind])
        self.answer_scores = data['score'][unique_ind]
        # print("problem_id: ",self.problem_id)
        # print("np array: ",unique_ind)
        # print(unique_ind.tolist())
        # print()
        # print("scores")
        # print(data['score'])
        # print()
        # print(data['score'].tolist())
        # self.answer_scores = []
        # print(unique_ind.tolist())
        # for num in unique_ind.tolist():
        #     print("num: ",num)
        #     self.answer_scores.append(data.loc[num])#DOESN'T WORK

        # ans_com_mapping = {}
        # for i in range(len(self.teacher_comments)):
        #     try:
        #         ans_com_mapping[self.stu_answers[i]].append(self.teacher_comments[i])
        #     except KeyError:
        #         ans_com_mapping[self.stu_answers[i]] = [self.teacher_comments[i]]

        ans_grade_mapping = {} #NEED THIS WORK
        for i, grade in enumerate(self.answer_scores):
            ans_grade_mapping[self.stu_answers[i]] = grade

        # self.ans_com_mapping = ans_com_mapping
        self.ans_grade_mapping = ans_grade_mapping

        self.stu_ans_embeddings = embeddingList
        self.embedder = None

    def load(self):
        """
        Loads, processes, and stores data for a specific problem or all problems

        load() only needs to be run once per model. If it has already been loaded, returns None
        :return: None
        """
        if self.is_loaded:
            return None

        self.embedder = Persistent.BERT_TRANSFORMER
        self.is_loaded = True

        return self


    @staticmethod
    def pack(mod, filename=None):
        return du.pickle_save(mod, filename)

    @staticmethod
    def unpack(obj):
        try:
            return du.pickle_load(obj)
        except (ValueError, AttributeError, TypeError):
            return du.pickle_load_from_file(obj)

    def train(self, data, headers):
        raise NotImplementedError()

    def test(self, data, headers):
        raise NotImplementedError()

    def predict(self, data, headers):
        """

        :param orig_answers: list of student answers needing feedback
        :return: list
        """
        if not self.is_loaded:
            self.load()

        return_single = False
        if isinstance(data,str):
            data = [data]
            return_single = True

        # generate sentence embeddings for new answers and calculate pairwise distance with historic answers
        embeddingGenerated = get_mathbert_sentence_embedding(data)
        orig_ans_embeddings = embeddingGenerated
        distances = scipy.spatial.distance.cdist(orig_ans_embeddings,self.stu_ans_embeddings, 'canberra')

        # order the distances from most-to-least similar
        ord_distance_ind = np.argsort(distances,axis=1)

        # calculate an acceptability threshold and identify which suggestions to keep
        threshold = np.tile((np.mean(distances, axis=1) - (1.5 * np.std(distances, axis=1))).reshape((-1,1)),
                      len(self.stu_ans_embeddings))
        keep_ind = np.argwhere(np.sort(distances,axis=1) < threshold)

        # for each answer, get the teacher comments associated with the ordered indices
        similar = []
        for i in range(len(data)):
            keep = keep_ind[np.argwhere(keep_ind[:,0] == i).ravel(),1]
            scores = self.answer_scores[ord_distance_ind[i, keep]]

            similar.append(scores)

        if return_single:
            return similar[0]
        return np.array(similar).tolist()

    def classify(self, data, headers):
        return_single = False
        if isinstance(data, str):
            data = [data]
            return_single = True

        pred = self.predict(data, headers)
        scores = np.zeros((len(pred),5))
        for p in range(len(pred)):
            if len(pred[p]) == 0:
                # TODO: alter front end to recognize a non-score and leave blank
                scores[p][4] = 1
            else:
                scores[p][int(pred[p][0])] = 1

        if return_single:
            return scores[0].tolist()
        return scores.tolist()
class Path:
    TRAINED_MODEL_ROOT = join(get_directory(),'trained_model')
    TRAINED_GRADING_MODELS = join(TRAINED_MODEL_ROOT, 'scoring_models')
class SPECIAL_CASE:
    NONE = '<<SP:None>>'
    ZERO_LENGTH = '<<SP:ZeroLength>>'
    NON_STRING = '<<SP:NonString>>'
    DEBUG = '<<SP:Debug>>'
    COMMON_GUESS = '<<SP:CommonGuess>>'

    @staticmethod
    def label(text):
        # NONE
        if text is None:
            return SPECIAL_CASE.NONE
        #NON_STRING
        if not isinstance(text, str):
            return SPECIAL_CASE.NON_STRING
        #ZERO_LENGTH
        if len(text.replace(' ','')) == 0:
            return SPECIAL_CASE.ZERO_LENGTH
        # DEBUG
        if text.lower() in ['test','asdf']:
            return SPECIAL_CASE.DEBUG
        # COMMON_GUESS
        if text.lower() in ['because','guessed','idk','i don\'t know','i guessed','math','because math']:
            return SPECIAL_CASE.COMMON_GUESS
        return text

    @staticmethod
    def score(text):
        if text in [SPECIAL_CASE.NONE,
                    SPECIAL_CASE.NON_STRING,
                    SPECIAL_CASE.ZERO_LENGTH,
                    SPECIAL_CASE.DEBUG,
                    SPECIAL_CASE.COMMON_GUESS]:
            return 0
        return -1

def get_word_indeces(tokenizer, text, word):
    '''
    Determines the index or indeces of the tokens corresponding to `word`
    within `text`. `word` can consist of multiple words, e.g., "cell biology".

    Determining the indeces is tricky because words can be broken into multiple
    tokens. I've solved this with a rather roundabout approach--I replace `word`
    with the correct number of `[MASK]` tokens, and then find these in the
    tokenized result.
    '''
    # Tokenize the 'word'--it may be broken into multiple tokens or subwords.
    word_tokens = tokenizer.tokenize(word)

    # Create a sequence of `[MASK]` tokens to put in place of `word`.
    masks_str = ' '.join(['[MASK]']*len(word_tokens))

    # Replace the word with mask tokens.
    text_masked = text.replace(word, masks_str)

    # `encode` performs multiple functions:
    #   1. Tokenizes the text
    #   2. Maps the tokens to their IDs
    #   3. Adds the special [CLS] and [SEP] tokens.
    input_ids = tokenizer.encode(text_masked)

    # Use numpy's `where` function to find all indeces of the [MASK] token.
    mask_token_indeces = np.where(np.array(input_ids) == tokenizer.mask_token_id)[0]

    return mask_token_indeces
def get_embedding(b_model, b_tokenizer, text, word=''):
    '''
    Uses the provided model and tokenizer to produce an embedding for the
    provided `text`, and a "contextualized" embedding for `word`, if provided.
    '''

    # If a word is provided, figure out which tokens correspond to it.
    if not word == '':
        word_indeces = get_word_indeces(b_tokenizer, text, word)

    # Encode the text, adding the (required!) special tokens, and converting to
    # PyTorch tensors.
    encoded_dict = b_tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        return_tensors = 'pt',     # Return pytorch tensors.
                )

    input_ids = encoded_dict['input_ids']

    b_model.eval()

    # Run the text through the model and get the hidden states.
    bert_outputs = b_model(input_ids)

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():

        outputs = b_model(input_ids)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # `hidden_states` has shape [13 x 1 x <sentence length> x 768]

    # Select the embeddings from the second to last layer.
    # `token_vecs` is a tensor with shape [<sent length> x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    # Convert to numpy array.
    sentence_embedding = sentence_embedding.detach().numpy()

    # If `word` was provided, compute an embedding for those tokens.
    if not word == '':
        # Take the average of the embeddings for the tokens in `word`.
        word_embedding = torch.mean(token_vecs[word_indeces], dim=0)

        # Convert to numpy array.
        word_embedding = word_embedding.detach().numpy()

        return (sentence_embedding, word_embedding)
    else:
        # print(sentence_embedding)
        return sentence_embedding
def encoding(scores):
    one_hot_labels = tf.keras.utils.to_categorical(scores,num_classes=5)
    return one_hot_labels
def get_scored_responses_from_dataset(df,problem_id,train_folds):
    rows = df[(df['raw_answer_text'].str.strip() !="") & (df['problem_id'] == problem_id) & (df['folds'].isin(train_folds))].values
    data = dict()

    for c in range(len(df.columns)):
        if df.columns[c] == 'raw_answer_text':
            data[df.columns[c]] = np.array(rows[:, c])
        else:
            try:
                data[df.columns[c]] = np.array(rows[:, c], dtype=np.float32)
            except ValueError:
                data[df.columns[c]] = np.array(rows[:, c])
    return data
def get_mathbert_sentence_embedding(data):
    print("DATA...")
    print(data)
    word = ''
    # print(data['raw_answer_text'])
    # sentences = data['raw_answer_text'].tolist()
    embeddings = []
    #grab the sentence
    for s in range(len(data)):
        if len(data[s]) > 512:
          data[s] = data[s][0:512]
        sentence_embed = get_embedding(mathbert_model, mathbert_tokenizer, data[s], word)
        embeddings.append(sentence_embed)
    return embeddings
def pickle_save(instance, fileName=None):
    if fileName is not None:
        compress_pickle.dump(instance, open(fileName,"wb"),compression="lz4")
    else:
        return compress_pickle.dumps(instance,compression="lz4")
def getfilenames(directory='./', extension=None, exclude_directory=False):
    names = []
    directory = str(directory).replace('\\','/')
    if extension is None:
        return os.listdir(directory)

    for file in os.listdir(directory):
        if file.endswith(extension):
            if exclude_directory:
                names.append(file)
            else:
                names.append(directory + ('/' if directory[-1] != '/' else '') + file)
    return names
def pickle_load_from_file(filename):
    try:
        return compress_pickle.load(open(filename, "rb"), compression="lz4")
    except RuntimeError:
        return pickle.load(open(filename, "rb"))
def file_exists(filename, directory='./',extension=None):
    return filename in getfilenames(directory,extension)
def get_best_grading_model(problem_id, teacher_id=None):
    # debug problem id
    if int(problem_id) == 1303090:
        return join(Path.TRAINED_GRADING_MODELS, 'trained_ensemble_{}.pkl'.format(1477560))
    else:
        model = 'WordCountModel'
        if file_exists(filename='{}_{}'.format(model, problem_id),
                          directory=join(dirname(realpath(__file__)), Path.TRAINED_GRADING_MODELS),
                          extension='.pkl'):
            return '{}{}_{}.pkl'.format(Path.TRAINED_GRADING_MODELS, model, problem_id)
        else:
            return join(Path.TRAINED_GRADING_MODELS, default_grading_model_file)
def get_default_grading_model():
    global default_grading_model, default_grading_model_file
    if default_grading_model is None:
        try:
            default_grading_model = pickle_load_from_file(join(dirname(realpath(__file__)),'{}/{}'.format(Path.TRAINED_GRADING_MODELS,default_grading_model_file))).load()
        except FileNotFoundError:
            raise NotImplementedError('No default model found')
    return default_grading_model
def predict_grade(answer_text, problem_id,model_file=None, return_model_file=False):
    answers = [answer_text] if isinstance(answer_text, str) else answer_text

    answers = [SPECIAL_CASE.label(ans) for ans in answers]
    special_case_scores = np.array([SPECIAL_CASE.score(ans) for ans in answers], dtype=np.float32)

    if model_file is None:
        model_file = get_best_grading_model(problem_id)

    try:
        mod = pickle_load_from_file(join(dirname(realpath(__file__)), model_file)).load()
        if mod is None:
            raise TypeError('Model loading failed')
    except (TypeError, FileNotFoundError) as e:
        print(e)
        model_file = join(dirname(realpath(__file__)), default_grading_model_file)
        mod = get_default_grading_model()

    prediction = np.argmax(np.array(mod.classify(answers, ['cleaned_answer_text'])).reshape((-1, 5)),
                           axis=1).ravel() #stops here

    special_cases = np.argwhere(special_case_scores > -1).ravel()
    prediction[special_cases] = special_case_scores[special_cases]

    if return_model_file:
        return prediction[0] if type(answer_text) is str else prediction.tolist(), model_file
    return prediction[0] if type(answer_text) is str else prediction.tolist()

df = pd.read_csv("1_bert_full_connected_responses_org.csv")
folds = df['folds'].unique()
df = df.fillna("")

for test_fold in folds:
    #test data set
    test = df[df['folds'] == test_fold]
    #train data set
    train = df[df['folds'] != test_fold]
    # #scores in the train dataset
    # scores = train["score"].tolist()
    #train model for unique problems in the train set
    unique_problems = train['problem_id'].unique()
    #grab training folds
    train_folds = np.setdiff1d(folds,[test_fold])
    #dictionary of predicted grades and the row id as key
    predicted_grade={}
    count = 0
    for problem_id in unique_problems:
        data = get_scored_responses_from_dataset(df,problem_id,train_folds)

        if len(data['raw_answer_text']) > 0:
            sentences_text = data['raw_answer_text'].tolist()
            list_of_embeddings = get_mathbert_sentence_embedding(sentences_text) #embedding (singular list)
            mod = SBERTCanberraScoringModel(problem_id,data,list_of_embeddings)
            print(Path.TRAINED_GRADING_MODELS)
            pickle_save(mod, r'{}\{}_{}.pkl'.format(Path.TRAINED_GRADING_MODELS, 'trained_BERT_canberra', problem_id))
    #get the predictions for the test set
    for index, data_t in test.iterrows():
        did = data_t['id']
        problem_id = data_t['problem_id']
        answer_text = data_t['raw_answer_text']   #replace with cleaned_answer_text

        print(answer_text, did, problem_id)

        if answer_text.strip() == "":
            grade = 0
        else:
            #grab the model file
            model_file = r'{}\{}_{}.pkl'.format(Path.TRAINED_GRADING_MODELS, 'trained_BERT_canberra', problem_id)
            grade = predict_grade(answer_text, problem_id, model_file, False)

        predicted_grade[did] = int(grade)

        # print(did, predicted_grade, answer_text)

    df["predicted_grade_test_model"] = df['id'].map(predicted_grade)
    one_hot_encoding = encoding([predicted_grade])

    # print(predicted_grade)

    for num in range(len(one_hot_encoding)):
      for i in range(len(one_hot_encoding[num])):
        col_name = "p"+str(i)+"_test"
        df.at[index,col_name] = one_hot_encoding[num][i]
    df.to_csv('1_bert_full_connected_responses_org.csv') #change file


# del test
# del train
# df.to_csv("1_bert_full_connected_responses.csv",index=False)


#################### old version below
    # sentences_data = train["no_stop_words_text"].astype("str")
    # sentences_data = sentences_data.tolist()
    # scores = df["score"].tolist()
    # embeddingList = []
    # word=''
    #
    # # Get embeddings for each sentence
    # for num in range(len(sentences_data)):
    #   # print(sentences_data[num])
    #   if len(sentences_data[num]) > 512:
    #     sentences_data[num] = sentences_data[num][0:512]
    #
    #
    #   sen_emb = get_embedding(mathbert_model, mathbert_tokenizer, sentences_data[num],word)
    #   embeddingList.append(sen_emb)

    #for each embedding in our embeddings list

##### old version of MathBertModel this grabbed a list of sentence embeddings
# def MathBertModel(problem_id,data):
#     word = ''
#     sentences = data['raw_answer_text']
#     #sentences for problem id
#     sentence_embeddings_list = []
#     for s in range(len(sentences)):
#         sen_emb = get_embedding(mathbert_model, mathbert_tokenizer, sentences[s],word)
#         sentence_embeddings_list.append(sen_emb)
#     return sentence_embeddings_list
