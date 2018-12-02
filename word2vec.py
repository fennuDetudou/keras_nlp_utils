import numpy as np
import os
import tqdm
import collections
import pickle
import tensorflow as tf
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Embedding(object):
    def __init__(self,inputs,max_words=50000,seq_len=50,embedding_size=64,
                 fine_tuning=True,embedding_file=None,one_hot=False):
        '''
        词向量层的建立
        :param inputs: 输入文本
        :param max_words: 表示成词嵌入的最大字符数
        :param seq_len: 每次输入序列长度
        :param embedding_size: 嵌入尺寸
        :param fine_tuning: 是否进行微调
        :param embedding_file: 加载词向量文件
        :param one_hot: 是否使用one_hot向量而不是词嵌入形式
        '''
        self.inputs=inputs
        self.max_words=max_words
        self.seq_len=seq_len
        self.embedding_size=embedding_size
        self.embedding_file=embedding_file
        self.fine_tuning=fine_tuning
        self.one_hot=one_hot

        self.encode_text()
        self.text_to_array()
        self.word_index()
        self.Pad_sequence(self.sequence)

    def encode_text(self):
        '''
        将字符tokenizer化
        '''
        self.tokenizer = Tokenizer(self.max_words)
        self.tokenizer.fit_on_texts(self.inputs)

    def text_to_array(self):
        '''
        将输入文本转化为整数序列
        '''
        self.sequence=self.tokenizer.texts_to_sequences(self.inputs)
        if self.one_hot:
            self.one_hot_encode=self.tokenizer.texts_to_matrix(self.inputs)

    def word_index(self):
        '''
        返回单词与整数索引字典
        '''
        self.word_index=self.tokenizer.word_index

    def Pad_sequence(self,sequence):
        '''
        填充字符长度
        '''
        self.padd_docs=pad_sequences(sequence,self.seq_len,padding='post')

    def build_embeding(self):
        '''
        使用词嵌入层
        '''
        return layers.Embedding(input_dim=self.max_words,output_dim=self.embedding_size,
                           input_length=self.seq_len,name='embedding_size')

    def load_embedding(self):
        '''
        载入一个预先训练好的词向量，有微调和不微调两种形式
        '''
        self.embedding_index={}
        print("loading word2vec file.......")
        with open(os.path.join(os.getcwd(),self.embedding_file),'rb') as f:
            for line in f:
                values=line.strip().split()
                word=values[0]
                index=np.asarray(values[1:],np.float32)
                self.embedding_index[word]=index
        print("successfully loadding file.....")

        self.embedding_matrix=np.zeros(shape=(self.max_words,self.embedding_size))
        word_index=list(zip(self.word_index.keys(),self.word_index.values()))
        for i in tqdm.tqdm(range(len(word_index))):
            self.embedding_matrix[word_index[i][1]]=self.embedding_index.get(word_index[i][0])
        print("successfully loading  embedding.......")

        if self.fine_tuning:
            return layers.Embedding(input_dim=self.max_words, output_dim=self.embedding_size, input_length=self.seq_len,
                                 weights=[self.embedding_matrix], trainable=True)
        else:
            return layers.Embedding(input_dim=self.max_words,output_dim=self.embedding_size,input_length=self.seq_len,
                           weights=[self.embedding_matrix],trainable=False)

class read_datas(object):
    def __init__(self,file,vocab_size=50000,chinese=True):
        '''
        tensorflow tokenizer 模块以及word2vec train_batch编写
        :param file: 输入文档
        :param vocab_size: 表示成词嵌入的最大字符数
        :param chinese: 文本是否为中文
        '''
        self.file=file
        self.vocab_size=vocab_size
        self.chinese=chinese

        self.process()
        self.word_index()
        self.text_to_sequence()

    def process(self):
        print("starting process ......")
        with open(self.file,'rb') as f:
            lines=f.readlines()
        if self.chinese:
            self.lines=[line.decode('utf-8') for line in lines]
        else:
            self.lines=lines

        words=' '.join(self.lines)
        self.words=words.replace('\n','').split(' ')
        print("共有{}个单词".format(len(words)))
        # 删除words,节省内存
        del lines
        del words
        # 用列表而不是元组，元组赋值不可变
        self.count=[['UNK',0]]
        self.count.extend(collections.Counter(self.words).most_common(self.vocab_size-1))

    def word_index(self):
        print("starting word_index......")
        self.word2id=dict()
        self.id2word=dict()
        for i,word in enumerate(self.count):
            self.word2id[word[0]]=i
            self.id2word[i]=word[0]

    def text_to_sequence(self):
        print('start text_to_sequence......')
        self.sequence=[]
        for i in tqdm.tqdm(range(len(self.lines))):
            d=[]
            for word in self.lines[i]:
                if word in self.word2id:
                    d.append(self.word2id.get(word))
                else:
                    self.count[0][1]+=1
                    d.append(0)
            self.sequence.append(d)
        print("the unk number is {}".format(self.count[0][1]))

    def generate_batch(self,skip_window=3):
        '''
        返回目标词，上下文字符对
        :param skip_window: 窗口大小，返回目标词左右上下文的大小
        '''
        print('starting generate train datas......')
        x_train=[]
        y_train=[]
        for i in tqdm.tqdm(range(len(self.sequence))):
            line=self.sequence[i]
            for j in range(len(line)):
                start=j-skip_window if (j-skip_window)>=0 else 0
                end=j+skip_window if (j+skip_window)<len(line) else (len(line)-1)
                while start<=end:
                    if start==j:
                        start+=1
                        continue
                    else:
                        x_train.append(line[j])
                        y_train.append(line[start])
                        start+=1

        self.x_train=np.squeeze(np.array(x_train))
        self.y_train=np.squeeze(np.array(y_train))
        self.y_train=np.expand_dims(self.y_train,-1)

    def generate(self,batch_size):
        '''
        注意，要与skip_gram模型中的batch_size一致
        :param batch_size:
        :return:训练batch_size个数据
        '''
        for i in range(0,len(self.x_train),batch_size):
            yield self.x_train[i:i+batch_size],self.y_train[i:i+batch_size]

class skip_gram(object):
    def __init__(self,batch_size=64,embedding_size=128,vocab_size=50000,nce_samples=64):
        '''
        使用skip-gram算法训练词向量
        :param batch_size:
        :param embedding_size:
        :param vocab_size:
        :param nce_samples: 负采样个数
        '''
        self.batch_size=batch_size
        self.embedding_size=embedding_size
        self.vocab_size=vocab_size
        self.nce_samples=nce_samples

        # 搭配tf.variable_scope( )搭配使用，避免出现reuse错误
        tf.reset_default_graph()
        self.build_inputs()
        self.inference()
        self.build_loss()
        self.build_optimizer()
        self.final_embeding()

        self.saver=tf.train.Saver()

    def build_inputs(self):
        with tf.variable_scope('inputs'):
            self.x=tf.placeholder(tf.int32,shape=[self.batch_size],name='x')
            self.y=tf.placeholder(tf.int32,shape=[self.batch_size,1],name='y')

    def inference(self):
        with tf.variable_scope('inference',reuse=tf.AUTO_REUSE):
            with tf.device('/cpu:0'):
                # 定义1个embeddings变量，相当于一行存储一个词的embedding
                self.embedding_layer = tf.get_variable(shape=[self.vocab_size,self.embedding_size],
                                                       name='embeddings',initializer=tf.truncated_normal_initializer)
                # 利用embedding_lookup可以轻松得到一个batch内的所有的词嵌入
                self.embed = tf.nn.embedding_lookup(self.embedding_layer,self.x)
                # 创建两个变量用于NCE Loss（即选取噪声词的二分类损失）
                self.nce_w =tf.get_variable('nce_w',shape=[self.vocab_size,self.embedding_size],
                                            initializer=tf.truncated_normal_initializer)
                self.nce_b = tf.get_variable('nce_b',shape=[self.vocab_size])
                tf.summary.histogram('embedding',self.embedding_layer)

    def build_loss(self):
        with tf.variable_scope('loss'):
            self.loss=tf.nn.nce_loss(weights=self.nce_w,biases=self.nce_b,inputs=self.embed,labels=self.y,
                                     num_sampled=self.nce_samples,num_classes=self.vocab_size)
            self.losses=tf.reduce_mean(self.loss)
    def build_optimizer(self):
        self.optimizer=tf.train.AdamOptimizer().minimize(self.loss)

    def final_embeding(self):
        vec_l2_model=tf.sqrt(tf.reduce_sum(tf.square(self.embedding_layer),1,keep_dims=True))
        self.norm_embedding=self.embedding_layer/vec_l2_model

    def train(self,datas_generator,save_every_n,log_every_n,log_dir,max_step=5000):
        print("start training.....")
        try:
            os.mkdir(log_dir)
        except:
            pass
        self.sess=tf.Session()
        init=(tf.global_variables_initializer(),tf.local_variables_initializer())
        merged=tf.summary.merge_all()
        with self.sess as sess:
            writer = tf.summary.FileWriter(log_dir+'/tensorboard', sess.graph)
            sess.run(init)
            step=0
            for x,y in datas_generator:
                step+=1
                feed={
                    self.x:x,
                    self.y:y
                }
                losses,_,summ=sess.run([self.losses,self.optimizer,merged],
                                       feed_dict=feed)
                writer.add_summary(summ,step)
                if step %1000==0:
                    print("step {}/{} completed!".format(step,max_step))
                if (step+1)%log_every_n==0:
                    print("step:{}".format(step))
                    print('loss :{}'.format(losses))
                if (step+1)%save_every_n==0:
                    self.saver.save(sess,log_dir+'/model',global_step=step)
                if step>max_step:
                    break
            self.saver.save(sess, log_dir + '/model', global_step=step)
            self.final_embeddings=sess.run(self.norm_embedding)

def save_embeddings(file,embeddings,word2id,id2word):
    '''
    保存词向量，以及单词索引对
    '''
    with open(file,'wb') as f:
       pickle.dump({'embedings':embeddings,'word2id':word2id,'id2word':id2word},
                   file=f,protocol=4)


        

