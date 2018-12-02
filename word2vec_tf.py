import os
import tensorflow as tf
import numpy as np
import collections
import tqdm
import pickle

FLAGS=tf.flags.FLAGS

tf.flags.DEFINE_string('file','default','words file')
tf.flags.DEFINE_integer('vocab_size',50000,'the vocabulary size')
tf.flags.DEFINE_bool('chinese',True,'whether the input file is chinese')
tf.flags.DEFINE_integer('skip_window',3,'the skip window')
tf.flags.DEFINE_integer('batch_size',64,'training batch size')
tf.flags.DEFINE_integer('nce_samples',64,'the nce negative samples')
tf.flags.DEFINE_integer('embedding_size',128,'the word embedding size')
tf.flags.DEFINE_integer('max_steps', 10000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_string('save_file','default','save word embedding')


class read_datas(object):
    def __init__(self,file,vocab_size=50000,chinese=True):
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
        for i in range(0,len(self.x_train),batch_size):
            yield self.x_train[i:i+batch_size],self.y_train[i:i+batch_size]

class skip_gram(object):
    def __init__(self,batch_size=64,embedding_size=128,vocab_size=10000,nce_samples=32):
        self.batch_size=batch_size
        self.embedding_size=embedding_size
        self.vocab_size=vocab_size
        self.nce_samples=nce_samples

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
        # with tf.device('/cpu:0'):
        with tf.variable_scope('inference',reuse=tf.AUTO_REUSE):
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
        #self.test_word=tf.placeholder(tf.int32,shape=[None])
        vec_l2_model=tf.sqrt(tf.reduce_sum(tf.square(self.embedding_layer),1,keep_dims=True))
        self.norm_embedding=self.embedding_layer/vec_l2_model
        #test_word_vector=tf.nn.embedding_lookup(norm_embedding,self.test_word)
        #self.similarity=tf.matmul(test_word_vector,norm_embedding,transpose_b=True)


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
                    # validation=sess.run([self.similarity],feed_dict={self.test_word:x})
                    print("step:{}".format(step))
                    #print('validation:{}'.format(validation))
                    print('loss :{}'.format(losses))
                if (step+1)%save_every_n==0:
                    self.saver.save(sess,log_dir+'/model',global_step=step)
                if step>max_step:
                    break
            self.saver.save(sess, log_dir + '/model', global_step=step)
            self.final_embeddings=sess.run(self.norm_embedding)

def save_embeddings(file,embeddings,word2id,id2word):
    with open(file,'wb') as f:
       pickle.dump({'embedings':embeddings,'word2id':word2id,'id2word':id2word},
                   file=f,protocol=4)

def main(_):
    datas = read_datas(FLAGS.file,FLAGS.vocab_size,FLAGS.chinese)
    datas.generate_batch(FLAGS.skip_window)
    generator = datas.generate(FLAGS.batch_size)
    id2word = datas.id2word
    word2id = datas.word2id
    word2vec = skip_gram(FLAGS.batch_size,FLAGS.embedding_size,FLAGS.vocab_size,FLAGS.nce_samples)
    word2vec.train(generator,FLAGS.save_every_n,FLAGS.log_every_n,
                   log_dir='word2vec', max_step=FLAGS.max_steps)
    embeddings = word2vec.final_embeddings
    save_embeddings(FLAGS.save_file+'.pkl', embeddings, word2id, id2word)

if __name__ == '__main__':
    tf.app.run()