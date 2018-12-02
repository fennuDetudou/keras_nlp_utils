# keras_nlp_utils
自然语言处理Keras工具包

## word2vec

快捷的使用词嵌入层的Python包，封装了几个keras embedding层的几个主要功能：

###训练词向量

* 在函数中使用词向量训练工具

```python
from word2vec import read_datas,skip_gram,save_embeddings
#加载文本处理模块以及训练数据生成模块
datas=read_datas(文件名)
datas.generate_batch()
generator=datas.generate()
#加载词向量训练模块
word2vec=skip_gram()
word2vec.train(generator,2000,1000,log_dir='words',max_step=5000)
#保存训练好的词向量
embeddings=word2vec.final_embeddings
word2id=datas.word2id
id2word=datas.id2word
save_embeddings('wiki.zh.word.pkl',embeddings,word2id,id2word)
```

* 使用脚本训练

  1. 参数定义如下：

  ```python
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
  ```

  2. 在命令行执行

     ` python word2vec_tf.py --file 'wiki.zh.word.text'  --save_file 'hello' `

### 在Keras函数式模型中使用词嵌入层

* 使用方法

```python
from word2vec import Embedding
embedding=Embedding()
x=embedding.build_embedding()(inputs)
# x=embedding.load_embedding()(inputs)
```

* Embedding()参数主要包括：

```
:param inputs: 输入文本
:param max_words: 表示成词嵌入的最大字符数
:param seq_len: 每次输入序列长度
:param embedding_size: 嵌入尺寸
:param fine_tuning: 是否进行微调
:param embedding_file: 加载词向量文件
:param one_hot: 是否使用one_hot向量而不是词嵌入形式
```

* 通过合理设置参数，可以实现：

1. 将embedding层作为神经网络的一部分进行训练`embedding.build_embedding() `
1. 使用预训练好的词向量，直接加载：`embedding.load_embedding()`
   1. 根据具体的任务进行微调（fine_tuning）
   1. 固定不变

