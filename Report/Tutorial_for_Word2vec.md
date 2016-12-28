<meta http-equiv="content-type" content="text/html; charset=UTF-8">
# 利用gensim调用word2vec模型
**Tutorial about Word2vec for IRLAB by Guoxiu in 2016/12/28**

## Python2环境配置
* 请参照[Python 廖雪峰官网](http://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/001374738150500472fd5785c194ebea336061163a8a974000)

## gensim安装
* '$ pip install --upgrade gensim'
* 具体参见[gensim Install](http://radimrehurek.com/gensim/install.html)

## 下载word2vec模型
* 常见的[模型下载列表](https://github.com/3Top/word2vec-api)，请选择合适的下载即可。
* 实验室已下载好一个wikipedia-pubmed-and-PMC-w2v.bin。

## 利用gensim调用word2vec模型
* 进入python环境：`$ python`
* 导入gensim：`import gensim`
* 设置word2vec模型路径：`word2vec_path='./wikipedia-pubmed-and-PMC-w2v.bin'`
* 加载word2vec模型：`model = gensim.models.Word2Vec.load_word2vec_format(word2vec_path, binary=True)  # C binary format`
* 获得某个词的词向量：`model['computer']`
* 获得和某个词最相似的3个词：`model.similar_by_word('computer', topn=3, restrict_vocab=None)`
* 获得某两个词的相似度：`model.similarity('woman', 'man')`
* 其他更多请参见[gensim官方API](http://radimrehurek.com/gensim/models/word2vec.html)

# 原生word2vec训练及调用等待以后补充...
