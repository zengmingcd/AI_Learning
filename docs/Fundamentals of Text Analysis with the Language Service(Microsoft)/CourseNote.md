# Fundamentals of optical character recognition
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/analyze-text-with-text-analytics-service/)
### Duration: 2 Hours
---

## Course Note

- Natural Language Processing (NLP), an area within AI that deals with understanding written or spoken language, and responding in kind.
- Text analysis describes NLP processes that extract information from unstructured text.
- Azure AI Language is a cloud-based service that includes features for understanding and analyzing text.

### Tokenization Token化
- The first step in analyzing a corpus is to break it down into tokens.
- Concepts that may apply to tokenization depending on the specific kind of NLP problem you're trying to solve:
  - Text normalization: Before generating tokens, you may choose to normalize the text by removing punctuation and changing all words to lower case. For analysis that relies purely on word frequency, this approach improves overall performance. However, some semantic meaning may be lost - for example, consider the sentence "Mr Banks has worked in many banks.". You may want your analysis to differentiate between the person Mr Banks and the banks in which he has worked. You may also want to consider "banks." as a separate token to "banks" because the inclusion of a period provides the information that the word comes at the end of a sentence. 规范化文本，在生成token前，将标点去掉，全部改为小写等方式对文本进行规范化，以提高整体表现。但是会丢失一些语义信息。
  - Stop word removal. Stop words are words that should be excluded from the analysis. For example, "the", "a", or "it" make text easier for people to read but add little semantic meaning. By excluding these words, a text analysis solution may be better able to identify the important words. 删除停止词。这些词没有语义，删除以让模型专注于识别重要词汇。
  - n-grams are multi-term phrases such as "I have" or "he walked". A single word phrase is a unigram, a two-word phrase is a bi-gram, a three-word phrase is a tri-gram, and so on. By considering words as groups, a machine learning model can make better sense of the text. 多术语短语，多个词组合表达一个意思。将这些词组合为一个词组进行分析，让模型更好理解。
  - Stemming is a technique in which algorithms are applied to consolidate words before counting them, so that words with the same root, like "power", "powered", and "powerful", are interpreted as being the same token.词干提取。将相同词干的词使用相同token表示。

### Frequency analysis
- After tokenizing the words, you can perform some analysis to count the number of occurrences of each token. 将字token化后，分析每个token出现次数。
- The most commonly used words can often provide a clue as to the main subject of a text corpus. 常用词可以提供语料库主要主题的线索。
- Simple frequency analysis in which you simply count the number of occurrences of each token can be an effective way to analyze a single document
- Term frequency - inverse document frequency (TF-IDF) is a common technique in which a score is calculated based on how often a word or term appears in one document compared to its more general frequency across the entire collection of documents.

### Machine learning for text classification
- use a classification algorithm, such as logistic regression, to train a machine learning model that classifies text based on a known set of categorizations.


### Semantic language models
- encoding of language tokens as vectors (multi-valued arrays of numbers) known as embeddings. 嵌入：将token编码到向量中。
- It can be useful to think of the elements in a token embedding vector as coordinates in multidimensional space, so that each token occupies a specific "location." The closer tokens are to one another along a particular dimension, the more semantically related they are. In other words, related words are grouped closer together.

### Application
- Common NLP tasks supported by language models include:
  - Text analysis 文本分析, such as extracting key terms or identifying named entities in text.
  - Sentiment analysis and opinion mining to categorize text as positive or negative. 情感分析及意见挖掘
  - Machine translation, in which text is automatically translated from one language to another. 机器翻译
  - Summarization, in which the main points of a large body of text are summarized. 摘要
  - Conversational AI solutions such as bots or digital assistants in which the language model can interpret natural language input and return an appropriate response. 对话式人工智能解决方案。

- Azure AI Language's text analysis features include:
  - Named entity recognition identifies people, places, events, and more. This feature can also be customized to extract custom categories.
  - Entity linking identifies known entities together with a link to Wikipedia.
  - Personal identifying information (PII) detection identifies personally sensitive information, including personal health information (PHI).
  - Language detection identifies the language of the text and returns a language code such as "en" for English.
  - Sentiment analysis and opinion mining identifies whether text is positive or negative.
  - Summarization summarizes text by identifying the most important information.
  - Key phrase extraction lists the main concepts from unstructured text.


### Text Analysis in Azure AI
- Entity recognition and linking 实体识别与链接
  - Provide Azure AI Language with unstructured text and it will return a list of entities in the text that it recognizes.
  - supports entity linking to help disambiguate entities by linking to a specific reference.
- Language detection 语言检测
  - Use the language detection capability of Azure AI Language to identify the language in which text is written
  - The language detection service will focus on the predominant language in the text. 
- Sentiment analysis and opinion mining 情绪分析及意见挖掘
  - The text analytics capabilities in Azure AI Language can evaluate text and return sentiment scores and labels for each sentence. This capability is useful for detecting positive and negative sentiment in social media, customer reviews, discussion forums and more.Azure AI 语言中的文本分析功能可以评估文本并返回每个句子的情感分数和标签。此功能对于检测社交媒体、客户评论、论坛等中的积极和消极情绪非常有用。
  - Azure AI Language uses a prebuilt machine learning classification model to evaluate the text. The service returns sentiment scores in three categories: positive, neutral, and negative. In each of the categories, a score between 0 and 1 is provided. Scores indicate how likely the provided text is a particular sentiment. One document sentiment is also provided.Azure AI 语言使用预构建的机器学习分类模型来评估文本。该服务返回三类情绪分数：积极、中性和消极。在每个类别中，提供 0 到 1 之间的分数。分数表明所提供的文本是特定情绪的可能性有多大。还提供了一份文档情绪。
- Key phrase extraction 关键词提取
  - Key phrase extraction identifies the main points from text. 

### Azure resource in Azure AI
- A Language resource - choose this resource type if you only plan to use Azure AI Language services, or if you want to manage access and billing for the resource separately from other services.
- An Azure AI services resource - choose this resource type if you plan to use Azure AI Language in combination with other Azure AI services, and you want to manage access and billing for these services together.