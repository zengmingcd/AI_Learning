# Fundamentals of Generative AI
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/fundamentals-generative-ai/)
### Duration: 2 Hours
---

## Course Note
### What is Generative AI 
- Artificial Intelligence (AI) imitates human behavior by using machine learning to interact with the environment and execute tasks without explicit directions on what to output.
- Generative AI describes a category of capabilities within AI that create original content.  生成式AI代表AI中生成原创内容的一类。
- People typically interact with generative AI that has been built into chat applications.
  - Natural language generation
  - Image generation
  - Code generation

### Large Language Model
- Generative AI applications are powered by large language models (LLMs)
- a specialized type of machine learning model that you can use to perform natural language processing (NLP) tasks.

### Transformer Model
- Transformer models are trained with large volumes of text, enabling them to represent the semantic relationships between words and use those relationships to determine probable sequences of text that make sense. 
- Transformer model architecture consists of two components, or blocks:
  - An encoder block that creates semantic representations of the training vocabulary.
  - A decoder block that generates new language sequences.
- Training a Transformer model
  - Tokenization: decompose the training text into tokens, identify each unique text value.
  - Embeddings: To create a vocabulary that encapsulates semantic relationships between the tokens, we define contextual vectors, known as embeddings, for them
    - Vectors are multi-valued numeric representations of information, each numeric element represents a particular attribute of the information.
    - For language tokens, each element of a token's vector represents some semantic attribute of the token. 
    - The specific categories for the elements of the vectors in a language model are determined during training based on how commonly words are used together or in similar contexts.
    - It can be useful to think of the elements in a token embedding vector as coordinates in multidimensional space, so that each token occupies a specific "location." The closer tokens are to one another along a particular dimension, the more semantically related they are.
  - Attention: a technique used to examine a sequence of text tokens and try to quantify the strength of the relationships between them.
    - The encoder and decoder blocks in a transformer model include multiple layers that form the neural network for the model. 
    - self-attention involves considering how other tokens around one particular token influence that token's meaning.
    - In an encoder block, attention is used to examine each token in context, and determine an appropriate encoding for its vector embedding.
      - The vector values are based on the relationship between the token and other tokens with which it frequently appears. 
      - This contextualized approach means that the same word might have multiple embeddings depending on the context in which it's used
    - In a decoder block, attention layers are used to predict the next token in a sequence.
      - For each token generated, the model has an attention layer that takes into account the sequence of tokens up to that point.
      - The model considers which of the tokens are the most influential when considering what the next token should be.
    - the attention layer is working with numeric vector representations of the tokens, not the actual text. 
    - In a decoder, the process starts with a sequence of token embeddings representing the text to be completed.
    - The first thing that happens is that another positional encoding layer adds a value to each embedding to indicate its position in the sequence.
    - During training, the goal is to predict the vector for the final token in the sequence based on the preceding tokens. 
    - The attention layer assigns a numeric weight to each token in the sequence so far. 
    - It uses that value to perform a calculation on the weighted vectors that produces an attention score that can be used to calculate a possible vector for the next token. 
    - A technique called multi-head attention uses different elements of the embeddings to calculate multiple attention scores. A neural network is then used to evaluate all possible tokens to determine the most probable token with which to continue the sequence. The process continues iteratively for each token in the sequence, with the output sequence so far being used regressively as the input for the next iteration – essentially building the output one token at a time.
    - A simplified representation of how Attention works:
      - A sequence of token embeddings is fed into the attention layer. Each token is represented as a vector of numeric values.
      - The goal in a decoder is to predict the next token in the sequence, which will also be a vector that aligns to an embedding in the model’s vocabulary.
      - The attention layer evaluates the sequence so far and assigns weights to each token to represent their relative influence on the next token.
      - The weights can be used to compute a new vector for the next token with an attention score. Multi-head attention uses different elements in the embeddings to calculate multiple alternative tokens.
      - A fully connected neural network uses the scores in the calculated vectors to predict the most probable token from the entire vocabulary.
      - The predicted output is appended to the sequence so far, which is used as the input for the next iteration.
- During training, the actual sequence of tokens is known – we just mask the ones that come later in the sequence than the token position currently being considered. 
- As in any neural network, the predicted value for the token vector is compared to the actual value of the next vector in the sequence, and the loss is calculated. 
- The weights are then incrementally adjusted to reduce the loss and improve the model. 
- When used for inferencing, the trained attention layer applies weights that predict the most probable token in the model’s vocabulary that is semantically aligned to the sequence so far.

### What is Azure OpenAI
- Azure OpenAI Service is Microsoft's cloud solution for deploying, customizing, and hosting large language models.
- It brings together the best of OpenAI's cutting edge models and APIs with the security and scalability of the Azure cloud platform.
- use an existing model as a foundational model - a starting point for further training with your own data. This approach is called fine-tuning.

### What are Copilots
- The availability of LLMs has led to the emergence of a new category of computing known as copilots.
- Copilots are often integrated into other applications and provide a way for users to get help with common tasks from a generative AI model.
- Copilots are based on a common architecture, so developers can build custom copilots for various business-specific applications and services.
- It's helpful to think of how the creation of a large language model is related to the process of creating a copilot application:
  - A large amount of data is used to train a large language model.
  - Services such as Azure OpenAI Service make pretrained models available. Developers can use these pretrained models as they are, or fine-tune them with custom data.
  - Deploying a model makes it available for use in applications.
  - Developers can build copilots that submit prompts to models and generate content for use in applications.
  - Business users can use copilots to boost their productivity and creativity with AI-generated content.

### Prompt Engineering
- The term prompt engineering describes the process of prompt improvement.
- Both developers who design applications and consumers who use those applications can improve the quality of responses from generative AI by considering prompt engineering.
- Prompts are ways we tell an application what we want it to do. An engineer can add instructions for the program with prompts.
- Prompts tips:
  - System messages
    - The message sets the context for the model by describing expectations and constraints.
  - Writing good prompts
    - You can get the most useful completions by being explicit about the kind of response you want. You can achieve better results when you submit clear, specific prompts.
  - Providing examples
    - LLMs generally support zero-shot learning in which responses can be generated without prior examples. 
    - provide one-shot learning prompts that include one, or a few, examples of the output you require. The model can then generate further responses in the same style as the examples provided in the prompt.
  - Grounding data
    - Prompts can include grounding data to provide context.
    - You can use grounding data as a prompt engineering technique to gain many of the benefits of fine-tuning without having to train a custom model.
    - To apply this technique, include contextual data in the prompt so that the model can use it to generate an appropriate output. 