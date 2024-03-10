# Fundamentals of conversational language understanding
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/create-language-model-with-language-understanding/)
### Duration: 2 Hours
---

## Course Note
### 3 core concepts
- Utterances 语言，指令
  - An utterance is an example of something a user might say, and which your application must interpret.
- Entities 操作对象
  - An entity is an item to which an utterance refers.
- Intents 行为，动作，意图
  - An intent represents the purpose, or goal, expressed in a user's utterance.
  - The intent should be a concise way of grouping the utterance tasks. 
  - the None intent： You should consider always using the None intent to help handle utterances that do not map any of the utterances you have entered. The None intent is considered a fallback, and is typically used to provide a generic response to users when their requests don't match any other intent.

### Azure resource in Azure AI
- Azure AI Language: A resource that enables you to build apps with industry-leading natural language understanding capabilities without machine learning expertise. You can use a language resource for authoring and prediction.
- Azure AI services: A general resource that includes conversational language understanding along with many other Azure AI services. You can only use this type of resource for prediction.

### Set Up Conversational Language Understanding
- Authoring
  - created an authoring resource
  - defining the entities and intents that your application will predict as well as utterances for each intent that can be used to train the predictive model.
    - Conversational language understanding provides a comprehensive collection of prebuilt domains that include pre-defined intents and entities for common scenarios; which you can use as a starting point for your model. 
    - When you create entities and intents, you can do so in any order. 
    - You can create an intent, and select words in the sample utterances you define for it to create entities for them; or you can create the entities ahead of time and then map them to words in utterances as you're creating the intents.
  - training the model
    - the process of using your sample utterances to teach your model to match natural language expressions that a user might say to probable intents and entities.
    - test it by submitting text and reviewing the predicted intents.
      - Training and testing is an iterative process.
- Predicting
  - publish your Conversational Language Understanding application to a prediction resource for consumption
    - Client applications can use the model by connecting to the endpoint for the prediction resource, specifying the appropriate authentication key; and submit user input to get predicted intents and entities. The predictions are returned to the client application, which can then take appropriate action based on the predicted intent.

