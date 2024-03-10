# Fundamentals of question answering with the Language Service
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/build-faq-chatbot-qna-maker-azure-bot-service/)
### Duration: 2 Hours
---

## Course Note
- Question answering supports natural language AI workloads that require an automated conversational element.

### Azure resource in Azure AI
- Azure AI Language: includes a custom question answering feature that enables you to create a knowledge base of question and answer pairs that can be queried using natural language input.
- Azure AI Bot Service: provides a framework for developing, publishing, and managing bots on Azure.

### Step to provision question answering
- Creating a custom question answering knowledge base
  - Define questions and answers
    - 2 techniques
      - Generated from an existing FAQ document or web page.
      - Entered and edited manually.
    - Questions in the project can be assigned alternative phrasing to help consolidate questions with the same meaning.
  - Test the project
    - Save the question-and-answer pairs. This process analyzes your literal questions and answers and applies a built-in natural language processing model to match appropriate answers to questions, even when they are not phrased exactly as specified in your question definitions. 保存问题和回答组合，系统将开始分析问题和回答并进行组合。
    - test your knowledge base by submitting questions and reviewing the answers that are returned.
- Build a bot with Azure AI Bot Service
  - Create bot
  - Content channel. 渠道就是输入来源，邮件，聊天软件，视频会议软件等。
  - 
