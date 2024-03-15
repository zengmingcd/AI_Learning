# Fundamentals of Azure OpenAI Service
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/explore-azure-openai/)
### Duration: 4 Hours
---

## Course Note
### Microsoft has partnered with OpenAI to deliver on three main goals:
- To utilize Azure's infrastructure, including security, compliance, and regional availability, to help users build enterprise-grade applications.
- To deploy OpenAI AI model capabilities across Microsoft products, including and beyond Azure AI products.
- To use Azure to power all of OpenAI's workloads.

---
- Azure OpenAI Service is a result of the partnership between Microsoft and OpenAI. The service combines Azure's enterprise-grade capabilities with OpenAI's generative AI model capabilities.

### Azure OpenAI is available for Azure users and consists of four components:
- Pre-trained generative AI models
- Customization capabilities; the ability to fine-tune AI models with your own data
- Built-in tools to detect and mitigate harmful use cases so users can implement AI responsibly
- Enterprise-grade security with role-based access control (RBAC) and private networks

### Azure OpenAI supports generative AI workloads:
- Generating Natural Language
  - Text completion: generate and edit text
  - Embeddings: search, classify, and compare text
- Generating Code: generate, edit, and explain code
- Generating Images: generate and edit images


### Azure OpenAI's relationship to Azure AI services
- Azure AI services are tools for solving AI workloads.
- Azure OpenAI Service may be more beneficial for use-cases that require highly customized generative models, or for exploratory research.

### How to use Azure OpenAI
- Azure OpenAI's capabilities are made possible by specific generative AI models. Different models are optimized for different tasks; some models excel at summarization and providing general unstructured responses, and others are built to generate code or unique images from text input.
- These Azure OpenAI models include:
  - GPT-4 models that represent the latest generative models for natural language and code.
  - GPT-3.5 models that can generate natural language and code responses based on prompts.
  - Embeddings models that convert text to numeric vectors for analysis - for example comparing sources of text for similarity.
  - DALL-E models that generate images based on natural language descriptions.
- Generative AI models always have a probability of reflecting true values. Higher performing models, such as models that have been fine-tuned for specific tasks, do a better job of returning responses that reflect true values. It is important to review the output of generative AI models.
- Playground
  - Completions playground: type in prompts, configure parameters, and see responses without having to code.
  - Chat playground: use the assistant setup to instruct the model about how it should behave. The assistant will try to mimic the responses you include in tone, rules, and format you've defined in your system message.

### Azure OpenAI's natural language capability
- Azure OpenAI's natural language models are able to take in natural language and generate responses.
- Natural language learning models are trained on words or chunks of characters known as tokens. 
- A key aspect of OpenAI's generative AI is that it takes an input, or prompt, to return a natural language, visual, or code response. GPT tries to infer, or guess, the context of the user's question based on the prompt.

### Azure OpenAI's code generation capability
- GPT models are able to take natural language or code snippets and translate them into code. 
- GPT models have been trained on both natural language and billions of lines of code from public repositories. The models are able to generate code from natural language instructions such as code comments, and can suggest ways to complete code functions.
- GPT models can help developers code faster, understand new coding languages, and focus on solving bigger problems in their application. Developers can break down their goals into simpler tasks and use GPT to help build out those tasks using known patterns.
- GPT can also summarize functions that are already written, explain SQL queries or tables, and convert a function from one programming language into another.
- When interacting with GPT models, you can specify libraries or language specific tags to make it clear to Codex what we want. 
- OpenAI partnered with GitHub to create GitHub Copilot, which they call an AI pair programmer.
  - Once the plugin is installed and enabled, you can start writing your code, and GitHub Copilot starts automatically suggesting the remainder of the function based on code comments or the function name.安装并启用插件后，您就可以开始编写代码，GitHub Copilot 会开始根据代码注释或函数名称自动建议函数的其余部分。
  - GitHub Copilot offers multiple suggestions for code completion, which you can tab through using keyboard shortcuts. When given informative code comments, it can even suggest a function name along with the complete function code. GitHub Copilot 提供了多种代码补全建议，您可以使用键盘快捷键进行选项卡浏览。当给出信息丰富的代码注释时，它甚至可以建议函数名称以及完整的函数代码。

### Azure OpenAI's image generation capability
- Image generation models can take a prompt, a base image, or both, and create something new. These generative AI models can create both realistic and artistic images, change the layout or style of an image, and create variations on a provided image.

### DALL-E
- The model that works with images is called DALL-E.
- Image capabilities generally fall into the three categories of image creation, editing an image, and creating variations of an image.
- Image generation
  - Original images can be generated by providing a text prompt of what you would like the image to be of. The more detailed the prompt, the more likely the model will provide a desired result.
- Editing an image
  - When provided an image, DALL-E can edit the image as requested by changing its style, adding or removing items, or generating new content to add. Edits are made by uploading the original image and specifying a transparent mask that indicates what area of the image to edit. Along with the image and mask, a prompt indicating what is to be edited instructs the model to then generate the appropriate content to fill the area.
- Image variations
  - Image variations can be created by providing an image and specifying how many variations of the image you would like. The general content of the image will stay the same, but aspects will be adjusted such as where subjects are located or looking, background scene, and colors may change.

### responsible AI policies
- It's important to consider the ethical implications of working with AI systems. 
- Six Microsoft AI principles:
  - Fairness: AI systems shouldn't make decisions that discriminate against or support bias of a group or individual.
  - Reliability and Safety: AI systems should respond safely to new situations and potential manipulation.
  - Privacy and Security: AI systems should be secure and respect data privacy.
  - Inclusiveness: AI systems should empower everyone and engage people.
  - Accountability: People must be accountable for how AI systems operate.
  - Transparency: AI systems should have explanations so users can understand how they're built and used.
- Transparency Notes are intended to help you understand how Microsoft's AI technology works, the choices system owners can make that influence system performance and behavior, and the importance of thinking about the whole system, including the technology, the people, and the environment.
