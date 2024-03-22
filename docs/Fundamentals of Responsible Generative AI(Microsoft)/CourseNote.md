# Fundamentals of Responsible Generative AI
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/responsible-generative-ai/)
### Duration: 2 Hours
---

## Course Note
### Plan a responsible generative AI solution
- Identify potential harms that are relevant to your planned solution.
  - Identify potential harms
    - The potential harms that are relevant to your generative AI solution depend on multiple factors, including the specific services and models used to generate output as well as any fine-tuning or grounding data used to customize the outputs.
    - common types of potential harm in a generative AI solution include:
      - Generating content that is offensive, pejorative, or discriminatory.
      - Generating content that contains factual inaccuracies.
      - Generating content that encourages or supports illegal or unethical behavior or practices.
  - Prioritize identified harms
    - For each potential harm you have identified, assess the likelihood of its occurrence and the resulting level of impact if it does. Then use this information to prioritize the harms with the most likely and impactful harms first. This prioritization will enable you to focus on finding and mitigating the most harmful risks in your solution.
    - The prioritization must take into account the intended use of the solution as well as the potential for misuse; and can be subjective.优先级必须考虑解决方案的预期用途以及误用的可能性；并且可以是主观的。
  - Test and verify the prioritized harms
    - test your solution to verify that the harms occur; and if so, under what conditions. 
    - A common approach to testing for potential harms or vulnerabilities in a software solution is to use "red team" testing. a team of testers deliberately probes the solution for weaknesses and attempts to produce harmful results.
    - Red teaming is a strategy that is often used to find security vulnerabilities or other weaknesses that can compromise the integrity of a software solution. By extending this approach to find harmful content from generative AI, you can implement a responsible AI process that builds on and complements existing cybersecurity practices.
  - Document and share the verified harms
    - When you have gathered evidence to support the presence of potential harms in the solution, document the details and share them with stakeholders. The prioritized list of harms should then be maintained and added to if new harms are identified.
- Measure the presence of these harms in the outputs generated by your solution.
  - test the solution to measure the presence and impact of harms.
  - goal is to create an initial baseline that quantifies the harms produced by your solution in given usage scenarios; and then track improvements against the baseline as you make iterative changes in the solution to mitigate the harms.
  - A generalized approach to measuring a system for potential harms consists of three steps:
    - Prepare a diverse selection of input prompts that are likely to result in each potential harm that you have documented for the system. 
    - Submit the prompts to the system and retrieve the generated output.
    - Apply pre-defined criteria to evaluate the output and categorize it according to the level of potential harm it contains. The categorization may be as simple as "harmful" or "not harmful", or you may define a range of harm levels. Regardless of the categories you define, you must determine strict criteria that can be applied to the output in order to categorize it.
  - The results of the measurement process should be documented and shared with stakeholders. 测量过程的结果应记录下来并与利益相关者共享。
  - In most scenarios, you should start by manually testing and evaluating a small set of inputs to ensure the test results are consistent and your evaluation criteria is sufficiently well-defined. Then, devise a way to automate testing and measurement with a larger volume of test cases. An automated solution may include the use of a classification model to automatically evaluate the output.
  - Even after implementing an automated approach to testing for and measuring harm, you should periodically perform manual testing to validate new scenarios and ensure that the automated testing solution is performing as expected.
- Mitigate the harms at multiple layers in your solution to minimize their presence and impact, and ensure transparent communication about potential risks to users.
  - take steps to mitigate the potential harms, and when appropriate retest the modified system and compare harm levels against the baseline.
  - Mitigation of potential harms in a generative AI solution involves a layered approach, in which mitigation techniques can be applied at each of four layers, 
    - The model layer
      - The model layer consists of the generative AI model(s) at the heart of your solution. 
        - Selecting a model that is appropriate for the intended solution use. 
        - Fine-tuning a foundational model with your own training data so that the responses it generates are more likely to be relevant and scoped to your solution scenario.
      - The safety system layer
        - The safety system layer includes platform-level configurations and capabilities that help mitigate harm.
        - Other safety system layer mitigations can include abuse detection algorithms to determine if the solution is being systematically abused and alert notifications that enable a fast response to potential system abuse or harmful behavior.
      - The metaprompt and grounding layer
        - The metaprompt and grounding layer focuses on the construction of prompts that are submitted to the model. 
          - Specifying metaprompts or system inputs that define behavioral parameters for the model.
          - Applying prompt engineering to add grounding data to input prompts, maximizing the likelihood of a relevant, nonharmful output.
          - Using a retrieval augmented generation (RAG) approach to retrieve contextual data from trusted data sources and include it in prompts.
      - The user experience layer
        - The user experience layer includes the software application through which users interact with the generative AI model as well as documentation or other user collateral that describes the use of the solution to its users and stakeholders.
        - Designing the application user interface to constrain inputs to specific subjects or types, or applying input and output validation can mitigate the risk of potentially harmful responses.
        - Documentation and other descriptions of a generative AI solution should be appropriately transparent about the capabilities and limitations of the system, the models on which it's based, and any potential harms that may not always be addressed by the mitigation measures you have put in place.
- Operate the solution responsibly by defining and following a deployment and operational readiness plan.
  - Complete prerelease reviews
    - Before releasing a generative AI solution, identify the various compliance requirements in your organization and industry and ensure the appropriate teams are given the opportunity to review the system and its documentation. Common compliance reviews include:
      - Legal
      - Privacy
      - Security
      - Accessibility
  - Release and operate the solution
    - A successful release requires some planning and preparation. Consider the following guidelines:
    - Devise a phased delivery plan that enables you to release the solution initially to restricted group of users. This approach enables you to gather feedback and identify problems before releasing to a wider audience. 设计一个分阶段的交付计划，使您能够首先向有限的用户组发布解决方案。这种方法使您能够在向更广泛的受众发布之前收集反馈并发现问题。
    - Create an incident response plan that includes estimates of the time taken to respond to unanticipated incidents.创建事件响应计划，其中包括响应意外事件所需时间的估计。
    - Create a rollback plan that defines the steps to revert the solution to a previous state in the event of an incident.创建回滚计划，定义在发生事件时将解决方案恢复到之前状态的步骤。
    - Implement the capability to immediately block harmful system responses when they're discovered.实现在发现有害系统响应时立即阻止它们的功能。
    - Implement a capability to block specific users, applications, or client IP addresses in the event of system misuse.实现在系统误用时阻止特定用户、应用程序或客户端 IP 地址的功能。
    - Implement a way for users to provide feedback and report issues. In particular, enable users to report generated content as "inaccurate", "incomplete", "harmful", "offensive", or otherwise problematic.实施一种供用户提供反馈和报告问题的方法。特别是，使用户能够将生成的内容报告为“不准确”、“不完整”、“有害”、“冒犯性”或其他有问题的内容。
    - Track telemetry data that enables you to determine user satisfaction and identify functional gaps or usability challenges. Telemetry collected should comply with privacy laws and your own organization's policies and commitments to user privacy.跟踪遥测数据，使您能够确定用户满意度并识别功能差距或可用性挑战。收集的遥测数据应遵守隐私法以及您自己组织的用户隐私政策和承诺。