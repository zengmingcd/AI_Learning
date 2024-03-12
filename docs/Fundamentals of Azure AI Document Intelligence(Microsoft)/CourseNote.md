# Fundamentals of Azure AI Document Intelligence
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/analyze-receipts-form-recognizer/)
### Duration: 2 Hours
---

## Course Note
- The ability to extract text, layout, and key-value pairs are known as document analysis.
- Document analysis provides locations of text on a page identified by bounding box coordinates.
- A challenge for automating the process of analyzing documents is that forms and documents come in all different formats. Separate machine learning models need to be trained to provide high quality results for different forms and documents.

### Features of Azure AI Document Intelligence 
- Prebuilt models - pretrained models that have been built to process common document types such as invoices, business cards, ID documents, and more. These models are designed to recognize and extract specific fields that are important for each document type. 预训构建模型，是一系列的预训练模型，用于处理普通文档。识别和提取文档中特定字段。
- Custom models - can be trained to identify specific fields that are not included in the existing pretrained models. 训练模型用于补充预构建模型没有的功能。
- Document analysis - general document analysis that returns structured data representations, including regions of interest and their inter-relationships. 文档分析。返回结构化数据表示的一般文档分析。

### Prebuilt models
- The prebuilt models apply advanced machine learning to accurately identify and extract text, key-value pairs, tables, and structures from forms and documents. 预构建的模型应用先进的机器学习来准确识别和提取表单和文档中的文本、键值对、表格和结构
- Each field and data pair has a confidence level, indicating the likely level of accuracy. This could be used to automatically identify when a person needs to verify a receipt.每个数据对都有一个置信度。表明准确性水平。
- For best results when using the prebuilt receipt model, images should be:
  - JPEG, PNG, BMP, PDF, or TIFF format
  - File size less than 500 MB for paid (S0) tier and 4 MB for free (F0) tier
  - Between 50 x 50 pixels and 10000 x 10000 pixels
  - For PDF documents, no larger than 17 inches x 17 inches
  - One receipt per document

