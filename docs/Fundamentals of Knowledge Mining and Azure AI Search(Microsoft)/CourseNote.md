# Fundamentals of Knowledge Mining and Azure AI Search
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/intro-to-azure-search/)
### Duration: 4 Hours
---

## Course Note
### What is Azure AI Search
- Azure AI Search provides the infrastructure and tools to create search solutions that extract data from various structured, semi-structured, and non-structured documents.
- Azure AI Search results contain only your data, which can include text inferred or extracted from images, or new entities and key phrases detection through text analytics. 
- It's a Platform as a Service (PaaS) solution.

### Azure AI Search features
- Azure AI Search exists to complement existing technologies and provides a programmable search engine built on Apache Lucene, an open-source software library.
- It's a highly available platform offering a 99.9% uptime SLA available for cloud and on-premises assets.
- features:
  - Data from any source: accepts data from any source provided in JSON format, with auto crawling support for selected data sources in Azure.
  - Full text search and analysis: offers full text search capabilities supporting both simple query and full Lucene query syntax. 全文搜索和分析
  - AI powered search: has Azure AI capabilities built in for image and text analysis from raw content.
  - Multi-lingual offers linguistic analysis for 56 languages to intelligently handle phonetic matching or language-specific linguistics. Natural language processors available in Azure AI Search are also used by Bing and Office. 多种语言
  - Geo-enabled: supports geo-search filtering based on proximity to a physical location. 支持地理位置
  - Configurable user experience: has several features to improve the user experience including autocomplete, autosuggest, pagination, and hit highlighting. 可配置的用户体验。

### Identify elements of a search solution
- A typical Azure AI Search solution starts with a data source that contains the data artifacts you want to search.
  - A hierarchy of folders and files in Azure Storage
  - Text in a database such as Azure SQL Database or Azure Cosmos DB
  - Regardless of where your data originates, if you can provide it as a JSON document, the search engine can index it
- If your data resides in supported data source, you can use an indexer to automate data ingestion, including JSON serialization of source data in native formats.
  - An indexer connects to a data source, serializes the data, and passes to the search engine for indexing. 
  - Most indexers support change detection, which makes data refresh a simpler exercise.
- Besides automating data ingestion, indexers also support AI enrichment. 除了自动数据摄取外，还支持AI能力的接入。

### Use a skillset to define an enrichment pipeline
- AI enrichment refers to embedded image and natural language processing in a pipeline that extracts text and information from content that can't otherwise be indexed for full text search. 在pipeline中嵌入图像和自然语言处理来抽取那些不能被全文搜索索引化的文本和信息
- AI processing is achieved by adding and combining skills in a skillset. A skillset defines the operations that extract and enrich data to make it searchable. 
- Built in skills
  - Natural language processing skills: with these skills, unstructured text is mapped as searchable and filterable fields in an index.
  - Image processing skills: creates text representations of image content, making it searchable using the query capabilities of Azure AI Search.

### Understand indexes
- An Azure AI Search index can be thought of as a container of searchable documents. 
- Index schema
  - In Azure AI Search, an index is a persistent collection of JSON documents and other content used to enable search functionality. 
  - The documents within an index can be thought of as rows in a table, each document is a single unit of searchable data in the index.
  - The index includes a definition of the structure of the data in these documents, called its schema.
- Index attributes
  - Azure AI Search needs to know how you would like to search and display the fields in the documents.
  - You specify that by assigning attributes, or behaviors, to these fields. For each field in the document, the index stores its name, the data type, and supported behaviors for the field.
  - The most efficient indexes use only the behaviors that are needed. If you forget to set a required behavior on a field when designing, the only way to get that feature is to rebuild the index.

### Use an indexer to build an index
- Azure AI Search lets you create and load JSON documents into an index with two approaches:
  - Push method: JSON data is pushed into a search index via either the REST API or the .NET SDK. Pushing data has the most flexibility as it has no restrictions on the data source type, location, or frequency of execution.
  - Pull method: Search service indexers can pull data from popular Azure data sources, and if necessary, export that data into JSON if it isn't already in that format.
- Use the pull method to load data with an indexer
  - Azure AI Search's indexer is a crawler that extracts searchable text and metadata from an external Azure data source and populates a search index using field-to-field mappings between source data and your index.
  - Using the indexer is sometimes referred to as a 'pull model' approach because the service pulls data in without you having to write any code that adds data to an index. An indexer maps source fields to their matching fields in the index.
- Data import monitoring and verification
  - The search services overview page has a dashboard that lets you quickly see the health of the search service.
  - When loading new documents into an index, the progress can be monitored by clicking on the index's associated indexer. 
  - Once the index is ready for querying, you can then use Search explorer to verify the results. An index is ready when the first document is successfully loaded.
  - Indexers only import new or updated documents, so it is normal to see zero documents indexed.
  - The Search explorer can perform quick searches to check the contents of an index, and ensure that you are getting expected search results. 
- Making changes to an index
  - You have to drop and recreate indexes if you need to make changes to field definitions. 
  - Adding new fields is supported, with all existing documents having null values. 
  - You'll find it faster using a code-based approach to iterate your designs, as working in the portal requires the index to be deleted, recreated, and the schema details to be manually filled out.
  - An approach to updating an index without affecting your users is to create a new index under a different name. You can use the same indexer and data source. After importing data, you can switch your app to use the new index.

### Persist enriched data in a knowledge store
- A knowledge store is persistent storage of enriched content. 
- The purpose of a knowledge store is to store the data generated from AI enrichment in a container.
- The outcome can be a search index, or projections in a knowledge store. The two outputs, search index and knowledge store, are mutually exclusive products of the same pipeline; derived from the same inputs, but resulting in output that is structured, stored, and used in different applications.
- A knowledge store can contain one or more of three types of projection of the extracted data:
  - Table projections are used to structure the extracted data in a relational schema for querying and visualization
  - Object projections are JSON documents that represent each data entity
  - File projections are used to store extracted images in JPG format

### Create an index in the Azure portal
- Azure portal support data sources include:
  - Cosmos DB (SQL API)
  - Azure SQL (database, managed instance, and SQL Server on an Azure VM)
  - Azure Storage (Blob Storage, Table Storage, ADLS Gen2)
- Using the Azure portal's Import data wizard, you can create:
  - Data Source: Persists connection information to source data, including credentials. A data source object is used exclusively with indexers.
  - Index: Physical data structure used for full text search and other queries.
  - Indexer: A configuration object specifying a data source, target index, an optional AI skillset, optional schedule, and optional configuration settings for error handling and base-64 encoding.
  - Skillset: A complete set of instructions for manipulating, transforming, and shaping content, including analyzing and extracting information from image files. Except for very simple and limited structures, it includes a reference to an Azure AI services resource that provides enrichment.
  - Knowledge store: Stores output from an AI enrichment pipeline in tables and blobs in Azure Storage for independent analysis or downstream processing.

### Query data in an Azure AI Search index
- Index and query design are closely linked.
- the schema of the index determines what queries can be answered.
- Azure AI Search supports two types of syntax: 
  - Simple syntax covers all of the common query scenarios.
  - Full Lucene is useful for advanced scenarios.

