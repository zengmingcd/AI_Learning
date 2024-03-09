# Fundamentals of optical character recognition
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/read-text-computer-vision/)
### Duration: 2 Hours
---

## Course Note
- The ability for computer systems to process written and printed text is an area of AI where computer vision intersects with natural language processing. 
- Azure AI Vision service has the ability to extract machine-readable text from images. 
- Azure AI Vision's Read API is the OCR engine that powers text extraction from images, PDFs, and TIFF files. 
- The Read API, otherwise known as Read OCR engine, uses the latest recognition models and is optimized for images that have a significant amount of text or have considerable visual noise. It can automatically determine the proper recognition model to use taking into consideration the number of lines of text, images that include text, and handwriting.
- The OCR engine takes in an image file and identifies bounding boxes, or coordinates, where items are located within an image. In OCR, the model identifies bounding boxes around anything that appears to be text in the image.

- Read API returns results arranged into the following hierarchy:
  - Pages - One for each page of text, including information about the page size and orientation.
  - Lines - The lines of text on a page.
  - Words - The words in a line of text, including the bounding box coordinates and text itself.

- Azure resources in Azure AI
  - Azure AI Vision: A specific resource for vision services. Use this resource type if you don't intend to use any other AI services, or if you want to track utilization and costs for your AI Vision resource separately.
  - Azure AI services: A general resource that includes Azure AI Vision along with many other Azure AI services such as Azure AI Language, Azure AI Speech, and others. Use this resource type if you plan to use multiple Azure AI services and want to simplify administration and development.

- use Azure AI Vision's Read API:
  - Vision Studio
  - REST API
  - Software Development Kits (SDKs): Python, C#, JavaScript

- Your default resource in Vision Studio must be an Azure AI services resource