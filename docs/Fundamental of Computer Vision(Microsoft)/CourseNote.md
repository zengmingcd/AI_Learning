# Fundamentals of Computer Vision
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/analyze-images-computer-vision/)
### Duration: 2 Hours
---

## Course Note
### Images as pixel arrays
- To a computer, an image is an array of numeric pixel values
- Image's resolution: X * Y pixel image
- Most digital images are multidimensional and consist of three layers (known as channels) that represent red, green, and blue (RGB) color hues.


### Using filters to process images
- A common way to perform image processing tasks is to apply filters that modify the pixel values of the image to create a visual effect. 通常处理图像的方式是应用滤镜来修改像素值以创建视觉效果。
- A filter is defined by one or more arrays of pixel values, called filter kernels. 滤波器由一个或多个像素阵列组成，称为滤波内核。
- The kernel is then convolved across the image, calculating a weighted sum for each XxY patch of pixels and assigning the result to a new image. 滤波内核在图像上卷积操作。计算每个X*Y大小块像素的加权和，并分配到新的图像中。
- steps：
  - Apply the filter kernel to the top left patch of the image, multiplying each pixel value by the corresponding weight value in the kernel and adding the results. 将滤波内核应用于左上方的块，将每个像素值与对应内核值进行相乘后相加，得到第一个新数据。
  - Move the filter kernel along one pixel to the right and repeat the operation. 向右移动一个像素，重复上面的计算。
  - Repeated until the filter has been convolved across the entire image. 重复操作，直到所有像素点都被覆盖。
  - Some of the values might be outside of the 0 to 255 pixel value range, so the values are adjusted to fit into that range. 一些像素的值会超过0~255的范围。会使用0或255填充
  - Because of the shape of the filter, the outside edge of pixels isn't calculated, so a padding value (usually 0) is applied. 由于过滤器的形状，边缘不会被计算。所以用填充值进行填充，例如0
  -  In this case, the filter has had the effect of highlighting the edges of shapes in the image. 由于边缘被填充，所以会突出边缘效果。
  -  Because the filter is convolved across the image, this kind of image manipulation is often referred to as convolutional filtering. 因为滤波器在图像上卷积，所以这种方式叫做卷积滤波。

### Convolutional neural networks (CNNs)
- One of the most common machine learning model architectures for computer vision.
- Use filters to extract numeric feature maps from images, and then feed the feature values into a deep learning model to generate a label prediction. 使用滤波器抽取图片数字化的特征地图，然后输送给深度学习模型来生成预测。
- How a CNN for an image classification model works: CNN模型工作原理
  - Images with known labels are fed into the network to train the model.将带标签的图片输送给网络以训练模型。
  - One or more layers of filters is used to extract features from each image as it is fed through the network. The filter kernels start with randomly assigned weights and generate arrays of numeric values called feature maps. 一个或多个滤波器被用于图片来抽取特征。滤波核心随机分配权重并生成数字化的特征图。
  - The feature maps are flattened into a single dimensional array of feature values. 特征图被展开成为一维特征值数组。
  - The feature values are fed into a fully connected neural network. 特征值被传送到全连接的神经网络中。
  - The output layer of the neural network uses a softmax or similar function to produce a result that contains a probability value for each possible class. 输出层使用softmax或者相似的函数来生成包含每个类型的概率值的结果。
  - During training the output probabilities are compared to the actual class label. The difference between the predicted and actual class scores is used to calculate the loss in the model, and the weights in the fully connected neural network and the filter kernels in the feature extraction layers are modified to reduce the loss. 将训练过程中的输出与实际标签对比，这个差值用来计算损失，然后调整权重和内核来减少损失。
  - The training process repeats over multiple epochs until an optimal set of weights has been learned. Then, the weights are saved and the model can be used to predict labels for new images for which the label is unknown. 重复训练，直到得到最佳权重。

### Transformer
- Transformers work by processing huge volumes of data, and encoding language tokens (representing individual words or phrases) as vector-based embeddings (arrays of numeric values). You can think of an embedding as representing a set of dimensions that each represent some semantic attribute of the token. The embeddings are created such that tokens that are commonly used in the same context are closer together dimensionally than unrelated words. Transformer的工作原理是处理大量数据，并将语言token编码为基于向量的的嵌入。你可以将一个嵌入视作一组维度，每个维度标识某个语义属性。创建嵌入使得上下文中语义相近的token更加靠近。

### Multi-modal model 多模式模型
- Trained using a large volume of captioned images, with no fixed labels. 使用大量带字幕的图片进行训练，没有固定的标签。
- An image encoder extracts features from images based on pixel values and combines them with text embeddings created by a language encoder. The overall model encapsulates relationships between natural language token embeddings and image features. 图形编码使用像素值抽取图形特征，语言编码创建文本嵌入，再将两者相结合。整体模型封装了语言token和图形特征的关系。
- Florence is an example of a foundation model. a pre-trained general model on which you can build multiple adaptive models for specialist tasks. Florence是一个基础模型的示例。可以使用一个预训练模型作为基础，再在专业任务上建立多个自适应模型。
  - Image classification: Identifying to which category an image belongs.图像分类
  - Object detection: Locating individual objects within an image. 对象识别
  - Captioning: Generating appropriate descriptions of images. 生成字幕
  - Tagging: Compiling a list of relevant text tags for an image. 标记


### Azure resources for Azure AI Vision service
- Azure AI Vision: A specific resource for the Azure AI Vision service. Use this resource type if you don't intend to use any other Azure AI services, or if you want to track utilization and costs for your Azure AI Vision resource separately. 只用视觉服务
- Azure AI services: A general resource that includes Azure AI Vision along with many other Azure AI services; such as Azure AI Language, Azure AI Custom Vision, Azure AI Translator, and others. Use this resource type if you plan to use multiple AI services and want to simplify administration and development. 通用服务，包含视觉

### Analyzing images with the Azure AI Vision service
- Optical character recognition (OCR) - extracting text from images.
- Generating captions and descriptions of images.
- Detection of thousands of common objects in images.
  - The predictions include a confidence score that indicates the probability the model has calculated for the predicted objects. 预测包括置信度。
  - Azure AI Vision returns bounding box coordinates that indicate the top, left, width, and height of the object detected. 对象的顶部、左边、宽度、高度信息
- Tagging visual features in images
  - suggest tags for an image based on its contents.
- Training custom models 训练自己的模型
  - Image classification
    - predict the category, or class of an image
  - Object detection
    - detect and classify objects in an image, returning bounding box coordinates to locate each object. 识别对象并分类。返回边界坐标以定位对象。
    - 


