# Fundamentals of Facial Recognition
### Level: Beginner
### Link: [Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/detect-analyze-faces/)
### Duration: 2 Hours
---

## Course Note
- Face detection involves identifying regions of an image that contain a human face, typically by returning bounding box coordinates that form a rectangle around the face. 人脸识别是识别存在人脸的图形中人脸的区域，一般返回人脸位置，用框框出。
- facial recognition uses multiple images of an individual to train the model. 面部识别是根据人脸特征识别确定的人，一般是用一个人的多张图片来训练。

### Azure sources in Azure AI
- Azure AI Vision, which offers face detection and some basic face analysis, such as returning the bounding box coordinates around an image.
- Azure AI Video Indexer, which you can use to detect and identify faces in a video.
- Azure AI Face, which offers pre-built algorithms that can detect, recognize, and analyze faces.

### Face Service
- The Azure Face service can return the rectangle coordinates for any human faces that are found in an image, as well as a series of attributes related to those face such as:
  - Accessories配件: indicates whether the given face has accessories. This attribute returns possible accessories including headwear, glasses, and mask, with confidence score between zero and one for each accessory.
  - Blur模糊: how blurred the face is, which can be an indication of how likely the face is to be the main focus of the image.
  - Exposure曝光: such as whether the image is underexposed or over exposed. This applies to the face in the image and not the overall image exposure.
  - Glasses眼睛: whether or not the person is wearing glasses.
  - Head pose头部姿势: the face's orientation in a 3D space.
  - Mask口罩: indicates whether the face is wearing a mask.
  - Noise噪点: refers to visual noise in the image. If you have taken a photo with a high ISO setting for darker settings, you would notice this noise in the image. The image looks grainy or full of tiny dots that make the image less clear.
  - Occlusion遮挡: determines if there might be objects blocking the face in the image.

### Responsible AI use
- The Limited Access policy requires customers to submit an intake form to access additional Azure AI Face service capabilities including:
  - The ability to compare faces for similarity.
  - The ability to identify named individuals in an image.


### Azure resources for Face
- Face: Use this specific resource type if you don't intend to use any other Azure AI services, or if you want to track utilization and costs for Face separately.
- Azure AI services: A general resource that includes Azure AI Face along with many other Azure AI services such as Azure AI Content Safety, Azure AI Language, and others. Use this resource type if you plan to use multiple Azure AI services and want to simplify administration and development.

### Tips for more accurate results
- Image format - supported images are JPEG, PNG, GIF, and BMP.
- File size - 6 MB or smaller.
- Face size range - from 36 x 36 pixels up to 4096 x 4096 pixels. Smaller or larger faces will not be detected.
- Other issues - face detection can be impaired by extreme face angles, extreme lighting, and occlusion (objects blocking the face such as a hand).