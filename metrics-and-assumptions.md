# Metrics and assumptions

Generally, within computer vision and machine learning tasks, it is possible to outline a number of discrete metrics, for example:



* **Precision and Recall**: where for precision we measure the correctness of the prediction, and with recall we calculate how many true predictions were made out of total predictions.
* **Average precision \(AP\)**: Using Average precision we can get numerical values which we can use in comparing our model with others. With the help of the precision-recall curve average precision calculates the weighted mean of precisions if there is an increase in recall. We calculate the average precision for each object.
*  **Mean Average Precision \(MAP\)**: MAP is an extension of average precision \(AP\). In AP we calculate it for each object but in MAP we compute the precision of the entire model. This gives the percentage of correct predictions.
* **Intersection over Union \(IoU\):** This metric finds the difference between the ground truth and the prediction of bounding boxes. With the help of confidence scores, we remove some unwanted bounding boxes from the output.

source: adapted from[ Object Detection Basics](https://datamahadev.com/object-detection-basics-and-performance-metrics/)

This is applicable in the case of using deep learning to track particles, thus these metrics are included in the performance evaluation of the tracking of a time lapse microscopy capture. However, these metrics are not the direct point of interest in a creative project. 

It is challenging to evaluate the effectiveness of a model in terms of its ability to generate novel images based on and origin and target dataset, as this is ultimately a subjective, human judgement based on complex cultural-aesthetic judgements that are not explicitly open to metrics-based assessment or analysis. 

However, while the creators of UGATIT do not go into detail in terms of metrics, they do note that their :

> model guides the translation to focus on more important regions and ignore minor regions by distinguishing between source and target domains based on the attention map obtained by the auxiliary classifier

where

> attention maps are embedded into the generator and discriminator to focus on semantically important areas, thus facilitating the shape transformation.

For this reason, the creators of UGATIT conducted qualitative analysis on the outputs of their model by conducting a perceptual study. 135 participants were shown translated results from different methods including the proposed method with source image, and asked to select the best translated image to target domain.

source: [UGATIT paper](https://arxiv.org/pdf/1907.10830v4.pdf)

In my particular instance, I would need weavers and microbiologists to qualitatively analyse the outputs of my model, and in addition would need to check that they can actually be woven. 

This is an interesting idea for a more in-depth project phase for the future. 



