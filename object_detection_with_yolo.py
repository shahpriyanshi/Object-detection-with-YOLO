import os
import numpy as np
import PIL
import tensorflow as tf
import keras as K
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):

    box_scores = np.multiply(box_confidence, box_class_probs)
    box_classes = K.backend.argmax(box_scores)
    box_class_scores = K.backend.max(box_scores, axis=-1)

    filtering_mask = (box_class_scores >= threshold)

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


def yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold):

    max_boxes_tensor = K.backend.variable(max_boxes, dtype='int32')
    K.backend.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.backend.gather(scores, nms_indices)
    boxes = K.backend.gather(boxes, nms_indices)
    classes = K.backend.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape, max_boxes=10, score_threshold=.6, iou_threshold=.5):

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


sess = K.backend.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")

im = PIL.Image.open("images/dog.jpg")
x, y = im.size
x, y = float(x), float(y)
image_shape = (x, y)

yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


def predict(sess, image_file):

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))
    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    out_scores, out_boxes, out_classes = sess.run(fetches=[scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data,
                                                             K.backend.learning_phase(): 0})
    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)

    return out_scores, out_boxes, out_classes


out_scores, out_boxes, out_classes = predict(sess, "dog.jpg")
