import keras_rcnn
import numpy

training_dictionary, test_dictionary = keras_rcnn.datasets.shape.load_data()

categories = {"circle": 1, "rectangle": 2, "triangle": 3}

generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

generator = generator.flow_from_dictionary(
    dictionary=training_dictionary,
    categories=categories,
    target_size=(224, 224)
)

validation_data = keras_rcnn.preprocessing.ObjectDetectionGenerator()

validation_data = validation_data.flow_from_dictionary(
    dictionary=test_dictionary,
    categories=categories,
    target_size=(224, 224)
)

target, _ = generator.next()

target_bounding_boxes, target_categories, target_images, target_masks, target_metadata = target

target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

target_images = numpy.squeeze(target_images)

target_categories = numpy.argmax(target_categories, -1)

target_categories = numpy.squeeze(target_categories)

keras_rcnn.util.show_bounding_boxes(target_image, target_bounding_boxes, target_categories)