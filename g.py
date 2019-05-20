import tensorflow as tf
from guess_new import main



def flag(model_dir):

    tf.app.flags.DEFINE_string('model_dir',model_dir,
                               'Model directory (where training data lives)')

    tf.app.flags.DEFINE_string('class_type', 'age',
                               'Classification type (age|gender)')


    tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                               'What processing unit to execute inference on')

    tf.app.flags.DEFINE_string('filename', '',
                               'File (Image) or File list (Text/No header TSV) to process')

    tf.app.flags.DEFINE_string('target', '',
                               'CSV file containing the filename processed along with best guess and score')

    tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                              'Checkpoint basename')

    tf.app.flags.DEFINE_string('model_type', 'inception',
                               'Type of convnet')

    tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

    tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

    tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

    tf.app.flags.DEFINE_string('face_detection_type', 'cascade', 'Face detection model type (yolo_tiny|cascade)')

    FLAGS = tf.app.flags.FLAGS
    return FLAGS


model_dir="/home/randheer/Desktop/6thsem/dl/hackthon/22801/"
image_path="/home/randheer/Desktop/6thsem/dl/hackthon/bounding_box/img_3.png"
FLAGS=flag(model_dir)

#main()
print(main(image_path,FLAGS))
#tf.reset_default_graph()
print(main(image_path,FLAGS))    
# if __name__ == '__main__':
#tf.app.run()
