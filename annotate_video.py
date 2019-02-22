import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from matplotlib.widgets import Slider, Button, RadioButtons
from math import floor, ceil, sqrt, log10, degrees, atan

import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import progressbar

# This is needed since this script is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# block tensorflow debugging output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'basketball_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
# Minimal score to draw box
MIN_SCORE=.5
# Where is the ball being shot from?
SHOT_FROM_RIGHT=True

# this method has been copied from: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# INFERENCE METHOD ---------------------------------------------------------------------------------------------------------
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

# ROUND (to 1 significant digit)
def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))


# MAIN ------------------------------------------------------------------------------------------
if __name__ == '__main__':

    file_name = sys.argv[1]
    if ("--left_to_right"  in sys.argv): SHOT_FROM_RIGHT = False

    all_images = []
    x_points = []
    y_points = []

    # load frozen tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # load label map
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    reader = imageio.get_reader(file_name, "ffmpeg") # create video reader
    path , name = os.path.split(file_name)
    new_file = os.path.join(path,"annotated_"+name)
    writer = imageio.get_writer(new_file,fps=reader.get_meta_data()['fps']) # create video writer

    # number of frames in video
    frames_num = reader.count_frames()

    # for each frame in the video:
    counter = 0 #frame number
    for _, image in enumerate(reader.iter_data()):
        counter += 1

        # show the frame
        """
        fig, ax = plt.subplots()
        ax.set_title(f"Frame {counter}")
        ax.imshow(image)
        plt.show()
        """

        # add boxes to the image
        image_np = np.asarray(image)

        if (counter%1 == 0): # if there are too many frames 'if (counter%N == 0):' would only pick every N-th frame
            all_images.append(image_np)

        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)

        # get image dimensions
        height,width,_ = image.shape

        # only the first detection box matters
        box = output_dict['detection_boxes'][0]
        (ymin,xmin,ymax,xmax)=(box[0]*height,box[1]*width,box[2]*height,box[3]*width)
        # calculate the center of the detection_box
        (x_avg,y_avg) = ((xmin+xmax)/2,(ymin+ymax)/2)

        # only the first score matters
        score = output_dict['detection_scores'][0]
        # print("prediction score: ",score)
        if (score > MIN_SCORE):
            x_points.append(x_avg)
            y_points.append(y_avg)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              min_score_thresh=MIN_SCORE,
              line_thickness=8,
              skip_scores=True,
              skip_labels=True)

        # show frame with object detection
        """
        fig, ax = plt.subplots()
        ax.set_title(f"Annotated Frame {counter}")
        ax.imshow(image_np)
        ax.scatter([xmin,xmin,xmax,xmax],[ymin,ymax,ymin,ymax],s=4)
        ax.scatter([x_avg],[y_avg],c="red",marker = '+',s=12)
        plt.show()
        """

        # add annotated image to the video writer
        writer.append_data(image_np)

        # update progressbar
        progressbar.progress(counter,frames_num)

    # end line after progressbar
    sys.stdout.write("\n")

    # close video writer
    writer.close()

    """
    out_image = (1/(len(all_images)*255))*np.sum(all_images,axis=0)
    print("Number of frames to merge:",len(all_images))
    """

    # make regression
    (a,b,c) = np.polyfit(x_points, y_points, 2)
    f = np.poly1d([a,b,c])

    # adjust the boundaries of the plot, such that it is only being shown on the image (and not also below the image)
    x_left = max(1,ceil((-b-sqrt(b*b-4*a*(c-height)))/(2*a)))
    x_right = min(width - 1,floor((-b+sqrt(b*b-4*a*(c-height)))/(2*a)))
    x = np.linspace(x_left,x_right,100)

    # animate the frames in the all_images
    fig, ax = plt.subplots()
    ax.set_title("Close the window at the release point.")
    i = 0
    im = plt.imshow(all_images[i], animated=True)

    # replaces i-th frame with (i+1)-th frame
    def updatefig(*args):
        global i
        i = (i+1 if i+1 != len(all_images) else 0)
        im.set_array(all_images[i])
        return im,

    # run the animation
    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
    plt.show()
    # ax.imshow(out_image)

    # create an initial release_point
    start_position = (int(3*width/4) if SHOT_FROM_RIGHT else int(width/4))
    # print(f"f'({start_position}) = {derivative_start}")

    # create plot in which the angle of the shot can be analysed
    fig, ax = plt.subplots()
    ax.set_title("SHOT ANALYSIS")
    ax.imshow(all_images[i])
    # ax.imshow(out_image)

    # fitting parabola
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # draw regression
    ax.plot(x,f(x),label=f'$f(x) = {("-" if a < 0 else "")}{abs(round_to_1(a))} x^2 {("-" if b < 0 else "+")} {abs(round_to_1(b))} x {("-" if c < 0 else "+")}{abs(round(c))}$')

    # DRAW ANGLE -------------------------------------------------------------------------------------------------------------
    def draw_angle(pos):
        # delete old angle
        for child in ax.get_children():
            if isinstance(child,(matplotlib.patches.FancyArrow,matplotlib.patches.Arc,matplotlib.collections.PathCollection)):
                child.remove()
        # other x-coordinate of the slope triangle
        x_0 = pos-step*side_factor
        # slope of parabola at pos
        derivative_start = 2*a*pos+b
        # value of tangent to the parabola at pos evaluated at x_0
        tangent_x_0 = derivative_start*x_0+f(pos)-derivative_start*pos
        # angle between the adjacent and the hypothenuse (looking from pos)
        angle = abs(degrees(atan((tangent_x_0-f(pos))/(100))))
        # plot the release_point
        release_point = ax.scatter([pos],[f(pos)],c="black",label="release point")

        arr_head_proportion = 0.15
        arr1_horiz = step*(-1)*side_factor # horizontal displacement of the first arrow
        arr2_length = abs((-1)*side_factor*sqrt(step**2+(tangent_x_0-f(pos))**2)) # length of the second arrow
        arr_head_size = min(abs(arr1_horiz)*arr_head_proportion,arr2_length*arr_head_proportion)

        # draw both arrows
        arrow_1 = ax.arrow(pos,f(pos),arr1_horiz,0,fc="red",ec="red",shape="full",head_width=arr_head_size,head_length=arr_head_size,length_includes_head = True)
        arrow_2 = ax.arrow(pos,f(pos),arr1_horiz,tangent_x_0-f(pos),fc="red",ec="red",shape="full",head_width=arr_head_size,head_length=arr_head_size,length_includes_head = True)

        # starting and ending angles of the arc
        theta_2 = (180.0+angle if SHOT_FROM_RIGHT else 360.0)
        theta_1 = (180.0 if SHOT_FROM_RIGHT else 360.0-angle)

        # create and plot arc
        arc = matplotlib.patches.Arc(xy=(pos,f(pos)),width=step,height=step,angle=0,theta1=theta_1,theta2=theta_2,color="red",label="angle $\\approx$" + " " + str(round(angle,2)) + "°")
        ax.add_patch(arc)

        # draw ball positions
        # ax.scatter(x_points,y_points,c="orange",marker = '+')

    # -------------------

    # SLIDER_ON_CHANGED -------------------------------------------------------------------------------------------------------
    def slider_on_changed(val):
        # draw the new angle at the new position
        draw_angle(int(val))
        # update the legend
        for child in ax.get_children():
            if isinstance(child,matplotlib.legend.Legend):
                child.remove()
                break
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)
        # redraw the canvas
        fig.canvas.draw_idle()

    # --------------------

    side_factor = (1 if SHOT_FROM_RIGHT else -1)
    step = 100

    # show slope traingle
    draw_angle(start_position)

    # create a slider with a bounded range (the release_point should not exceed the extreme_point)
    ax_pos = ax.get_position()
    slider_ax  = fig.add_axes([ax_pos.x0, ax_pos.y0 -0.1, ax_pos.width, ax_pos.height/40])
    extreme_point = int(-b/(2*a))
    slider = Slider(slider_ax, 'release:', (x_left + 50 if not SHOT_FROM_RIGHT else extreme_point + 20), (x_right - 50 if SHOT_FROM_RIGHT else extreme_point - 20), valinit=start_position, valstep = 1)
    slider.on_changed(slider_on_changed)

    # create the legend and show the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)
    plt.show()

    # save the shown plot as an image
    fig.savefig(file_name.replace(".mp4",".png"))
