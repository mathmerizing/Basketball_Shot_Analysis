# Basketball Shot Analysis
Plotting the trajectory of a basketball shot with Tensorflow's Object Detection API.

<h2>Setup:</h2><br>
STEP 1: Clone Tensorflow's Object Detection Repository:
https://github.com/tensorflow/models/tree/master/research/object_detection
<br><br>

STEP 2: Follow their installation guide:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
<br><br>

STEP 3: Download this repository and move all the files into the folder "object_detection" of the tensorflow repositor
<br><br>

STEP 4: Place your video of your shot in the same folder and run it. <br>
e.g. How to run it on "test.mp4":
<pre> >>> python3 annotate_video.py test.mp4 </pre>
<br>
<b> NOTE: </b> If the ball is being shot from the left side of the screen is moving towards the right, add the flag "--left_to_right":
<pre> >>> python3 annotate_video.py test_flipped.mp4  --left_to_right </pre>
 <br><br>
 
 STEP 5: Close the window with the animated frames of the shot around the time when the ball is being released.
 <br><br>
 
 STEP 6: Adjust the the release point with the slider at the bottom of the window. Close the window. <br><br>
 
 RESULT: The analyzed picture and the annotated movie have been saved.
 
 <h2>Example:</h2><br>
 <img src="https://github.com/mathmerizing/Basketball_Shot_Analysis/blob/master/test.png" width="900">
 <br><br>
 
 <h2>Requirements:</h2><br>
 
 - imageio<br>
 - numpy<br>
 - matplotlib<br>
 - tensorflow<br>
 - pillow
