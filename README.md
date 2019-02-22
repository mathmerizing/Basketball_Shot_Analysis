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
 
 STEP 5: Adjust the release point with the slider at the bottom of the window. Pick a frame of the video with the second slider. Close the window. <br><br>
 
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
 
 <h2>Tips for best performance:</h2><br>
 
 - Stablize the camera or phone when you are recording the video.
 - Cut the video beforehand and feed in short videos to the script.
 - Use a high quality video becuase this maximizes the likelyhood of the basketball being recognized.
 - Make sure that there is only ONE basketball in the video. Otherwise a "wrong basketball" might be recognized.
 - Use a brown basketball.
