## ORCNN annotation procedure, using LabelMe
ORCNN is an extended Mask R-CNN network with two mask head branches. To be able to train ORCNN, the annotations need to have two masks as well:
<br/>
1. An amodal mask (of the visible and the invisible pixels)
2. A visible mask

<br/>

We recommend the following procedure, using the LabelMe program (https://github.com/wkentaro/labelme):
<br/>
1. For each superclass, make two subclasses using a common separator: 
   1. one class for the amodal masks (separator: [class]_amodal)
   2. one class for the visible masks (separator: [class]_visible)
2. Use distinct group ids, to link the amodal and the visible masks to the same object.
3. If a visible mask comprises of two or more separated parts, please use the same group id.
4. For both the visible and the amodal masks, please use the following shapes in LabelMe: rectangle, circle, polygon. 
<br/> <br/> ![LabelMe procedure](./annotation/labelme_screenshot.png?raw=true)
<br/> *This is an example of an amodal and visible mask annotation of two objects of the superclass "broccoli". Each broccoli object has one amodal mask (a red circle with the label "broccoli_amodal") and two visible masks (green polygons that are separated by a leaf, expressed with the labels "broccoli_visible"). To link the multiple masks to the same object, please use distinct group ids (see the (1) and (2) group id behind the annotation).*

<br/> <br/>
When the annotations are finished, please convert the annotations to one JSON file that can be processed by ORCNN:
<br/>
1. In a new terminal, activate the virtual environment (for example: conda activate sizecnn)
2. Go to the annotation folder (cd /home/[user_name]/sizecnn/annotation, replace [user_name] by your own username)
3. Run the python file **labelme_to_orcnn.py**, using the appropriate arguments (see below): <br/> <br/>

| Argument        	| Description           						|
| ----------------------|-----------------------------------------------------------------------|
| --annotation_dir      | directory with the images and the annotations 			|
| --write_dir     	| directory to store the ORCNN images and annotation file	 	|
| --amodal_separator 	| subclass separator of the amodal masks    				|
| --visible_separator	| subclass separator of the visible masks     				|
| --description 	| description of your dataset      					|
| --contributor 	| your name     							|

<br/>
Example (replace [user_name] by your own username): 
<br/> python labelme_to_orcnn.py --annotation_dir "/home/[user_name]/sizecnn/data/annotations/train" --write_dir "/home/[user_name]/sizecnn/data/orcnn_annotations/train" --classes broccoli apple --amodal_separator _amodal --visible_separator _visible --description broccoli_apple_amodal_visible --contributor PieterBlok <br/> <br/> <br/> 

***Additional features of the labelme_to_orcnn.py program:***
<br/>
1. The program automatically checks whether all images have an annotation and all annotations have an image (if not: an error is raised, and the program is automatically stopped)
2. Before starting the conversion, the program checks whether all annotations have at least one amodal and one visible mask (if not: an error is raised, and the program is automatically stopped)
3. Before starting the conversion, the program checks whether there are empty group ids (if so: an error is raised, and the program is automatically stopped)
4. If no errors in step 1-3, then the conversion is started. 
5. Besides the above mentioned arguments, you can also input the other arguments:<br/> <br/> 

| Argument        	| Description           						|
| ----------------------|-----------------------------------------------------------------------|
| --creator_url      	| this is an URL that is linked to the contributor 			|
| --version      	| this is the version number of the dataset (input should be an integer!)      |
| --license_url 	| this is an URL of the license      					|
| --license_id 		| this is the identifier of the license      				|
| --license_name 	| this is the name of the license      					|
