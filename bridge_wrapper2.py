'''
A Module which binds Yolov7 repo with Deepsort with modifications
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers2 import read_class_names, create_box_encoder
from detection_helpers import *

from image_helpers import highlight_green, increase_saturation, filter_green


 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True



class YOLOv7_DeepSORT:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, reID_model_path:str, detector, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="./io_data/input/classes/coco.names",  ):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.detector = detector
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric, max_iou_distance=0.4, max_age=75, n_init=3) # initialize tracker


    def track_video(self,video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
        '''
        try: # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        out = None
        if output: # get video ready to save locally if flag is set
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        frame_num = 0
        while True: # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num +=1

            if skip_frames and not frame_num % skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
            if verbose >= 1:start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            yolo_dets = self.detector.detect(frame.copy(), plot_bb = False)  # Get the detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0
            
            else:
                bboxes = yolo_dets[:,:4]
                bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
                bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

                scores = yolo_dets[:,4]
                classes = yolo_dets[:,-1]
                num_objects = bboxes.shape[0]
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            
            names = []
            for i in range(num_objects): # loop through objects and use class index to get class name
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)

            names = np.array(names)
            count = len(names)

            if count_objects:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            self.tracker.predict()  # Call the tracker
            self.tracker.update(detections) #  updtate using Kalman Gain

            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
        
                color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
                # add the image size to the top left corner
                
                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count}")
            
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if output: out.write(result) # save output video

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cv2.destroyAllWindows()


TRACKER_PARAMS = {"conf_thres":0.25, "iou_thresh":0.45, "img_size":1088,  
                  "model_path":"./weights/v7-clean-dataset.pt", #  used
                #   "model_path":"./weights/yolov7-custom-dataset.pt",
                  "max_cosine_distance":0.4, "nms_max_overlap":1.0, 
                  "max_iou_distance":0.7, "max_age":20, "n_init":1, #very important for tracking performance
                  "clear_trace_at":30, "trace":True,
                  "reID_model_path":"./deep_sort/model_weights/mars-small128.pb",
                  "coco_names_path":"./io_data/input/classes/coco.names",}
class YoloDeepsortUtility(YOLOv7_DeepSORT):
    """ 
    Subclass of YOLOv7_DeepSORT with added functinoality
    
    Args:
        
        """
    def __init__(self, reID_model_path:str="./deep_sort/model_weights/mars-small128.pb", detector=None, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="./io_data/input/classes/coco.names", max_age = 75, n_init = 3, max_iou_distance = 0.4, trace=True, 
    clear_trace_at = 30, params_dict=None) -> None:
        if params_dict is not None:
            try: max_cosine_distance = params_dict["max_cosine_distance"]
            except KeyError: print("max_cosine_distance not in params_dict. Using default")
            try: nn_budget = params_dict["nn_budget"]
            except KeyError: print("nn_budget not in params_dict. Using default")
            try: nms_max_overlap = params_dict["nms_max_overlap"]
            except KeyError: print("nms_max_overlap not in params_dict. Using default")
            try: max_iou_distance = params_dict["max_iou_distance"]
            except KeyError: print("max_iou_distance not in params_dict. Using default")
            try: max_age = params_dict["max_age"]
            except KeyError: print("max_age not in params_dict. Using default")
            try: n_init = params_dict["n_init"]
            except KeyError: print("n_init not in params_dict. Using default")
            try: clear_trace_at = params_dict["clear_trace_at"]
            except KeyError: print("clear_trace_at not in params_dict. Using default")
        
        super().__init__(reID_model_path, detector, max_cosine_distance, nn_budget, nms_max_overlap, coco_names_path)
        self.tracker.max_iou_distance = max_iou_distance
        self.tracker.max_age = max_age
        self.tracker.n_init = n_init
        self.trace = trace
        self.centroids = {}
        self.clear_trace_at = clear_trace_at
        self.max_trace_length = 15
        self.satturation = 1
        self.frames_to_treat = 15
        self.rotate = False
        self.draw_object_detections = False

    def add_centroid(self, bbx, id, frame_num):
        centroid = (int((bbx[0]+bbx[2])/2), int((bbx[1]+bbx[3])/2))
        
        try:    self.centroids[id].append(centroid)
        except: self.centroids[id] = [centroid]
        if f"{id}_frame" not in self.centroids:
            self.centroids[f"{id}_frame"] = frame_num
            self.centroids[f"{id}_frame_og"] = frame_num

        if frame_num - self.centroids[f"{id}_frame"] > self.clear_trace_at: 
            self.centroids[id] = []
        self.centroids[f"{id}_frame"] = frame_num
        

        for cent_id in self.centroids:
            # check if it's a string
            if isinstance(cent_id, str): continue
            if len(self.centroids[cent_id]) > self.max_trace_length:
                self.centroids[cent_id].pop(0)
            
    def draw_centroids(self, frame, cent_id, color, circle_colour=(250,20,20)):
        #check if cent_id
        if cent_id not in self.centroids:
            return
        
        # draw centroids as a continuous line
        if self.trace:
            for cent in self.centroids[cent_id]:
                cv2.circle(frame, cent, 5, circle_colour, -1)

            for j in range(1, len(self.centroids[cent_id])):
                if self.centroids[cent_id][j-1] is None or self.centroids[cent_id][j] is None: continue
                cv2.line(frame, self.centroids[cent_id][j-1], self.centroids[cent_id][j], color, 4)
    
    def reset_centroids(self):
        self.centroids = {}

    def yolo_detect(self, frame, draw=False):
        
        # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
        yolo_dets = self.detector.detect(frame.copy(), plot_bb=False)  # Get the detections

        if yolo_dets is None:
            bboxes = []
            scores = []
            classes = []
            num_objects = 0
        
        else:
            bboxes = yolo_dets[:,:4]

            scores = yolo_dets[:,4]
            classes = yolo_dets[:,-1]
            num_objects = bboxes.shape[0]
        # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
        # Draw bounding boxes and class names on the frame
        names = []
        for i in range(num_objects): # loop through objects and use class index to get class name
            class_indx = int(classes[i])
            class_name = self.class_names[class_indx]
            names.append(class_name)

        names = np.array(names)
        count = len(names)


        if num_objects > 0 and draw:
            for i in range(num_objects):
                cv2.rectangle(frame, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])), (0, 255, 0), 2)
                cv2.putText(frame, names[int(classes[i])], (int(bboxes[i][0]), int(bboxes[i][1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)   
            # result = np.asarray(frame)
            # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else: 
            result = frame

        if yolo_dets is not None:
            bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

        return frame, bboxes, scores, classes, num_objects
        
    def track_video(self,video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0,
                    frame_start:int=0, do_yield:bool=False):
            '''
            Track any given webcam or video
            args: 
                video: path to input video or set to 0 for webcam
                output: path to output video
                skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
                show_live: Whether to show live video tracking. Press the key 'q' to quit
                count_objects: count objects being tracked on screen
                verbose: print details on the screen allowed values 0,1,2
            '''
            try: # begin video capture
                vid = cv2.VideoCapture(int(video))
            except:
                vid = cv2.VideoCapture(video)

            out = None
            if output: # get video ready to save locally if flag is set
                width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
                height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(vid.get(cv2.CAP_PROP_FPS))
                codec = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(output, codec, fps, (width, height))

            frame_num = 0
            while True: # while video is running
                return_value, frame = vid.read()
                if not return_value:
                    print('Video has ended or failed!')
                    break
                if self.satturation != 1:
                    try:
                        frame = increase_saturation(frame, self.satturation)
                    except Exception as e:
                        print("Error increasing saturation: ", e)
                # crop the image at the top third
                # frame = frame[0:int(frame.shape[0]/3), 0:frame.shape[1]]
                # frame = filter_green(frame, threshold=160)
                frame_num +=1
                if frame_num < frame_start: continue

                if skip_frames and not frame_num % skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
                if verbose >= 1:start_time = time.time()

                # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
                frame, bboxes, scores, classes, num_objects = self.yolo_detect(frame, draw=self.draw_object_detections)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
                
                names = []
                for i in range(num_objects): # loop through objects and use class index to get class name
                    class_indx = int(classes[i])
                    class_name = self.class_names[class_indx]
                    names.append(class_name)

                names = np.array(names)
                count = len(names)

                # if count_objects:
                #     cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)
                # display image size/resolution ont top left corner
                cv2.putText(frame, "Width: " + str(frame.shape[1]) + "  Height: " + str(frame.shape[0]), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)
                # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------

                self.deepSortTrack(frame, bboxes, scores, classes, names)

                cmap = plt.get_cmap('tab20b') #initialize color map
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                objects_on_screen = []

                for track in self.tracker.tracks:  # update new findings AKA tracks
                    class_name = track.get_class()
                    # if track.get_class() == 'weed':
                    #     print("weed")
                    if track.time_since_update > 1:
                        continue 
                    if not track.is_confirmed() and track.get_class() != 'weed':
                        # small weeds have problems passing this.
                        continue
                    bbox = track.to_tlbr()
                    # object_list_name = class_name + " : " + str(track.track_id)
                    object_list_name = f"{class_name}{str(track.track_id)}"
                    # if object_list_name not in objects_on_screen:
                    objects_on_screen.append(object_list_name)
            
                    color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                    color = [i * 255 for i in color]
                    
                    frames_of_detection = 0
                    if class_name == "weed":
                        self.add_centroid(bbox, track.track_id, frame_num)
                        frames_of_detection = len(self.centroids[track.track_id])
                        # turn circles blue if it has been detected for frames_to_treat
                        circle_colour = (0,255,0) if frames_of_detection >= self.frames_to_treat else (250,20,20)
                        self.draw_centroids(frame, track.track_id, color, circle_colour=circle_colour)
                        # print(f"> > > > > {frame_num}-{self.centroids[f"{track.track_id}_frame_og"]}={frames_of_detection}?")
                        
                        # for centroid in self.centroids[track.track_id]:
                        #     cv2.circle(frame, centroid, 4, color, -1)
                    
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
                    if frames_of_detection != 0: cv2.putText(frame, str(frames_of_detection),(int(bbox[0]), int(bbox[1]+11)),0, 0.6, (255,10,10),1, lineType=cv2.LINE_AA)    
                    
                    
                    
                    if verbose == 2:
                        print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                            
                # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
                if verbose >= 1:
                    fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                    if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                    else: print(f"Frame: {frame_num} || Current FPS: {round(fps,2)} || {count} objects on screen: {objects_on_screen}")
                
                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                if output: out.write(result) # save output video

                if show_live:
                    # rotate frame
                    if self.rotate:
                        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

                    # resize to fit screen
                    result = cv2.resize(result, (720, 1080) )
                    
                    cv2.imshow("Output Video", result)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            cv2.destroyAllWindows()

            if output: 
                print("Saving video...")
                out.release()
        
    def track_video2(self,video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0,
                    frame_start:int=0, do_yield:bool=False):
            '''
            Same as track_video, but yields the tracker objects.
            args: 
                video: path to input video or set to 0 for webcam
                output: path to output video
                skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
                show_live: Whether to show live video tracking. Press the key 'q' to quit
                count_objects: count objects being tracked on screen
                verbose: print details on the screen allowed values 0,1,2
            '''
            try: # begin video capture
                vid = cv2.VideoCapture(int(video))
            except:
                vid = cv2.VideoCapture(video)

            out = None
            if output: # get video ready to save locally if flag is set
                width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
                height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(vid.get(cv2.CAP_PROP_FPS))
                codec = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(output, codec, fps, (width, height))

            frame_num = 0
            while True: # while video is running
                return_value, frame = vid.read()
                if self.satturation != 1:
                    try:
                        frame = increase_saturation(frame, self.satturation)
                    except Exception as e:
                        print("Error increasing saturation: ", e)
                # crop the image at the top third
                # frame = frame[0:int(frame.shape[0]/3), 0:frame.shape[1]]
                # frame = filter_green(frame, threshold=160)
                if not return_value:
                    print('Video has ended or failed!')
                    break
                frame_num +=1
                if frame_num < frame_start: continue

                if skip_frames and not frame_num % skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
                if verbose >= 1:start_time = time.time()

                # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
                frame, bboxes, scores, classes, num_objects = self.yolo_detect(frame, draw=self.draw_object_detections)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
                
                names = []
                for i in range(num_objects): # loop through objects and use class index to get class name
                    class_indx = int(classes[i])
                    class_name = self.class_names[class_indx]
                    names.append(class_name)

                names = np.array(names)
                count = len(names)

                # if count_objects:
                #     cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)
                # display image size/resolution ont top left corner
                cv2.putText(frame, "Width: " + str(frame.shape[1]) + "  Height: " + str(frame.shape[0]), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)
                # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------

                self.deepSortTrack(frame, bboxes, scores, classes, names)

                cmap = plt.get_cmap('tab20b') #initialize color map
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                objects_on_screen = []

                if do_yield: 
                    # for track in self.tracker.tracks: 
                    yield frame_num, frame, self.tracker.tracks
                else:
                    for track in self.tracker.tracks:  # update new findings AKA tracks
                        # if do_yield:
                        #     yield frame_num, frame, track
                        
                        class_name = track.get_class()
                        # if track.get_class() == 'weed':
                        #     print("weed")
                        if track.time_since_update > 1:
                            continue 
                        if not track.is_confirmed() and track.get_class() != 'weed':
                            # small weeds have problems passing this.
                            continue
                        bbox = track.to_tlbr()
                        # object_list_name = class_name + " : " + str(track.track_id)
                        object_list_name = f"{class_name}{str(track.track_id)}"
                        # if object_list_name not in objects_on_screen:
                        objects_on_screen.append(object_list_name)
                
                        color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                        color = [i * 255 for i in color]
                        
                        frames_of_detection = 0
                        if class_name == "weed":
                            self.add_centroid(bbox, track.track_id, frame_num)
                            frames_of_detection = len(self.centroids[track.track_id])
                            # turn circles blue if it has been detected for frames_to_treat
                            circle_colour = (0,255,0) if frames_of_detection >= self.frames_to_treat else (250,20,20)
                            self.draw_centroids(frame, track.track_id, color, circle_colour=circle_colour)
                            # print(f"> > > > > {frame_num}-{self.centroids[f"{track.track_id}_frame_og"]}={frames_of_detection}?")
                            
                            # for centroid in self.centroids[track.track_id]:
                            #     cv2.circle(frame, centroid, 4, color, -1)
                        
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                        cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
                        if frames_of_detection != 0: cv2.putText(frame, str(frames_of_detection),(int(bbox[0]), int(bbox[1]+11)),0, 0.6, (255,10,10),1, lineType=cv2.LINE_AA)    
                        
                        
                        
                        if verbose == 2:
                            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                            
                # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
                if verbose >= 1:
                    fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                    if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                    else: print(f"Frame: {frame_num} || Current FPS: {round(fps,2)} || {count} objects on screen: {objects_on_screen}")
                
                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                if output: out.write(result) # save output video

                if show_live:
                    # rotate frame
                    if self.rotate:
                        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

                    # resize to fit screen
                    result = cv2.resize(result, (720, 1080) )
                    
                    cv2.imshow("Output Video", result)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            cv2.destroyAllWindows()

            if output: 
                print("Saving video...")
                out.release()
    
    def deepSortTrack(self, frame, bboxes, classes, scores, names):
        """
        Track using DeepSORT. Updates the tracker object
        
        Args:
            frame (numpy array): frame to process
            bboxes (numpy array): bounding boxes of detections on the frame
            classes (numpy array): class of each detection
            scores (numpy array): confidence of each detection
            names (list): list of class names
            
        Returns:
            None
        """
        features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

        cmap = plt.get_cmap('tab20b') #initialize color map
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        self.tracker.predict()  # Call the tracker
        self.tracker.update(detections) #  updtate using Kalman Gain
    
    def track_video_from_image_folder(self, img_folder, img_format="png", output="pred_labels.txt"):

        """ 
         Go through images in a folder and track them, instead of using a video
         Store output as a txt in MOTA format: """
        
        frames = sorted([f for f in os.listdir(img_folder) if f.endswith(img_format)])
        frame_num = 0

        tracker_results = []

        for frame in frames:
            print(f"Tracking frame {frame_num}/{len(frames)}")
            frame_path = os.path.join(img_folder, frame)
            frame = cv2.imread(frame_path)

            # detect
            frame, bboxes, scores, classes, num_objects = self.yolo_detect(frame, draw=self.draw_object_detections)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            names = []
            for i in range(num_objects): # loop through objects and use class index to get class name
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)

            names = np.array(names)
            count = len(names)
            
            # Track
            self.deepSortTrack(frame, bboxes, None, scores, names)

            # cmap = plt.get_cmap('tab20b') #initialize color map
            # colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            for track in self.tracker.tracks:
                class_name = track.get_class()
                
                if track.time_since_update > 1:
                    continue 
                if not track.is_confirmed() and track.get_class() != 'weed':
                    # small weeds have problems passing this.
                    continue

                bbox = track.to_tlbr()  # Bounding box in (x1, y1, x2, y2)
                track_id = track.track_id

                # Convert bbox from (x1, y1, x2, y2) to (x_center, y_center, width, height)
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                x_center = x1 + width / 2
                y_center = y1 + height / 2

                # Use a fixed confidence score for tracker predictions (optional: get it from detection if available)
                confidence = 1.0

                # Visibility (set to 1 if object is fully visible; adjust as needed)
                visibility = 1.0

                # Normalize coordinates by frame dimensions
                frame_height, frame_width = frame.shape[:2]
                x_center /= frame_width
                y_center /= frame_height
                width /= frame_width
                height /= frame_height
                
                class_id = 0 if class_name == 'crop' else 1
                # print(class_name)

                # Append result in MOTChallenge format
                tracker_results.append([
                    frame_num, track_id, x_center, y_center, width, height, confidence, class_id, visibility
                ])

            frame_num += 1
            # if 32 > frame_num > 29: frame_num = 32
        
        print(f"Saving tracking results to {output}")
        with open(output, "w") as f:
            for result in tracker_results:
                f.write(",".join(map(str, result)) + "\n")