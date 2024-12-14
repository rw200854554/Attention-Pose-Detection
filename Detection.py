from ultralytics import YOLO
import numpy as np
import json
import os

class Detector:
    def __init__(self,model:str,file_path:str):
        """
        :param model: path to the model
        :param file_path: path to the file where the record will be stored
        :param fps: frame rate of the input
        """
        try:
            self.model = YOLO(model)
        except FileNotFoundError:
            print("model not found at",model,". Using default model: yolo11n-pose.pt")
            self.model = YOLO("yolo11n-pose.pt")
        self.pose = None
        self.file_path = file_path
        self.t = 0
        self.avg_t = 0
        self.it = 0
        self.last_save = 0
        #import existing record file if a valid one exist
        if file_path is not None:
            try:
                with open(file_path, 'r') as json_file:
                    self.pose = json.load(json_file)
                    try:
                        print("sit:",self.pose["sit"],"stand:",self.pose["stand"],"lying:",self.pose["lying"],"look_away:",self.pose["look_away"],"look_at:",self.pose["look_at"],"distracted:",self.pose["distracted"],"14days_avg_action",self.pose["14days_avg_action"][0],self.pose["last_day"][0])
                    except KeyError:
                        print("JSON file in wrong format, creating new JSON file at action_data/pose_data.json")
                        self.pose = None
            except FileNotFoundError:
                print("The file does not exist! Creating new JSON file at action_data/pose_data.json")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}. Creating new JSON file at action_data/pose_data.json")
        # create a new record if no existing record
        if self.pose is None:
            self.pose = {}
            self.pose["sit"] = 0
            self.pose["stand"] = 0
            self.pose["lying"] = 0
            self.pose["look_away"] = 0
            self.pose["last_day"] = [0,0,0]
            self.pose["look_at"] = 0
            self.pose["distracted"] = False
            self.pose["14days_avg_action"] = [0,0,0] # [stand,sit, lying]

            #update file path
            if self.file_path is None:
                self.file_path = "action_data/pose_data.json"
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w') as json_file:
                json.dump(self.pose, json_file, indent=4)

    def detect(self,source,stream=True,show=False,fps=30):
        """
        :param source: path to the input or 0 for webcam
        :param stream: process the input as a stream
        :param show: show the detection
        :return: Void
        """
        results = self.model(source=source,show=show,stream=stream)

        for result in results:
            d = result.keypoints.data.cpu().detach().numpy()
            t = 1/fps
            self.t+=t
            self.it +=1
            self.avg_t = self.t/self.it
            # get position of body part using mean of the keypoints
            head_pos = d[0, :3].mean(axis=0)
            body_pos = d[0, 11:13].mean(axis=0)
            knees_pos = d[0, 13:15].mean(axis=0)
            feet_pos = d[0, 15:].mean(axis=0)

            #computation for the yaw and roll of the user's head
            nose = d[0, 0]
            l_eye = d[0, 1]
            r_eye = d[0, 2]
            eye_center = (l_eye + r_eye) / 2
            hvec = l_eye - r_eye
            hvec = hvec / np.linalg.norm(hvec)
            vvec = nose - eye_center
            vvec = vvec / np.linalg.norm(vvec)
            roll = np.arctan2(hvec[1], hvec[0]) * 180 / np.pi  # Degrees
            yaw = np.arcsin(vvec[1]) * 180 / np.pi  # Degrees

            # if face shifted left or right for more than 40 degrees
            if abs(yaw) > 40:
                self.pose["look_away"] += t
                # reset the attention counter
                self.pose["look_at"] = 0
                if self.pose["look_away"] > 10:
                    self.pose["distracted"] = True
            else:
                self.pose["look_at"] += t
                # reset the distraction counter if user restarts looking at screen
                if self.pose["look_at"] > 5:
                    self.pose["look_away"] = 0
                    self.pose["distracted"] = False
            # get the position relationship between each body part
            h2b = head_pos - body_pos
            h2b[0] = abs(h2b[0])
            b2k = body_pos - knees_pos
            b2k[0] = abs(b2k[0])
            k2f = knees_pos - feet_pos
            k2f[0] = abs(k2f[0])
            # determining the action
            if h2b[0] > h2b[1] and b2k[0] > b2k[1]:
                self.pose["lying"] += t

            elif h2b[1] > h2b[0] and b2k[1] > b2k[0] and k2f[1] > k2f[0]:
                self.pose["stand"] += t
            elif h2b[1] > h2b[0] and b2k[0] > b2k[1]:
                self.pose["sit"] += t
            #record a 14 days average of the action data
            if self.t>=86400:
                self.t = self.t%86400
                self.it = 0
                self.last_save = 0
                self.pose["14days_avg_action"] = [*((1/14)*np.array([self.pose["stand"],self.pose["sit"],self.pose["lying"]])+(13/14)*np.array(self.pose["14days_avg_action"]))]
                self.pose["last_day"] = [self.pose["stand"],self.pose["sit"],self.pose["lying"]]
                self.pose["stand"] = 0
                self.pose["sit"] = 0
                self.pose["lying"] = 0
                with open(self.file_path, 'w') as json_file:
                    json.dump(self.pose, json_file, indent=4)
                self.last_save = self.t
            # save when the current inference time is quicker than the average inference time or every 5 seconds
            if t<self.avg_t or self.t-self.last_save>5:
                with open(self.file_path, 'w') as json_file:
                    json.dump(self.pose, json_file, indent=4)
                self.last_save = self.t
        with open(self.file_path, 'w') as json_file:
            json.dump(self.pose, json_file, indent=4)
        self.last_save = self.t