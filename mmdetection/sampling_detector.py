import numpy as np
import torch
import scipy.stats
import torch.nn as nn

class DisjointSet:
    def __init__(self):
        self.sets = []

    # ==================================
    def create_set(self, elem):
        if type(elem) == list:
            self.sets.append(elem)
        else:
            self.sets.append([elem])

    # ==================================
    def find(self,elem):
        for s in self.sets:
            if elem in s:
                return s
        # print('Could not find ', elem)
        return None


    # ==================================
    def merge(self, a, b):
        setA = self.find(a)
        setB = self.find(b)

        if setA is None or setB is None:
            return

        if setA != setB:
            setA.extend(setB)
            self.sets.remove(setB)
    # ==================================
    def __str__(self):
        string = ''

        for s in self.sets:
            string += str(s) + '\n'

        return string

class SamplingDetector():
    """Base for any sampling-based detector

    Args:
        iou: spatial affinity IoU cutoff during merging
        min_dets: minimum number of detections in each cluster
        label: label affinity measure to use, if any
        min_score: minimum known softmax score a detection must have to be considered

    """

    def __init__(self, min_dets = 2, iou = 0.8, label = None, vis = False):
        self.iou = iou
        self.label = label
        self.min_dets = min_dets
        self.visualise = vis

    def form_final(self, detections):
        #detections are in format of [logits, bbox]

        #cluster detections from each forward pass 
        det_set = DisjointSet()
        det_set = self.cluster(detections, det_set)

        #remove clusters not meeting the minimum number of detections
        det_set = self.remove_for_min(det_set)
        if (len(det_set.sets) == 0): 
            return []

        #form observations from cluster
        observations = self.form_observations(detections, det_set)

        final_detections = self.form_final_detections(observations)

        return final_detections


    def remove_for_min(self, det_set):
        #no minimum detections, just return set
        if self.min_dets == 0:
            return det_set

        remove = []
        #only keep observations that were detected to have the minimum number of detections
        for i in range(len(det_set.sets)):
            s = det_set.sets[i]
            if (len(s)) < self.min_dets:
                remove.append(s)
        for r in remove:
            det_set.sets.remove(r)

        return det_set

    
    def cluster(self, detections, det_set):
        bboxes = detections[:, -4:]

        distributions = detections[:, :-4]
        
        #create a set for every detection
        for idx in range(len(detections)):
            det_set.create_set([idx])

        #find affinity matrix
        spatial_matrix = np.asarray(self.spatial_association(bboxes, bboxes))

        if self.label == None:
           #no label association, only spatial 
            matrix = spatial_matrix
        else:
            label_matrix = np.asarray(self.label_association(distributions, distributions, self.label))
           
            #detections have same winning label
            matrix = (label_matrix) * spatial_matrix

        for i in range(len(detections)):
            #which other sets meet the threshold minimum?
            candidates = np.nonzero(matrix[i,] >= self.iou)[0]
            for c in candidates:
                if c != i:
                    det_set.merge(i, c)

        return det_set

    def spatial_association(self, old_bboxes, new_bboxes):
        #finds the IoU between bboxes
        assoc_matrix = []

        old_bboxes = np.asarray(old_bboxes)
        new_bboxes = np.asarray(new_bboxes)
       
        for idx in range(len(new_bboxes)):
            nx1 = new_bboxes[idx, -4] * np.ones(len(old_bboxes))
            ny1 = new_bboxes[idx, -3] * np.ones(len(old_bboxes))
            nx2 = new_bboxes[idx, -2] * np.ones(len(old_bboxes))
            ny2 = new_bboxes[idx, -1] * np.ones(len(old_bboxes))
           
            #find the iou with the detection bbox
            ox1 = old_bboxes[:, 0] 
            oy1 = old_bboxes[:, 1]
            ox2 = old_bboxes[:, 2]
            oy2 = old_bboxes[:, 3]

            #centroids, width and heights
            ncx = (nx1+nx2)/2
            ncy = (ny1+ny2)/2
            nw = nx2-nx1
            nh = ny2-ny1

            ocx = (ox1+ox2)/2
            ocy = (oy1+oy2)/2
            ow = ox2-ox1
            oh = oy2-oy1
            
            ### 1 is good, 0 is bad
            xx1 = np.max([nx1, ox1], axis = 0)
            yy1 = np.max([ny1, oy1], axis = 0)
            xx2 = np.min([nx2, ox2], axis = 0)
            yy2 = np.min([ny2, oy2], axis = 0)       

            w = xx2 - xx1
            h = yy2 - yy1
            w = w * (w > 0)
            h = h * (h > 0)  

            inter = w*h
            Narea = (nx2-nx1)*(ny2-ny1)
            Oarea = (ox2-ox1)*(oy2-oy1)
            union = Narea + Oarea - inter
            IoUs = inter/union
            assoc_matrix.append(IoUs)

        return assoc_matrix

    def label_association(self, old_dists, new_dists, method):
        assoc_matrix = []
        if method == None:
            return assoc_matrix

        if 'kl' in method:
            assoc_matrix = scipy.stats.entropy(new_dists.T[:, :, None], old_dists.T[:, None, :])
            return assoc_matrix

        for idx in range(len(new_dists)):
            new_dist = new_dists[idx]

            if 'label' in method:
                assoc_matrix.append(np.argmax(new_dist) == np.argmax(old_dists, axis = 1))           ### 1 is good, 0 is bad

        return assoc_matrix

    def form_observations(self, total_detections, det_set, img = None):
        #merge observations into final detections
        observations=[]

        for S in det_set.sets:
            D = []
            for detection_id in S:
                D.append(total_detections[detection_id])
            observations += [np.asarray(D)]

        return observations

    def form_final_detections(self, observations):
        detections = []
        for ob_individ in observations:
            distribution = np.mean(ob_individ[:, :-4], axis = 0)
            bbox = np.mean(ob_individ[:, -4:], axis = 0) 

            detections += [distribution.tolist() + bbox.tolist()]

        return np.asarray(detections)
