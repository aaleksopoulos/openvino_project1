class Tracked_Person():
    '''
    Keeps track of the tracked person
    '''
    FRAMERATE = 10
    _class_counter = 0
    def __init__(self, x1, x2, y1, y2, frame_in=-1, frame_out=-1, alert=False, disappearedFrames=0, _maxDisappearedFrames=10*FRAMERATE):
        '''
        initialize the general object to be found
        '''
        #placeholder for the next id of the tracked person
        self.personId = Tracked_Person._class_counter
        self.__class__._class_counter +=1
        #to indicate if this person was tracked or not
        self.tracked = True
        #at what frame we tracked the object, default -1
        self.frame_in = frame_in
        #at what frame we lost track of it, default -1
        self.frame_out = frame_out

        #get the corner coordinates of each tracked bounding box
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        #calculate the centroid of the person tracked
        self.centroid = (int((x1+x2)/2), int((y1+y2)/2))

        #if the user in the frame for too long
        self.alert = alert

        #how many frames the person has been disappeared from the list (default 10 secs)
        self.disappearedFrames = disappearedFrames

        #number of frames to delete the current person
        self._maxDisappearedFrames = _maxDisappearedFrames

    def setX1(self, x1):
        self.x1 = x1

    def setX2(self, x2):
        self.x2 = x2

    def setY1(self, y1):
        self.y1 = y1

    def setY2(self, y2):
        self.y2 = y2

    def getX1(self):
        return self.x1

    def getX2(self):
        return self.x2

    def getY1(self):
        return self.y1

    def getY2(self):
        return self.y2

    def setTracked(self, tracked):
        self.tracked = tracked
    
    def getArea(self):
        return (abs(self.x1-self.x2) * abs(self.y1-self.y2))

    def isTracked(self):
        return self.tracked
    
    def setCentroid(self, centroid):
        self.centroid = centroid
    
    def getCentroid(self):
        return self.centroid
    
    def updateCentroid(self):
        self.centroid = (int((self.x1+self.x2)/2), int((self.y1+self.y2)/2))

    def setFrameIn(self, frame_in):
        self.frame_in = frame_in
    
    def getFrameIn(self):
        return self.frame_in

    def setFrameOut(self, frame_out):
        self.frame_out = frame_out
    
    def getFrameOut(self):
        return self.frame_out

    def setPeronId(self,personId):
        self.personId = personId

    def getPersonId(self):
        return self.personId

    def toString(self):
        s = "Id:" + str(self.personId)
        return s

    def setAlert(self, alert):
        self.alert = alert
    
    def hasAlert(self):
        return self.alert

    def setDisappearedFrames(self, disappearedFrames):
        self.disappearedFrames = disappearedFrames
    
    def getDisappearedFrames(self):
        return self.disappearedFrames