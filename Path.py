import numpy as np
import cv2
import time


PATH = "C:/Users/Soumik/Desktop/Task Round/Task 1/Task_1_Low.png"
START_POINT = [45,204,113]
END_POINT = [231,76,60]
OBSTACLES = [255,255,255]
NAVIGABLE = [0,0,0]

IMAGE = cv2.imread(PATH, 1)
(h,w) = IMAGE.shape[:2]

visited = np.zeros((h,w), np.uint8)
parent = np.zeros((h,w,2), np.uint8)


def main():

    START_TIME=time.time()

    img = np.copy(IMAGE)

    start = (0,0)
    end = (0,0)

    for i in range(h):
        for j in range(w):
            
            if (img[i,j] == START_POINT).any():
                start = (i,j)

            if (img[i,j] == END_POINT).any():
                end = (i,j)

    # print(start)
    # print(end)


    dijkstra1(img,start,end) #116.42868208885193s #181
    # dijkstra2(img,start,end) #130.20893049240112s #124

    # astar_adm1(img,start,end) #48.226601123809814s #187
    # astar_adm2(img,start,end) #67.60557842254639s #141

    # astar_nonadm1(img,start,end) #51.77106857299805s #191
    # astar_nonadm2(img,start,end) #38.427468061447144s #124

    # astar_euclidean1(img,start,end) #91.35817074775696s #191
    # astar_euclidean2(img,start,end) #73.28083443641663s #124

    # astar_diagonal1(img,start,end) #78.97302865982056s #191
    # astar_diagonal2(img,start,end) #70.28685092926025s #126

    # astar_manhattan1(img,start,end) #94.38470268249512s #197
    # astar_manhattan2(img,start,end) #76.88299560546875s #124
    
    
    END_TIME=time.time()

    cost=0

    current=end
    for i in range(h):
        for j in range(w):
            if (visited[i,j]==1):
                img[i,j]=[0,0,255]
                
            if (visited[i,j]==2):
                img[i,j]=[10,245,245]
                

    while (current!=start):
        point = parent[current][0],parent[current][1]
        img[(point[0]),(point[1])] = [0,105,33]
        current = (point[0],point[1])
        cost+=1

    img1=upScale(IMAGE, (1000,1000))
    img=upScale(img, (1000,1000))
    

    cv2.namedWindow('Path1',cv2.WINDOW_NORMAL)
    cv2.namedWindow('Path2',cv2.WINDOW_NORMAL)
    cv2.imshow('Path1',img1)
    cv2.imshow('Path2',img)
    cv2.waitKey(0)
  
    print("Time Consumed= {}s".format(END_TIME-START_TIME))
    print("Cost of the final path= {}".format(cost))


def upScale(img, shape):
    
    img1 = img.copy()
    (h, w) = img.shape[:2] # (h,w)=(100,100)
    (h1, w1) = shape[:2] # (h1, w1)=(1000,1000)
    h2 = h1 // h # 10
    w2 = w1 // w # 10
    img_big = np.ndarray((h1,w1,3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            img_big[i*h2: i*h2 + h2, j*w2:j*w2 + w2] = img1[i, j]
    return img_big


def isValid(img,x,y): #checking whether the point lies inside the image
    return (x>=0 and x<img.shape[0] and y >=0 and y<img.shape[1])

def euclidean(point,current): #euclidean distance
    return ((point[0] - current[0])**2 + (point[1] - current[1])**2)**0.5

def manhattan(point,current): #manhattan distance
    return (abs(point[0] - current[0]) + abs(point[1] - current[1]))

def dist_diag(point,current): #diagonal distance
    return (max(abs(current[0]-point[0]) , abs(current[1]-point[1])))

def adm(point,current): #diagonal distance
    return (min(abs(current[0]-point[0]) , abs(current[1]-point[1])))


def dijkstra1(img,start,end): #Case 1

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                if ((i!=-1 and i!=1) or (j!=-1 and j!=1)):
                    point= (current[0]+i,current[1]+j)
                    #print(point)
                    if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                        if (euclidean(point,current)+dist[current] < dist[point]):
                            dist[point]=euclidean(point,current)+dist[current]
                            visited[point]=1
                            parent[point]=[current[0],current[1]]

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]<min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()

def dijkstra2(img,start,end): #Case 2

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                point= (current[0]+i,current[1]+j)
                #print(point)
                if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                    if (euclidean(point,current)+dist[current] < dist[point]):
                        dist[point]=euclidean(point,current)+dist[current]
                        visited[point]=1
                        parent[point]=[current[0],current[1]]

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]<min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()
    

def astar_adm1(img,start,end): #Case 1

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                if ((i!=-1 and i!=1) or (j!=-1 and j!=1)):
                    point= (current[0]+i,current[1]+j)
                    #print(point)
                    if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                        if (euclidean(point,current)+dist[current]+ adm(point,end) < dist[point]):
                            dist[point]=euclidean(point,current)+dist[current]+ adm(point,end)
                            visited[point]=1
                            parent[point[0],point[1]]=[current[0],current[1]]

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]<min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()


def astar_adm2(img,start,end): #Case 2

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                point= (current[0]+i,current[1]+j) #(3,2)
                #print(point)
                if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                    if (euclidean(current,point)+dist[current]+ adm(point,end) < dist[point]):
                        dist[point]=euclidean(current,point)+dist[current]+ adm(point,end)
                        visited[point]=1
                        parent[point]=[current[0],current[1]]
                        

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]< min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()



def astar_nonadm1(img,start,end): #Case 1

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                if ((i!=-1 and i!=1) or (j!=-1 and j!=1)):
                    point= (current[0]+i,current[1]+j)
                    #print(point)
                    if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                        if (euclidean(point,current)+dist[current]+ euclidean(point,end)**3 < dist[point]):
                            dist[point]=euclidean(point,current)+dist[current]+euclidean(point,end)**3
                            visited[point]=1
                            parent[point[0],point[1]]=[current[0],current[1]]

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]<min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()


def astar_nonadm2(img,start,end): #Case 2

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                point= (current[0]+i,current[1]+j) #(3,2)
                #print(point)
                if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                    if (euclidean(current,point)+dist[current]+ euclidean(point,end)**3 < dist[point]):
                        dist[point]=euclidean(current,point)+dist[current]+euclidean(point,end)**3
                        visited[point]=1
                        parent[point]=[current[0],current[1]]
                        

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]< min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()


def astar_euclidean1(img,start,end): #Case 1

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                if ((i!=-1 and i!=1) or (j!=-1 and j!=1)):
                    point= (current[0]+i,current[1]+j)
                    #print(point)
                    if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                        if (euclidean(point,current)+dist[current]+ euclidean(point,end) < dist[point]):
                            dist[point]=euclidean(point,current)+dist[current]+euclidean(point,end)
                            visited[point]=1
                            parent[point]=[current[0],current[1]]

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]<min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()


def astar_euclidean2(img,start,end): #Case 2

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                point= (current[0]+i,current[1]+j)
                #print(point)
                if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                    if (euclidean(point,current)+dist[current]+ euclidean(point,end) <= dist[point]):
                        dist[point]=euclidean(point,current)+dist[current]+euclidean(point,end)
                        visited[point]=1
                        parent[point]=[current[0],current[1]]

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]<min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()


def astar_manhattan1(img,start,end): #Case 1

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                if ((i!=-1 and i!=1) or (j!=-1 and j!=1)):
                    point= (current[0]+i,current[1]+j)
                    #print(point)
                    if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                        if (euclidean(point,current)+dist[current]+ manhattan(point,end) < dist[point]):
                            dist[point]=euclidean(point,current)+dist[current]+manhattan(point,end)
                            visited[point]=1
                            parent[point]=[current[0],current[1]]

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]<min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()
    
def astar_manhattan2(img,start,end): #Case 2

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                point= (current[0]+i,current[1]+j)
                #print(point)
                if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                    if (euclidean(point,current)+dist[current]+ manhattan(point,end) < dist[point]):
                        dist[point]=euclidean(point,current)+dist[current]+manhattan(point,end)
                        visited[point]=1
                        parent[point]=[current[0],current[1]]

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]<min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()


def astar_diagonal1(img,start,end): #Case 1

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                if ((i!=-1 and i!=1) or (j!=-1 and j!=1)):
                    point= (current[0]+i,current[1]+j)
                    #print(point)
                    if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                        if (euclidean(point,current)+dist[current]+ dist_diag(point,end) < dist[point]):
                            dist[point]=euclidean(point,current)+dist[current]+dist_diag(point,end)
                            visited[point]=1
                            parent[point]=[current[0],current[1]]

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]<min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()


def astar_diagonal2(img,start,end): #Case 2

    dist = np.full((h,w),fill_value=np.inf)
    dist[start]=0
    current=start
    visited[start]=2

    while (current!=end):

        visited[current]=2
        #print(current)
        for i in range(-1,2):
            for j in range(-1,2):
                point= (current[0]+i,current[1]+j)
                #print(point)
                if (isValid(img,point[0],point[1]) and visited[point]!=2 and ((img[point] != OBSTACLES).all())):
                    if (euclidean(point,current)+dist[current]+ dist_diag(point,end) < dist[point]):
                        dist[point]=euclidean(point,current)+dist[current]+dist_diag(point,end)
                        visited[point]=1
                        parent[point]=[current[0],current[1]]

        min_=np.inf
        for i in range(h):
            for j in range(w):
                if (dist[i,j]<min_ and visited[i,j]!=2):
                    min_=dist[i,j]
                    current=(i,j)
        showPath(img,start,current)

    cv2.destroyAllWindows()


def showPath(img,start,current):
    
    img1 = np.copy(img)
    while (current!=start):
        point = parent[current][0],parent[current][1]
        img1[(point[0]),(point[1])] = [255,0,0]
        current = (point[0],point[1])
        
    cv2.namedWindow('Path',cv2.WINDOW_NORMAL)
    cv2.imshow('Path',img1)
    cv2.waitKey(1)


if __name__ == '__main__':
    main()
