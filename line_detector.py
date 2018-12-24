
import cv2 
import numpy as np 
from tqdm import tqdm
from glob import glob
import random
import os
import sys

from pprint import pprint, pformat

def mkdir_if_exist_not(name):
    if not os.path.isdir(name):
        return os.makedirs(name)

def imshow(name, img, resize_factor = 0.25):
    return cv2.imshow(name,
                      cv2.resize(img,
                                 (0,0),
                                 fx=resize_factor,
                                 fy=resize_factor))



#cluster lines together
def cluster_lines(lines, threshold=100):
    clusters = []
    current = lines[0]
    for line in lines:
        if current[0] + threshold > line[0]:
            #current = (current + line)/ 2.0
            pass
        else:
            clusters.append(current)
            current = line

    clusters.append(current)
    
        
    return clusters


def intersection(line1, line2):
    r1, t1 = line1
    r2, t2 = line2
    a = np.array([
        [np.cos(t1), np.sin(t1)],
        [np.cos(t2), np.sin(t2)]
    ])
    b = np.array([[r1], [r2]])
    x, y = np.linalg.solve(a, b)
    x, y = int(np.round(x)), int(np.round(y))
    return (x, y)

def process(args, filepath=None):
    if filepath == None:
        filepath = args.filepath
        
    source = cv2.imread(filepath) 
    cv2.imwrite('source.jpg', source)

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 5)

    img = cv2.adaptiveThreshold(img,
                               255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,
                               11,
                               2)

    cv2.imwrite('threshold.jpg',img)
    if args.verbose:
        imshow( os.path.basename(filepath) + ' ' + 'threshold',img)
        cv2.waitKey(0)

    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    cv2.imwrite('edges.jpg',edges)
    
    if args.verbose:
        imshow( os.path.basename(filepath) + ' ' + 'edges',edges)
        cv2.waitKey(0)

    # interms of rho and theta values 
    lines = cv2.HoughLines(edges, 1, np.pi/180, 300)

    lines = lines[0]
    hlines, vlines = [], []
    for line in lines:
        r, t = line
        if t > 1.4 and t < 1.6:
            hlines.append(line)
        elif t > -0.1 and t < 0.1 :
            vlines.append(line)


    hlines = sorted(hlines, key=lambda x: x[0])
    vlines = sorted(vlines, key=lambda x: x[0])

    h, w = img.shape
    hlines_cluster = cluster_lines(hlines, 120)
    vlines_cluster = cluster_lines(vlines, 210)

    #image boundary points
    hlines_cluster.append(np.array([0, hlines_cluster[0][1]]))
    vlines_cluster.append(np.array([0, vlines_cluster[0][1]]))

    hlines_cluster.append(np.array([h, hlines_cluster[0][1]]))
    vlines_cluster.append(np.array([w, vlines_cluster[0][1]]))


    hlines_cluster = sorted(hlines_cluster, key=lambda x: x[0])
    vlines_cluster = sorted(vlines_cluster, key=lambda x: x[0])


    print('===hlines====\n', np.array(hlines))
    print('===vlines====\n', np.array(vlines))
    print(len(hlines), len(hlines_cluster))
    print(len(vlines), len(vlines_cluster))

    lines = np.concatenate([hlines_cluster, vlines_cluster], axis = 0)
    print(lines)
    print(h, w)

    line_overlay = source.copy()
    for r, theta in tqdm(lines): 
        a = np.cos(theta) 
        b = np.sin(theta) 
        x0, y0 = a * r, b * r  


        x1, y1 = int(x0 + 4000*(-b)), int(y0 + 4000*(a)) 
        x2, y2 = int(x0 - 4000*(-b)), int(y0 - 4000*(a)) 

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
        # (0,0,255) denotes the colour of the line to be  
        #drawn. In this case, it is red.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_overlay = np.ones((h, w, 3), dtype=np.uint8)
        """
        cv2.line(line_overlay, (x1,y1), (x2,y2), (0,0,255), 5)
        """
        cv2.putText(line_overlay,
                   '{},{}'.format(r, theta),
                   (500, 250),
                   font, 8,
                   (0, 255, 0),
                   2,)
        imshow( os.path.basename(filepath) + ' ' + "linesDetected",line_overlay) 
        cv2.waitKey(0)
        cv2.destroyWindow("linesDetected")
        """



    for vline in vlines_cluster:
        for hline in hlines_cluster:
            X = intersection(vline, hline)
            cv2.circle(line_overlay,
                       X,
                       25,
                       (0,20,0),
                       -1)


    cv2.imwrite(os.path.basename(filepath) + ' ' + 'linesDetected.jpg', line_overlay)
    if args.verbose:
        imshow( os.path.basename(filepath) + ' ' + "linesDetected",line_overlay) 
        cv2.waitKey(0)


    total_count = 0
    shapes = []
    for v1, v2 in zip(vlines_cluster, vlines_cluster[1:]):
        for h1, h2 in zip(hlines_cluster, hlines_cluster[1:]):
            try:
                x1, y1 = intersection(h1, v1)
                x2, y2 = intersection(h2, v2)

                if args.very_verbose:
                    print(h1, v1)
                    print(h2, v2)
                    print(x1, y1, x2, y2)
                    
                if x2 > line_overlay.shape[1]: x2 = line_overlay.shape[1] - 1
                if y2 > line_overlay.shape[0]: y2 = line_overlay.shape[0] - 1

                name = '({}, {}) - ({}, {})'.format(x1, y1, x2, y2)

                char = source[y1:y2, x1:x2]
                if char.shape[0] < 10 or char.shape[1] < 10:
                    print("Very small segment")
                    print(char)
                    continue
                if args.very_verbose:
                    cv2.imshow( os.path.basename(filepath) + ' ' + name, char)
                    cv2.waitKey(0)
                    cv2.destroyWindow(os.path.basename(filepath) + ' ' + name)

                total_count += 1
                shapes.append(char)
                
            except AssertionError as e:
                if 'size.width>0' in e.args[0]:
                    pass
                else:
                    raise e

    cv2.destroyAllWindows()
    return total_count, shapes
    


def write_shapes(args, shapes):
    for i, shape in enumerate(shapes):

        if args.very_verbose:
            cv2.imshow("shape", shape)
            key = cv2.waitKey(0)
        
        gray = cv2.cvtColor(shape, cv2.COLOR_BGR2GRAY)

        gray = cv2.adaptiveThreshold(gray,
                                    255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,
                                    7,
                                    2)
        if args.very_verbose:
            cv2.imshow("gray", gray)
            key = cv2.waitKey(0)

        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        if args.very_verbose:
            cv2.imshow("edges", edges)
            key = cv2.waitKey(0)

                
        op = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = op

        contour_overlay = shape.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:
                cv2.drawContours(contour_overlay, contour, -1, (0, 255, 0), 3)

        if args.very_verbose:
            cv2.imshow("contour", contour_overlay)
            key = cv2.waitKey(0)

        mkdir_if_exist_not('{}/{}'.format(args.prefix, i))
        newimg = cv2.resize(gray, (args.size, args.size))

        if args.verbose:
            cv2.imshow("shape", newimg)
            key = cv2.waitKey(0)
            
        cv2.imwrite('{}/{}/{}'.format(args.prefix, i, os.path.basename(args.filepath)), newimg)

import argparse
if __name__ == '__main__':

    filepath = random.choice(glob('../raw_data/sheets/brahmi_data/*.jpg'))
    
    parser = argparse.ArgumentParser(description='Grid-segmenter')
    parser.add_argument('-f','--filepath',
                        help='path to the image file',
                        default=filepath, dest='filepath')

    parser.add_argument('-d','--prefix-dir',
                        help='path to the image file',
                        default='output', dest='prefix')

    parser.add_argument('-s','--size',
                        help='size of the resulting shape',
                        default=120, dest='size')
    
    parser.add_argument('-v', '--verbose',
                        help='shows all the grid overlayed in input image',
                        action='store_true', default=False, dest='verbose')

    parser.add_argument('-V', '--very-verbose',
                        help='shows all the pieces of the characters',
                        action='store_true', default=False, dest='very_verbose')
        
    args = parser.parse_args()

    
    total_count, shapes = process(args)
    write_shapes(args, shapes)
