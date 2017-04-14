import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import color
from scipy import ndimage, signal
import ipdb
import Queue
import graphviz

def getColorSelectionSection(img):
    return img[1260:, 300:]

def getGameBoardSection(img):
    return img[:1209, :]

def convertRGBToLAB(colors):
    shapeModified = False
    while (len(colors.shape) < 3):
        shapeModified = True
        colors = np.expand_dims(colors, 0)
    newColor = color.rgb2lab(colors)
    if shapeModified:
        newColor = np.squeeze(newColor)
    return newColor

def convertLABToRGB(colors):
    shapeModified = False
    while (len(colors.shape) < 3):
        shapeModified = True
        colors = np.expand_dims(colors, 0)
    newColor =  color.lab2rgb(colors)
    if shapeModified:
        newColor = np.squeeze(newColor)
    return newColor

def labDistance(color1, color2):
    return color.deltaE_ciede2000(color1, color2)

def colorOptionsFromScreenshot(img, debug=False):
    colorSection = getColorSelectionSection(img)
    avgColumnColor = np.expand_dims(np.mean(colorSection, axis=0), 0)
    labColors = convertRGBToLAB(avgColumnColor)
    #compute euclidean distance between two adjacent columns
    distThreshold = 1.5
    dists = []
    colorBuckets = [[]]
    for i in range(labColors.shape[1]-1):
        prevPoint = labColors[0, i, :]
        nextPoint = labColors[0, i+1, :]
        dists.append(labDistance(prevPoint, nextPoint))

        if dists[-1] > distThreshold and len(colorBuckets[-1]) > 0:
            colorBuckets.append([])
        elif dists[-1] < distThreshold:
            colorBuckets[-1].append(prevPoint)

    colorBuckets.append([convertRGBToLAB(np.array([225.0, 225.0, 225.0])/255.0)])
    colorOptions = np.array([np.mean(bucket,axis=0) for bucket in colorBuckets])
    if debug:
        plt.figure()
        plt.plot(dists)
        plt.xlabel('Column Index')
        plt.ylabel('Distance')
        plt.title('Distance between adjacent columns')

        rgbColorOptions = convertLABToRGB(np.expand_dims(colorOptions, 0))
        plt.figure()
        plt.subplot(211)
        plt.imshow(getColorSelectionSection(img))
        plt.title('Raw Image')
        plt.subplot(212)
        plt.imshow(rgbColorOptions, interpolation='none')
        plt.title('Extracted Colors')

    return colorOptions

# Assumes that gameBoard has been zoomed
def getVerticesFromTriangleCoords(row, col, gameBoard):
    midRow = gameBoard.shape[0] / 2
    midCol = gameBoard.shape[1] / 2
    #Spacing of the vertical lines
    verticalSpacing = 11.25
    #change in y for each space
    diagonalSlope = 6.5
    # vertical separation of diagonals
    diagonalSpacing = 12.875

    # if the row is odd, the top most vertex is not shifted by a
    # half unit
    isV1Aligned = False
    if row % 2 == 1:
        isV1Aligned = True

    isBaseRight = False
    if (col + row)  % 2 == 1:
        isBaseRight = True

    # v1 is the vertex with the lowest row
    # v3 is the vertex with the highest row
    # v2 is the last vertex
    if isV1Aligned:
        v1AlignedY = (row - 1) / 2
        v1yRaw = midRow + (v1AlignedY - 7) * diagonalSpacing
        V1Y = int(np.round(v1yRaw))
        V2Y = int(np.round(v1yRaw + diagonalSpacing * .5))
        V3Y = int(np.round(v1yRaw + diagonalSpacing))
        if V2Y > gameBoard.shape[0]:
            V2Y = gameBoard.shape[0] - 1

        if V3Y > gameBoard.shape[0]:
            V3Y = gameBoard.shape[0] - 1

        if isBaseRight:
            v1xRaw = midCol + (col - 4) * verticalSpacing
            V1X = int(np.round(v1xRaw))
            V2X = int(np.round(v1xRaw - verticalSpacing))
            V3X = V1X
        else:
            v1xRaw = midCol + (col - 5) * verticalSpacing
            V1X = int(np.round(v1xRaw))
            V2X = int(np.round(v1xRaw + verticalSpacing))
            V3X = V1X

    else:
        v2AlignedY = row / 2
        v2yRaw = midRow + (v2AlignedY - 7)  * diagonalSpacing
        V1Y = max(0, int(np.round(v2yRaw - .5 * diagonalSpacing)))
        V2Y = max(0, int(np.round(v2yRaw)))
        V3Y = max(0, int(np.round(v2yRaw + .5 * diagonalSpacing)))
        if V2Y > gameBoard.shape[0]:
            V2Y = gameBoard.shape[0] - 1

        if V3Y > gameBoard.shape[0]:
            V3Y = gameBoard.shape[0] - 1

        if isBaseRight:
            v2xRaw = midCol + (col - 5) * verticalSpacing
            V1X = int(np.round(v2xRaw + verticalSpacing))
            V2X = int(np.round(v2xRaw))
            V3X = V1X
        else:
            v2xRaw = midCol + (col - 4) * verticalSpacing
            V1X = int(np.round(v2xRaw - verticalSpacing))
            V2X = int(np.round(v2xRaw))
            V3X = V1X

    return np.array([(V1X, V1Y), (V2X, V2Y), (V3X, V3Y)])

def sampleColorFromVertices(vertices, gameBoard):
    centroid = np.round(np.mean(vertices, axis=0))
    centroidRow, centroidCol = int(centroid[1]) , int(centroid[0])
    colorSample = gameBoard[centroidRow-1:centroidRow+2,
                            centroidCol-1:centroidCol+2, :]

    labSample = convertRGBToLAB(colorSample)
    labMean = np.mean(labSample, axis=(0, 1))
    return labMean

def assignColorToOption(color, options):
    dists = [labDistance(color, opt) for opt in options]
    option = np.argmin(dists)
    if option == len(options) - 1:
        option = -1
    return option

def segmentImage(img, colorOptions, debug=False):
    gameBoard = getGameBoardSection(img)
    zoomFactor = .15
    gameBoard = ndimage.zoom(gameBoard, [zoomFactor, zoomFactor, 1.], order=1)
    colors = []
    boardWidth = 10
    boardHeight = 29
    board = np.zeros((boardHeight, boardWidth))
    for row in range(boardHeight):
        colors.append([])
        for col in range(boardWidth):
            vertices = getVerticesFromTriangleCoords(row, col, gameBoard)
            for v in list(vertices):
                gameBoard[v[1], v[0], :] = [0.0, 1.0, 0.0]
            color = sampleColorFromVertices(vertices, gameBoard)
            option = assignColorToOption(color, colorOptions)
            board[row, col] = option

            colors[-1].append(colorOptions[option])

            centroid = np.round(np.mean(vertices, axis=0))
            gameBoard[int(centroid[1]), int(centroid[0])] = convertLABToRGB(colorOptions[option])
    colors = np.array(colors)

    if debug:
        plt.figure()
        plt.subplot(121)
        plt.imshow(gameBoard, interpolation='none')
        ax = plt.subplot(122)
        ax.imshow(convertLABToRGB(colors), interpolation='none')
        def format_coord(x, y):
            col = int(x+0.5)
            row = int(y+0.5)
            return 'x=%d, y=%d'%(col, row)
        ax.format_coord = format_coord
    return board

class Node:
    def __init__(self, value, position):
        self.connections = set()
        self.value = value
        self.position = position

    def addConnection(self, node):
        if self == node:
            return
        self.connections = self.connections | {node}
        node.connections = node.connections | {self}

    def removeConnection(self, node):
        self.connections = self.connections - {node}
        node.connections = node.connections - {self}

    def __repr__(self):
        return '<Node Value: {0} Pos: {1}>'.format(self.value, self.position)

    def __str__(self):
        return '<{0!r} Connections:{1!r}>'.format(self, self.connections)

# Create an adjacency list from the segments
def createBoardGraph(segs):
    numRows = segs.shape[0]
    numCols = segs.shape[1]
    nodeDict = {}
    # Create a 2d array of nodes with the relevant values
    for r in range(numRows):
        for c in range(numCols):
            if segs[r,c] >= 0:
                nodeDict[(r, c)] = Node(segs[r,c], (r,c))

    # Now add the connections between the nodes
    for r in range(numRows):
        for c in range(numCols):
            # Every node is connected to the nodes above and below it
            # (if they exists). Since the addConnection method adds the
            # node to both nodes, we only need to add the row above in
            # order to create the connections
            if (r > 0 and (r,c) in nodeDict and (r-1,c) in nodeDict):
                nodeDict[(r,c)].addConnection(nodeDict[(r-1,c)])

            # Now we add connections to right and left. Since we are working
            # our way from left to right, it is okay to only add neighbors to
            # the left of the current row. The nodes which need to have
            # connections added are the ones with the bases on the left. This
            # can be determined by adding the row and column index together
            # and checking if the result is even.
            if (c > 0 and (r + c) % 2 == 0 and 
                    (r,c) in nodeDict and (r, c-1) in nodeDict):
                nodeDict[(r,c)].addConnection(nodeDict[(r,c-1)])

    return nodeDict

def simplifyNodeDict(g, debug=False):
    step = 0
    for key in list(g.keys()):
        currNode = g.get(key, None)
        if currNode is None:
            continue

        run = True
        while run:
            sameVals = [conn for conn in currNode.connections
                            if conn.value == currNode.value]
            if len(sameVals) == 0:
                run = False
                continue
            
            for other in sameVals:
                # These are the nodes that have the same values
                # Copy over all connections
                for otherConn in other.connections:
                    other.removeConnection(otherConn)
                    currNode.addConnection(otherConn)
                    if debug:
                        print 'Step {}'.format(step),
                        print 'Removing {} from {} and adding to {}'.format(otherConn.position, 
                                                                        other.position,
                                                                        currNode.position)
                del g[other.position]
        if debug:
            drawNodeDict(g, 'step{0:03}'.format(step))
            step += 1
        

    return g

def drawNodeDict(nd, fname):
    dot = graphviz.Graph(engine='neato')
    dot.attr('node', colorscheme='accent8', style='filled',
             shape='box')
    dot.attr('edge', splines='curved')

    createdNodes = set()
    createdEdges = set()
    
    # First create all of the nodes
    for key in nd:
        n = nd[key]
        if n.value < 0:
            continue

        nodeName = '{}\r\n{}'.format(n.position, n.value)
        nodePos = '{}, {}!'.format(*n.position)
        if n not in createdNodes:
            dot.node(nodeName, pos=nodePos, fontSize='10', color=str(int(n.value)+1))
            createdNodes |= {n}

        for conn in n.connections:
            connName = '{}\r\n{}'.format(conn.position, conn.value)
            connPos = '{}, {}!'.format(*conn.position)
            if conn not in createdNodes:
                dot.node(connName, pos=connPos, fontSize='10', color=str(int(conn.value)+1))
                createdNodes |= {conn}

            if ((n, conn) in createdEdges or
                    (conn, n) in createdEdges):
                continue
            
            dot.edge(nodeName, connName)
            createdEdges |= {(n, conn)}
            
    dot.render(fname, directory='steps', cleanup=True)

def copyNodeDict(nd):
    # Create all nodes
    newNd = {}
    for key in nd:
        currNode = nd[key]
        newNd[key] = Node(currNode.value, currNode.position)

    # Now create the appropriate connections
    for key in nd:
        currNode = nd[key]
        for conn in currNode.connections:
            newNd[key].addConnection(newNd[conn.position])

    return newNd

def checkIfSolved(nd):
    values = set()
    for key in nd:
        values |= {nd[key].value}

    if len(values) > 1:
        return False
    return True

def solvePuzzle(nd, movesRemaining, moveList):
    if movesRemaining == 0 and checkIfSolved(nd):
        return moveList
    elif movesRemaining == 0:
        print moveList, len(nd.keys())
        return None

    # Sort the nodes by the number of connections they have
    nodeList = [nd[key] for key in nd]
    nodeList = sorted(nodeList, key=lambda n: len(n.connections))[::-1]

    options = set()
    for n in nodeList:
        options = options | {n.value}

    if len(options) > movesRemaining + 1:
        print moveList, 'Not enough moves remaining'
        return None

    for n in nodeList:
        options = set()
        for conn in n.connections:
            options = options | {conn.value}

        if len(options) == 0:
            for o in nodeList:
                options |= {o.value}

            options = options - {n.value}

        for o in options:
            nd2 = copyNodeDict(nd)
            nd2[n.position].value = o
            nd2 = simplifyNodeDict(nd2)
            
            answer = solvePuzzle(nd2, movesRemaining-1, moveList + [(n.position, o)])
            if answer:
                return answer

if __name__ == '__main__':
    import glob
    fileList = glob.glob('test17.tiff')
    for f in fileList:
        img = plt.imread(f)/255.0
        plt.figure()
        plt.imshow(img)
        plt.show(block=False)
        movesRemaining = int(raw_input('Enter number of allowed moves:'))
        plt.close('all')
        colorOptions = colorOptionsFromScreenshot(img, debug=False)

        segs = segmentImage(img, colorOptions, debug=True)

        g = createBoardGraph(segs)
        g = simplifyNodeDict(g)
        drawNodeDict(g, f)
        plt.show(block=False)
        solution = solvePuzzle(g, movesRemaining, [])
        print solution
        print segs

    plt.show(block=False)

