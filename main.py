# Librarys
import os, ogr, gdal
from matplotlib import pyplot as plt
import rasterio
from rasterio.enums import Resampling
from rasterio.plot import show
from rasterio import features
import rasterio.mask
from scipy.spatial import distance
import time
from rasterstats import zonal_stats, point_query
from tqdm import tqdm
import numpy as np
import math
import json
import csv
import concurrent.futures

start_time = time.time()
arm = 0

# Parameter
turbineDist = 500
gridDist = 100
calcRes = 2
minYield = 1000

# Wake Parameter
r0 = 70 # Rotor radius
z = 150 # Hub height
z0 = 0.1 # Terrain roughness
alpha = 0.5 / math.log(z / z0)


#Calc WindDir
def getWindDir0(windDir):
    windDir0 = None
    if windDir > 180:
        windDir0 = windDir -180
    else:
        windDir0 = windDir + 180
    return windDir0

# Initialization Grid
def initData(baseArea):
    baseAreaGeom = baseArea.GetGeometryRef()
    (minX, maxX, minY, maxY) = baseAreaGeom.GetEnvelope()
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minX, minY)
    ring.AddPoint(maxX, minY)
    ring.AddPoint(maxX, maxY)
    ring.AddPoint(minX, maxY)
    ring.AddPoint(minX, minY)
    baseAreaBBox = ogr.Geometry(ogr.wkbPolygon)
    baseAreaBBox.AddGeometry(ring)
    baseAreaBBoxGeoJSON = [json.loads(baseAreaBBox.ExportToJson())]
    baseAreaBBoxGeoJSON[0]["coordinates"][0][0][0] = baseAreaBBoxGeoJSON[0]["coordinates"][0][0][0] - turbineDist
    baseAreaBBoxGeoJSON[0]["coordinates"][0][0][1] = baseAreaBBoxGeoJSON[0]["coordinates"][0][0][1] - turbineDist
    baseAreaBBoxGeoJSON[0]["coordinates"][0][1][0] = baseAreaBBoxGeoJSON[0]["coordinates"][0][1][0] + turbineDist
    baseAreaBBoxGeoJSON[0]["coordinates"][0][1][1] = baseAreaBBoxGeoJSON[0]["coordinates"][0][1][1] - turbineDist
    baseAreaBBoxGeoJSON[0]["coordinates"][0][2][0] = baseAreaBBoxGeoJSON[0]["coordinates"][0][2][0] + turbineDist
    baseAreaBBoxGeoJSON[0]["coordinates"][0][2][1] = baseAreaBBoxGeoJSON[0]["coordinates"][0][2][1] + turbineDist
    baseAreaBBoxGeoJSON[0]["coordinates"][0][3][0] = baseAreaBBoxGeoJSON[0]["coordinates"][0][3][0] - turbineDist
    baseAreaBBoxGeoJSON[0]["coordinates"][0][3][1] = baseAreaBBoxGeoJSON[0]["coordinates"][0][3][1] + turbineDist
    baseAreaBBoxGeoJSON[0]["coordinates"][0][4] = baseAreaBBoxGeoJSON[0]["coordinates"][0][0]
    
    # Load Yield Raster
    yieldPath = r"Data/adjusted_yield.asc"
    yieldRaster = rasterio.open(yieldPath)
    out_image, out_transform = rasterio.mask.mask(yieldRaster, baseAreaBBoxGeoJSON, crop=True)
    
    memFile = rasterio.io.MemoryFile()
    yieldDataset = memFile.open (driver='GTiff',
            height=out_image[0].shape[0],
            width=out_image[0].shape[1],
            count=1,
            dtype=out_image.dtype,
            transform=out_transform,)
    yieldDataset.write(out_image[0], 1)
    
    
    # Resample
    upscale_factor = calcRes
    yieldData = yieldDataset.read(
        1,
        out_shape=(
            yieldDataset.count,
            int(yieldDataset.height * upscale_factor),
            int(yieldDataset.width * upscale_factor)
        ),
        resampling=Resampling.bilinear
    )
    
    # Load WindDir Raster
    windDirPath = r"Data/wind_dir_EPSG_31467.asc"
    windDirRaster = rasterio.open(windDirPath)
    out_image_windDir, out_transform_windDir = rasterio.mask.mask(windDirRaster, baseAreaBBoxGeoJSON, crop=True)
    
    memFile_windDir = rasterio.io.MemoryFile()
    windDirDataset = memFile_windDir.open (driver='GTiff',
            height=out_image_windDir[0].shape[0],
            width=out_image_windDir[0].shape[1],
            count=1,
            dtype=out_image_windDir.dtype,
            transform=out_transform_windDir,)
    windDirDataset.write(out_image_windDir[0], 1)
    
    
    # Resample
    upscale_factor = calcRes
    windDirData = windDirDataset.read(
        1,
        out_shape=(
            windDirDataset.count,
            int(windDirDataset.height * upscale_factor),
            int(windDirDataset.width * upscale_factor)
        ),
        resampling=Resampling.bilinear
    )

    # scale image transform
    transform = yieldDataset.transform * yieldDataset.transform.scale(
        (yieldDataset.width / yieldData.shape[-1]),
        (yieldDataset.height / yieldData.shape[-2])
    )
    return yieldDataset, yieldData, transform, windDirDataset, windDirData

#Plot Data

def plotData(area, layout, transform):
    fig, ax = plt.subplots()
    show(area, transform=transform, cmap='viridis', ax=ax)
    ax.scatter(layout[:, 0], layout[:, 1])
    #show(layout, cmap='viridis', ax=ax)
    
def areaEvaluate(areaGeom, calcAreaGeom):
    calcAreaValue = calcAreaGeom.Area()
    intersect = calcAreaGeom.Intersection(areaGeom)
    intersectAreaValue = intersect.Area()
    percentageArea = intersectAreaValue/calcAreaValue
    return percentageArea
    
def evaluateLayout(layout, rasterData, transform, areaGeom, bestPos, windDirData):
    wakeRasterFloat = evaluateDownwindWakeStatus (bestPos, transform, rasterData, areaGeom, windDirData)
    wakeRasterFloat = np.where(wakeRasterFloat == 0, 1, wakeRasterFloat)
    rasterData = rasterData * wakeRasterFloat
    return rasterData


def getAreaRaster(transform, rasterData, areaGeom, layout, rasterDataBase, bestPos, windDirData):
    areaJson = {'type': 'Polygon', 'coordinates': json.loads(areaGeom.ExportToJson())['coordinates']}
    rasterData = evaluateLayout(layout, rasterData, transform, areaGeom, bestPos, windDirData)
    try:
        areaRaster = features.rasterize([areaJson], out_shape=rasterData.shape, transform=transform) 
        rasterData = rasterData * areaRaster
    except: 
        areaRaster = np.where(rasterData * 0 == -0, 0, 0)
        for part in areaJson["coordinates"]:
            partAreaJson = {'type': 'Polygon', 'coordinates': part}
            areaRaster = areaRaster + features.rasterize([partAreaJson], out_shape=rasterData.shape, transform=transform) 
        rasterData = rasterData * areaRaster
    return rasterData

def calculateCone(coords, Dir, areaGeom):
    r = r0 + alpha * 6000
    
    # Polarpunktberechnung - Rotationsmatrix
    Xa = coords[0] + r0/2 * math.sin(math.radians(Dir+90))
    Ya = coords[1] + r0/2 * math.cos(math.radians(Dir+90))
    Xb = coords[0] + r0/2 * math.sin(math.radians(Dir-90))
    Yb = coords[1] + r0/2 * math.cos(math.radians(Dir-90))
    
    xHelp = coords[0] + 6000 * math.sin(math.radians(Dir))
    yHelp = coords[1] + 6000 * math.cos(math.radians(Dir))
    
    Xc = xHelp + r/2 * math.sin(math.radians(Dir+90))
    Yc = yHelp + r/2 * math.cos(math.radians(Dir+90))
    Xd = xHelp + r/2 * math.sin(math.radians(Dir-90))
    Yd = yHelp + r/2 * math.cos(math.radians(Dir-90))
    
    wakeCone = {'type': 'Polygon', 'coordinates': [[(Xa, Ya), (Xc, Yc), (Xd, Yd), (Xb, Yb), (Xa, Ya)]]}
    wakeConeGeoJSON = json.dumps(wakeCone)
    wakeGeom = ogr.CreateGeometryFromJson(wakeConeGeoJSON)
    intersect = wakeGeom.Intersection(areaGeom)
    wakeConeGeoJSON = intersect.ExportToJson()
    #wakeCone = json.loads(wakeConeGeoJSON)
    return wakeCone, wakeConeGeoJSON, wakeGeom

def evaluateDownwindWakeStatus (coords, transform, rasterData, areaGeom, windDirData):
    point0 = ogr.Geometry(ogr.wkbPoint)
    point0.AddPoint(coords[0], coords[1])
    windDir = point_query(point0.ExportToWkt(),windDirData, affine=transform,)
    windDir0 = getWindDir0(windDir[0])
    cone, coneGeoJSON, wakeGeom = calculateCone(coords, windDir0, areaGeom)
    indizes, wakeRasterFloat, percentageArea = wakeEvaluate(coords, transform, cone, rasterData, coneGeoJSON, areaGeom)
    return wakeRasterFloat

def evaluateDownwindWake (coords, transform, rasterData, areaGeom, bufferRaster, point, layout, yieldVal, windDirData):
    windDir = point_query(point.ExportToWkt(),windDirData, affine=transform,)
    windDir0 = getWindDir0(windDir[0])
    cone, coneGeoJSON, wakeGeom = calculateCone(coords, windDir0, areaGeom)
    indizes, wakeRasterFloat, percentageArea = wakeEvaluate(coords, transform, cone, rasterData, coneGeoJSON, areaGeom)
    wakeRasterFloat = wakeRasterFloat * bufferRaster
    yieldCalc = rasterData[indizes] - wakeRasterFloat[indizes] * rasterData[indizes]
    #yieldCalcVal = np.sum(yieldCalc)
    yieldCalcVal = np.mean(yieldCalc)
    #yieldCalcVal = np.sum(np.power(yieldCalc, 2))/np.sum(yieldCalc)
    #yieldCalcVal = yieldCalcVal * percentageArea
    return yieldCalcVal, wakeRasterFloat

def evaluateLayoutDownwindWake (coords, transform, rasterData, areaGeom, bufferRaster, point, layout, yieldVal, windDirData):
    windDir = point_query(point.ExportToWkt(),windDirData, affine=transform,)
    windDir0 = getWindDir0(windDir[0])
    cone, coneGeoJSON, wakeGeom = calculateCone(coords, windDir0, areaGeom)
    yieldCalcLayoutVal = 0
    for pos in layout:
        posPoint = ogr.Geometry(ogr.wkbPoint)
        posPoint.AddPoint(pos[0], pos[1])
        if wakeGeom.Intersects(posPoint):
            distPoints = posPoint.Distance(point)
            alphaX = distPoints * alpha
            r = alphaX + r0 
            fact = 1 - ((2 / 3) * math.pow((r0 / r), 2))
            yieldValLayout = point_query(point.ExportToWkt(),rasterData, affine=transform,)
            diffVal = yieldValLayout[0] - yieldValLayout[0] * fact
            yieldCalcLayoutVal = yieldCalcLayoutVal + diffVal
    return yieldCalcLayoutVal

def evaluateUpwindWake(coords, transform, rasterData, yieldVal, areaGeom, bufferRaster, point, layout, windDirData):
    windDir = point_query(point.ExportToWkt(),windDirData, affine=transform,)
    cone, coneGeoJSON, wakeGeom = calculateCone(coords, windDir[0], areaGeom)
    indizes, wakeRasterFloat, percentageArea = wakeEvaluate(coords, transform, cone, rasterData, coneGeoJSON, areaGeom)
    wakeRasterFloat = wakeRasterFloat * bufferRaster
    yieldCalc = yieldVal - wakeRasterFloat[indizes] * yieldVal
    #yieldCalcVal = np.sum(yieldCalc)
    yieldCalcVal = np.mean(yieldCalc)
    #yieldCalcVal = np.sum(np.power(yieldCalc, 2))/np.sum(yieldCalc)
    #yieldCalcVal = yieldCalcVal * percentageArea
    return yieldCalcVal

def wakeEvaluate(coords, transform, cone, rasterData, coneGeoJSON, areaGeom):
    wakeRaster = features.rasterize([cone], out_shape=rasterData.shape, transform=transform) 
    indizes = np.where(wakeRaster == 1)
    indizesY, indizesX = np.where(wakeRaster == 1)
    indizesCoordsX = (indizesX * transform[0])+transform[2]+(transform[0]/2)
    indizesCoordsY = (indizesY * transform[4])+transform[5]+(transform[0]/2)
    indizesCoords = np.vstack((indizesCoordsX, indizesCoordsY)).T
    coordsNP = np.array([coords])
    dist = distance.cdist(coordsNP, indizesCoords)
    alphaX = dist * alpha
    r = alphaX + r0 
    fact = 1 - ((2 / 3) * np.power((r0 / r), 2))
    wakeRasterFloat = wakeRaster.astype(float)
    wakeRasterFloat[indizes] = fact
    calcAreaGeom = ogr.CreateGeometryFromJson(coneGeoJSON)
    percentageArea = areaEvaluate(areaGeom, calcAreaGeom)
    return indizes, wakeRasterFloat, percentageArea
    
# Calculate Perimeter Points
def weightedMean(x):
    wMean = np.sum(np.power(x, 2))/np.sum(x)
    return wMean

def bufferEvaluate(coords, rasterData, transform, areaGeom):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(coords[0], coords[1])
    buffer = point.Buffer(turbineDist)
    bufferWkt = buffer.ExportToWkt()
    bufferGeoJSON = buffer.ExportToJson()
    bufferDict = json.loads(bufferGeoJSON)
    bufferRaster = features.rasterize([bufferDict], out_shape=rasterData.shape, transform=transform) 
    bufferRaster = np.where(bufferRaster == 1, 0, 1)
    stats = zonal_stats(bufferWkt, rasterData, affine=transform, stats=["mean"])
    finalValue = stats[0]['mean']
    #stats = zonal_stats(bufferWkt, rasterData, affine=transform, add_stats={'wMean':weightedMean})
    #percentageArea = areaEvaluate(areaGeom, buffer)
    #finalValue = stats[0]['wMean']*percentageArea
    return finalValue, buffer, bufferRaster    

# Calculate the best Position 
def getBestPoint(areaGeom, yieldRaster, layout, rasterData, transform, arm, rasterDataBase, pointNr, windDirData):
    print ('')
    xMin, xMax, yMin, yMax = areaGeom.GetEnvelope()
    insetX = -(gridDist/2)
    insetY = (gridDist/2)
    xMin = xMin + insetX
    yMax = yMax - insetY
    y = yMax
    xDiff = xMax - xMin
    yDiff = yMax - yMin
    bestVal = -9999999999999
    bestPos = []
    bestBuffer = []
    wakeRaster = None
    while y >= yMin:
        x = xMin
        while x <= xMax:
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(x, y)
            if areaGeom.Intersects(point):
                coords = [x,y]
                
                # The magic starts here:

                # Get Yield at Pos
                yieldVal = point_query(point.ExportToWkt(),rasterData, affine=transform,)
                ###### TARGETS:
                
                # Get Sum of all Pixel in Buffer
                bufferValue, buffer, bufferRaster = bufferEvaluate(coords, rasterData, transform, areaGeom)
                
                # Wake Calculations
                wakeValueDownwind, wakeRaster = evaluateDownwindWake(coords, transform, rasterData, areaGeom, bufferRaster, point, layout, yieldVal, windDirData)
                wakeValueDownwindLayout = evaluateLayoutDownwindWake(coords, transform, rasterData, areaGeom, bufferRaster, point, layout, yieldVal, windDirData)
                wakeValueUpwind = evaluateUpwindWake(coords, transform, rasterData, yieldVal, areaGeom, bufferRaster, point, layout, windDirData)
                meanWakeValue = (wakeValueDownwind + wakeValueDownwindLayout + wakeValueUpwind)/3
                #######
                # Calculate Position-Value 
                posVal = yieldVal[0] - bufferValue - meanWakeValue
                #print(posVal)
                # Check best Value
                if posVal > bestVal and yieldVal[0] > minYield:
                    bestVal = posVal
                    bestPos = coords
                    bestBuffer = buffer
                    
                elif posVal == bestVal:
                    arm = arm + 1 
            x += gridDist
        y = y - gridDist

    if type(wakeRaster) == float:
        wakeRaster[wakeRaster == 0] = 1
        rasterData = rasterData * wakeRaster
    # if len(bestPos)>0:
    #     rasterData = getAreaRaster(transform, rasterData, areaGeom, layout, rasterDataBase, bestPos)  
    return bestVal, bestPos, bestBuffer, rasterData
    

def calculateLayout(calcAreaGeom, yieldRaster, layout, rasterData, transform, arm, calcAreaValue, baseAreaValue, rasterDataBase, baseArea, windDirData):
    pointNr = 1
    with tqdm(total=baseAreaValue, position = 1, desc = "Area FID "+str(baseArea.GetFID())+"", leave = True) as pBarArea:
        while calcAreaValue > 0:
            bestVal, bestPos, bestBuffer, rasterData = getBestPoint(calcAreaGeom, yieldRaster, layout, rasterData, transform, arm, rasterDataBase, pointNr, windDirData)
            calcAreaValueTemp = calcAreaValue
            if len(bestPos)>0:
                rasterData = getAreaRaster(transform, rasterData, calcAreaGeom, layout, rasterDataBase, bestPos, windDirData)  
            try:
                calcAreaGeom = calcAreaGeom.Difference(bestBuffer)
            except:
                print('Erase impossible')
                break
            calcAreaValue = calcAreaGeom.GetArea()
            layout.append(bestPos)
            layoutPlot = np.array(layout)
            plotData(rasterData,layoutPlot, transform)
            pBarArea.update(int(calcAreaValueTemp - calcAreaValue + 0.5))
            pointNr = pointNr + 1
    return layout

def processArea(baseArea):
    layout = []
    baseAreaGeom = baseArea.GetGeometryRef()
    baseAreaValue = baseAreaGeom.GetArea()
    calcAreaValue = baseAreaValue
    calcAreaGeom = baseAreaGeom
    calcArea = baseArea
    yieldRaster, rasterData, transform, windDirRaster, windDirData = initData(calcArea)
    rasterDataBase = rasterData
    areaJson = {'type': 'Polygon', 'coordinates': json.loads(calcAreaGeom.ExportToJson())['coordinates']}
    areaRaster = features.rasterize([areaJson], out_shape=rasterData.shape, transform=transform) 
    rasterData = rasterData * areaRaster
    rasterData = np.where (rasterData == -0 , 0, rasterData)
    print('')
    layout = calculateLayout(calcAreaGeom, yieldRaster, layout, rasterData, transform, arm, calcAreaValue, baseAreaValue, rasterDataBase, baseArea, windDirData)
    print('')
    print('Number of turbines Layout FID '+str(baseArea.GetFID())+': ', len(layout)) 
    print('Number of arms: ', arm) 
    layout = np.array(layout)
    plotData(rasterData, layout, transform)
    csv_header = ["x","y"]
    with open('Results/Layout_FID_'+str(baseArea.GetFID())+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for row in layout:
            writer.writerow(row)


# Main
def main():
    print ('Full Stock WFLO-Algorithm')
    # Load Area 
    #areasPath = r"Data/area_EPSG_31476_small.shp"
    areasPath = r"Data/area_Testflaeche.shp"
    areasDs = ogr.Open(areasPath, 0)
    areasLayer = areasDs.GetLayer()
    featureCount = areasLayer.GetFeatureCount()
    with concurrent.futures.ThreadPoolExecutor() as executer:
        executer.map(processArea, areasLayer)
                
    print('')
    print("--- Total Calculation Time: %s seconds ---" % (time.time() - start_time))
 

if __name__ == "__main__":
    main()