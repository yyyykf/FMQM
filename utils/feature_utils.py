import pandas as pd
import os
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
from math import sqrt

def RGB_to_Lab(R, G, B):
    R, G, B = R / 255, G / 255, B / 255

    # Inverse Gamma correction
    R = ((R > 0.0404482362771076) * ((R + 0.055) / 1.055) ** 2.4 + (R <= 0.0404482362771076) * (R / 12.92))
    G = ((G > 0.0404482362771076) * ((G + 0.055) / 1.055) ** 2.4 + (G <= 0.0404482362771076) * (G / 12.92))
    B = ((B > 0.0404482362771076) * ((B + 0.055) / 1.055) ** 2.4 + (B <= 0.0404482362771076) * (B / 12.92))

    # MATLAB Transform
    X = 0.412396 * R + 0.357583 * G + 0.180493 * B
    Y = 0.212586 * R + 0.71517 * G + 0.0722005 * B
    Z = 0.0192972 * R + 0.119184 * G + 0.950497 * B

    # MATLAB White point
    Xo, Yo, Zo = 0.950456, 1.000000, 1.088754

    L = 116.0 * F(Y / Yo) - 16.0  # maximum L = 100
    a = 500.0 * (F(X / Xo) - F(Y / Yo))
    b = 200.0 * (F(Y / Yo) - F(Z / Zo))

    return L, a, b


def F(input):
    return np.where(input > 0.008856, input ** (1 / 3), (841 / 108) * input + 4 / 29)


def calculateSimilarity(refFeature, disFeature):
    eps = 1e-20
    refFeatureMean = np.mean(refFeature, axis=0)
    disFeatureMean = np.mean(disFeature, axis=0)
    magnetDiff = abs(2 * refFeatureMean * disFeatureMean + eps) / (refFeatureMean ** 2 + disFeatureMean ** 2 + eps)

    refStdvar = np.std(refFeature, axis=0)
    disStdvar = np.std(disFeature, axis=0)
    stdvarDiff = (2 * refStdvar * disStdvar + eps) / (refStdvar ** 2 + disStdvar ** 2 + eps)

    featureCov = abs(np.cov(refFeature, disFeature, ddof=0)[0][1])
    covDiff = (featureCov + eps) / (refStdvar * disStdvar + eps)
    return magnetDiff, stdvarDiff, covDiff


def getColorGradients(sdfSampleXYZ, sdfGradient, sdfSampleColor, tangentDirThreshold, usedSampleIdx):
    numberSamples = sdfSampleXYZ.shape[0]
    # color gradient: max_gradient_direction max_gradient_value
    colorGradientInfo = np.zeros((numberSamples, 4, 3))
    for idx in usedSampleIdx:
        selectedXYZ = sdfSampleXYZ[idx, :]
        selectedNormal = sdfGradient[idx, :]
        selectedcolor = sdfSampleColor[idx, :]
        
        xyzOffset = sdfSampleXYZ - selectedXYZ
        xyzOffset = np.delete(xyzOffset, idx, 0)
        xyzOffsetLength = np.linalg.norm(xyzOffset, axis=1)
        xyzOffsetNormalized = xyzOffset / xyzOffsetLength.reshape(-1, 1)
        colorOffset = sdfSampleColor - selectedcolor
        colorOffset = np.delete(colorOffset, idx, 0)

        # first choose tanget plane samples
        cosValues = np.sum(selectedNormal * xyzOffsetNormalized, axis=1)
        tangentDirIndex = np.where(abs(cosValues) < tangentDirThreshold)
        tangentOffsetLength = xyzOffsetLength[tangentDirIndex]
        if tangentOffsetLength.shape[0] == 0:
            continue
        tangentXYZOffset = xyzOffsetNormalized[tangentDirIndex]
        tangentColorOffset = colorOffset[tangentDirIndex]

        # find the max gradient direction
        gradientValues = tangentColorOffset / tangentOffsetLength.reshape(-1, 1)
        # find max of each column of gradientValues
        maxIndices = np.argmax(gradientValues, axis=0)
        # find the max gradient direction
        maxGradientDirection = tangentXYZOffset[maxIndices]
        maxGradientValue = gradientValues[maxIndices]
        for tt in range(3):
            indicies = maxIndices[tt]
            maxGradientDirection = tangentXYZOffset[indicies]
            maxGradientValue = gradientValues[indicies, tt]
            colorGradientInfo[idx, :, tt] = np.array([maxGradientDirection[0], maxGradientDirection[1], maxGradientDirection[2], maxGradientValue])
    
    return colorGradientInfo
  
def getColorGradientDiff(refColorGradientInfo, disColorGradientInfo):
    eps = 1e-20
    colorGradientDiff = np.zeros((3, 2))
    for i in range(3):
        ciRefInfo = refColorGradientInfo[:, :, i]
        ciDisInfo = disColorGradientInfo[:, :, i]
        
        cosAngle = np.sum(ciRefInfo[:, 0:3] * ciDisInfo[:, 0:3], axis=1)
        colorGradientAngle = np.mean(cosAngle ** 2) ** (1/2)
        
        refG = ciRefInfo[:, 0:3] * ciRefInfo[:, 3].reshape(-1, 1)
        refGN = np.linalg.norm(refG, axis=1)
        disG = ciDisInfo[:, 0:3] * ciDisInfo[:, 3].reshape(-1, 1)
        disGN = np.linalg.norm(disG, axis=1)
        colorGradientSMAPE = np.mean(abs(refGN**2 - disGN**2 + eps) / (refGN**2 + disGN**2 + eps))
        colorGradientDiff[i, 0] = colorGradientAngle
        colorGradientDiff[i, 1] = colorGradientSMAPE
    colorGradientDiff = np.mean(colorGradientDiff, axis=0)
    return colorGradientDiff

def calculateSDFFeatures(localRefSDFInfo, localDisSDFInfo, tangentThreLevel, 
                         normalThreLevel, tangentPlaneLevel, radius):
    debug = False
    colorUsed = "RGB"
    tangentDirBase = 0.01
    normalDirBase = 0.01
    tangentDirThreshold = tangentThreLevel * tangentDirBase
    normalDirThreshold = 1 - normalThreLevel * normalDirBase
    tangentPlaneNumber = tangentPlaneLevel * 10
    number_samples = localRefSDFInfo.shape[0]
    
    # maxNumber = 10000
    # if number_samples > maxNumber:
    #     localRefSDFInfo = localRefSDFInfo[0:maxNumber, :]
    #     localDisSDFInfo = localDisSDFInfo[0:maxNumber, :]
    #     number_samples = maxNumber
    if tangentPlaneNumber == 0:
        tangentPlaneNumber = number_samples
    
    if debug:
        tangentPlaneNumber = number_samples

    # geometry features
    refSDF = localRefSDFInfo[:, 12]
    disSDF = localDisSDFInfo[:, 12]
    sdfMagnetDiff, sdfStdvarDiff, sdfCovDiff = calculateSimilarity(refSDF, disSDF)
    sdfSSIM = (sdfMagnetDiff * sdfStdvarDiff * sdfCovDiff) ** (1/3)
    
    sdfSampleXYZ = localRefSDFInfo[:, 0:3]
    refNearestNormal = localRefSDFInfo[:, 9:12]
    disNearestNormal = localDisSDFInfo[:, 9:12]

    refSDFGradient = -np.sign(refSDF).reshape(-1, 1) * refNearestNormal
    disSDFGradient = -np.sign(disSDF).reshape(-1, 1) * disNearestNormal
    cosAngle = np.sum(refSDFGradient * disSDFGradient, axis=1)
    sdfGradientAngle = np.mean(cosAngle ** 2) ** (1/2)

    # color info
    refRed = localRefSDFInfo[:, 6]
    refGreen = localRefSDFInfo[:, 7]
    refBlue = localRefSDFInfo[:, 8]
    refL, refa, refb = RGB_to_Lab(refRed, refGreen, refBlue)

    disRed = localDisSDFInfo[:, 6]
    disGreen = localDisSDFInfo[:, 7]
    disBlue = localDisSDFInfo[:, 8]
    disL, disa, disb = RGB_to_Lab(disRed, disGreen, disBlue)
    if colorUsed == "RGB":
        refColor = np.vstack((refRed, refGreen, refBlue)).T
        disColor = np.vstack((disRed, disGreen, disBlue)).T
    elif colorUsed == "LAB":
        refColor = np.vstack((refL, refa, refb)).T
        disColor = np.vstack((disL, disa, disb)).T

    # color consistency, first all color
    C1MagnetDiff, C1StdvarDiff, C1CovDiff = calculateSimilarity(refColor[:, 0], disColor[:, 0])
    C2MagnetDiff, C2StdvarDiff, C2CovDiff = calculateSimilarity(refColor[:, 1], disColor[:, 1])
    C3MagnetDiff, C3StdvarDiff, C3CovDiff = calculateSimilarity(refColor[:, 2], disColor[:, 2])
    color1SSIM = (C1MagnetDiff * C1StdvarDiff * C1CovDiff) ** (1 / 3)
    color2SSIM = (C2MagnetDiff * C2StdvarDiff * C2CovDiff) ** (1 / 3)
    color3SSIM = (C3MagnetDiff * C3StdvarDiff * C3CovDiff) ** (1 / 3)
    colorALLSSIM = np.mean([color1SSIM, color2SSIM, color3SSIM])
    
    usedSampleIdx = np.sort(np.random.choice(number_samples, size=tangentPlaneNumber, replace=False))

    refColorGradientInfo = getColorGradients(sdfSampleXYZ, refSDFGradient,  refColor, tangentDirThreshold, usedSampleIdx)
    disColorGradientInfo = getColorGradients(sdfSampleXYZ, disSDFGradient,  disColor, tangentDirThreshold, usedSampleIdx)
    colorGradientDiff = getColorGradientDiff(refColorGradientInfo, disColorGradientInfo)
    colorGradientDiff[1] = 1 - colorGradientDiff[1]
    
    # color consistency and colorfulness, normal and tangent
    colorFeatureN = np.zeros((number_samples, 1))
    colorFeatureT = np.zeros((number_samples, 1))

    # for idx in np.random.choice(number_samples, size=tangentPlaneNumber, replace=False):
    for idx in usedSampleIdx:
        refNormalDir = refSDFGradient[idx, :]
        rdVector = sdfSampleXYZ - sdfSampleXYZ[idx, :]
        rdVector_length = np.linalg.norm(rdVector, axis=1)
        rdVectorNormalized = rdVector / np.linalg.norm(rdVector, axis=1).reshape(-1, 1)
        refCosValues = np.sum(refNormalDir * rdVectorNormalized, axis=1)
        normalDirIndex = list(np.where(abs(refCosValues) > normalDirThreshold)[0])       
        tangentDirIndex = list(np.where(abs(refCosValues) < tangentDirThreshold)[0])
        rdRadius = radius
        pointInRadius = list(np.where(rdVector_length < rdRadius)[0])
    
        normalDirIndex = list(set(normalDirIndex).intersection(pointInRadius))
        tangentDirIndex = list(set(tangentDirIndex).intersection(pointInRadius))
        
        normalDirIndex.append(idx)
        tangentDirIndex.append(idx)
        normalDirIndex.sort()
        tangentDirIndex.sort()

        # normal color consistency
        refNormalColor = refColor[normalDirIndex]
        disNormalColor = disColor[normalDirIndex]
        if refNormalColor.shape[0] > 0:
            C1MagnetDiff_N, C1StdvarDiff_N, C1CovDiff_N = calculateSimilarity(refNormalColor[:, 0],
                                                                              disNormalColor[:, 0])
            C2MagnetDiff_N, C2StdvarDiff_N, C2CovDiff_N = calculateSimilarity(refNormalColor[:, 1],
                                                                              disNormalColor[:, 1])
            C3MagnetDiff_N, C3StdvarDiff_N, C3CovDiff_N = calculateSimilarity(refNormalColor[:, 2],
                                                                              disNormalColor[:, 2])
            color1SSIM_N = (C1MagnetDiff_N * C1StdvarDiff_N * C1CovDiff_N) ** (1 / 3)
            color2SSIM_N = (C2MagnetDiff_N * C2StdvarDiff_N * C2CovDiff_N) ** (1 / 3)
            color3SSIM_N = (C3MagnetDiff_N * C3StdvarDiff_N * C3CovDiff_N) ** (1 / 3)
            colorSSIM_N = np.mean([color1SSIM_N, color2SSIM_N, color3SSIM_N])
            colorFeatureN[idx, :] = colorSSIM_N

        # tangent color coklorfulness
        refTangentColor = refColor[tangentDirIndex]
        disTangentColor = disColor[tangentDirIndex]
        if refTangentColor.shape[0] > 0:
            C1MagnetDiff_T, C1StdvarDiff_T, C1CovDiff_T = calculateSimilarity(refTangentColor[:, 0],
                                                                              disTangentColor[:, 0])
            C2MagnetDiff_T, C2StdvarDiff_T, C2CovDiff_T = calculateSimilarity(refTangentColor[:, 1],
                                                                              disTangentColor[:, 1])
            C3MagnetDiff_T, C3StdvarDiff_T, C3CovDiff_T = calculateSimilarity(refTangentColor[:, 2],
                                                                              disTangentColor[:, 2])
            color1SSIM_T = (C1MagnetDiff_T * C1StdvarDiff_T * C1CovDiff_T) ** (1 / 3)
            color2SSIM_T = (C2MagnetDiff_T * C2StdvarDiff_T * C2CovDiff_T) ** (1 / 3)
            color3SSIM_T = (C3MagnetDiff_T * C3StdvarDiff_T * C3CovDiff_T) ** (1 / 3)
            colorSSIM_T = np.mean([color1SSIM_T, color2SSIM_T, color3SSIM_T])
            colorFeatureT[idx, :] = colorSSIM_T
    colorFeatureN = colorFeatureN[(colorFeatureN != 0).any(axis=1)]
    colorFeatureT = colorFeatureT[(colorFeatureT != 0).any(axis=1)]
    if colorFeatureN.shape[0] == 0:
        colorFeatureN = np.ones((1, 1))
    if colorFeatureT.shape[0] == 0:
        colorFeatureT = np.ones((1, 1))
    
    colorNSSIM = np.mean(colorFeatureN)
    colorTSSIM = np.mean(colorFeatureT)
    colorSSIM = sqrt(colorNSSIM * colorTSSIM)
    
    # geoAll = (sdfSSIM * sdfGradientAngle)**(1/2)
    # colorGAll = (colorGradientDiff[0] * colorGradientDiff[1]) ** (1/2)
    # colorAll = sqrt(colorSSIM * colorGAll)
    # colorAll_1 = (colorNSSIM * colorTSSIM * colorGradientDiff[0] * colorGradientDiff[1])**(1/4)
    # colorAll_2 = (colorNSSIM * colorTSSIM * colorGAll)**(1/3)
    # final1 = (sdfSSIM * sdfGradientAngle * colorNSSIM * colorTSSIM * colorGAll)**(1/5)
    # final2 = (geoAll * colorAll_2)**(1/2)
    # final3 = (sdfSSIM * sdfGradientAngle * (colorNSSIM * colorTSSIM)**(1/2) * colorGAll)**(1/4)
    # final4 = (sdfSSIM * sdfGradientAngle * colorNSSIM * colorTSSIM * colorGradientDiff[0] * colorGradientDiff[1])**(1/6)
    # final5 = (geoAll * colorNSSIM * colorTSSIM * colorGAll)**(1/4)
    # localFeature = np.hstack((sdfSSIM, sdfGradientAngle, colorALLSSIM, colorNSSIM, colorTSSIM, \
    #     colorGradientDiff[0], colorGradientDiff[1], geoAll, colorGAll, colorAll_1, colorAll_2,\
    #     final1, final2, final3, final4, final5, colorSSIM, colorAll))
    feature_geo = sdfSSIM
    feature_geo_grad = sdfGradientAngle
    feature_color = colorSSIM
    feature_color_grad = (colorGradientDiff[0] * colorGradientDiff[1]) ** (1/2)
    if sdfSSIM < 1:
        feature_fused = (feature_geo * feature_geo_grad * feature_color * feature_color_grad) ** (1/4)
    else:
        feature_fused = (feature_color * feature_color_grad) ** (1/2)
    localFeature = np.hstack((feature_geo, feature_geo_grad, feature_color, feature_color_grad, feature_fused))
    return localFeature