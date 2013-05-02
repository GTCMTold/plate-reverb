#!/usr/bin/env python
#
#
#       reverb.py
#
#
#       description:
#       This is an offline artificial reverb based on Jon Dattorro's
#       plate reverb
#
#

import argparse
import os
from scikits.audiolab import wavread, wavwrite
import delay
import pdb

from pylab import *

BLOCK_SIZE = 4096 * 4.


def modCos(position, sampleRate):

    freq = 1
    sampleRate = sampleRate * 1000.
    timePos = position / sampleRate

    output = cos(2 * pi * freq * timePos) * 0.5

    return output


def modSin(position, sampleRate):

    freq = 1.5
    sampleRate = sampleRate * 1000.
    timePos = position / sampleRate

    output = sin(2 * pi * freq * timePos) * 0.6

    return output


def slowCos(position, sampleRate):

    freq = 0.7
    sampleRate = sampleRate * 1000.
    timePos = position / sampleRate

    output = sin(2 * pi * freq * timePos) * 2

    return output


def rpole(audioBuffer, coeff):
    """
    A single pole filter that processes the entire audioBuffer
    """
    filterCoeff = 1. - coeff
    y_prev = 0.0

    for index, sample in enumerate(audioBuffer):
        audioBuffer[index] = (audioBuffer[index] * coeff) + filterCoeff * y_prev
        y_prev = audioBuffer[index]


def delayTest(audioBuffer, sampleRate):
    """
    Testing the delay class.
    """
    delayTime = 4.771
    varDel = delay.VariableAllpass(100, delayTime, sampleRate, 0.5)
    varDel.registerControl(modulate)

    numBlocks = int(ceil((len(audioBuffer) + (sampleRate / 1000) * delayTime) / BLOCK_SIZE))

    for block in range(numBlocks):
        pos = block * BLOCK_SIZE
        endPos = pos + BLOCK_SIZE
        if endPos > len(audioBuffer):
            endPos = len(audioBuffer)

        x = audioBuffer[pos:endPos]

        varDel.process(x, pos)


def diffusionTest(audioBuffer, sampleRate, inputDamping, inputDecay, inputPreDelay, inputBandWidth, tankOffset, freeze=False):
    """
    Starting to test reverb
    """
    scale = 0.3
    preDelayTime = inputPreDelay
    bandWidth = inputBandWidth

    # Diffusor One values
    diffusorOneTime = 4.771
    #diffusorOneGain = 0.5
    diffusorOneGain = 0.75
    # Diffusor Two values
    diffusorTwoTime = 3.595
    diffusorTwoGain = diffusorOneGain
    # Diffusor Three values
    diffusorThreeTime = 12.73
    #diffusorThreeGain = 0.6
    diffusorThreeGain = 0.625
    # Diffusor Four values
    diffusorFourTime = 9.307
    diffusorFourGain = diffusorThreeGain

    numBlocks = int(ceil(len(audioBuffer) / BLOCK_SIZE))
    finalOut = zeros((2, numBlocks * BLOCK_SIZE))

    # create delay objects and whatnot
    preDelay        = delay.VariableAllpass(100, preDelayTime, sampleRate)
    diffusorOne     = delay.VariableAllpass(100, diffusorOneTime, sampleRate, diffusorOneGain)
    diffusorTwo     = delay.VariableAllpass(100, diffusorTwoTime, sampleRate, diffusorTwoGain)
    diffusorThree   = delay.VariableAllpass(100, diffusorThreeTime, sampleRate, diffusorThreeGain)
    diffusorFour    = delay.VariableAllpass(100, diffusorFourTime, sampleRate, diffusorFourGain)

    diffusorOne.registerControl(modCos)
    diffusorTwo.registerControl(modSin)
    diffusorThree.registerControl(modSin)
    diffusorFour.registerControl(modCos)

    x = zeros(BLOCK_SIZE)

    for block in range(numBlocks):
        pos = block * BLOCK_SIZE
        endPos = pos + BLOCK_SIZE

        if endPos > len(audioBuffer):
            endPos = len(audioBuffer)

        # Scale input audio
        if pos > len(audioBuffer):
            x = array([])
        else:
            x = audioBuffer[pos:endPos]

        if len(x) < BLOCK_SIZE:
            x = append(x, zeros(BLOCK_SIZE - len(x)))

        # Needed so multiply operation doesn't create copy
        for i in range(len(x)):
            x[i] = x[i] * scale

        # Now we process everything according to block diagram
        # Predelay
        preDelay.process(x, pos)

        # # Bandwidth
        rpole(x, bandWidth)

        # Diffusion
        diffusorOne.process(x, pos)
        diffusorTwo.process(x, pos)
        diffusorThree.process(x, pos)
        diffusorFour.process(x, pos)


def reverbTest(audioBuffer, sampleRate, inputDamping, inputDecay, inputPreDelay, inputBandWidth, tankOffset, freeze=False):
    """
    Starting to test reverb
    """
    scale = 0.3
    damping = inputDamping
    decay = inputDecay
    preDelayTime = inputPreDelay
    bandWidth = inputBandWidth

    gain = 0.95
    # Diffusor One values
    diffusorOneTime = 4.771
    #diffusorOneGain = 0.5
    diffusorOneGain = 0.75
    # Diffusor Two values
    diffusorTwoTime = 3.595
    diffusorTwoGain = diffusorOneGain
    # Diffusor Three values
    diffusorThreeTime = 12.73
    #diffusorThreeGain = 0.6
    diffusorThreeGain = 0.625
    # Diffusor Four values
    diffusorFourTime = 9.307
    diffusorFourGain = diffusorThreeGain

    numBlocks = int(ceil(len(audioBuffer) / BLOCK_SIZE))
    finalOut = zeros((2, numBlocks * BLOCK_SIZE))

    # create delay objects and whatnot
    preDelay        = delay.VariableAllpass(100, preDelayTime, sampleRate)
    diffusorOne     = delay.VariableAllpass(100, diffusorOneTime, sampleRate, diffusorOneGain)
    diffusorTwo     = delay.VariableAllpass(100, diffusorTwoTime, sampleRate, diffusorTwoGain)
    diffusorThree   = delay.VariableAllpass(100, diffusorThreeTime, sampleRate, diffusorThreeGain)
    diffusorFour    = delay.VariableAllpass(100, diffusorFourTime, sampleRate, diffusorFourGain)

    diffusorOne.registerControl(modCos)
    diffusorTwo.registerControl(modSin)
    diffusorThree.registerControl(modSin)
    diffusorFour.registerControl(modCos)

    # create variable delaylines
    # Right 1
    decayDiffusorRightOneTime = 30.51
    #decayDiffusorRightOneGain = -0.65
    decayDiffusorRightOneGain = -0.6

    decayDiffusorRightOne = delay.VariableAllpass(100, decayDiffusorRightOneTime, sampleRate, decayDiffusorRightOneGain)
    decayDiffusorRightOne.registerControl(modCos)
    # Left 1
    decayDiffusorLeftOneTime = 22.58
    decayDiffusorLeftOneGain = decayDiffusorRightOneGain

    decayDiffusorLeftOne = delay.VariableAllpass(100, decayDiffusorLeftOneTime, sampleRate, decayDiffusorLeftOneGain)
    decayDiffusorLeftOne.registerControl(modSin)

    # Right 2
    decayDiffusorRightTwoTime = 89.24
    decayDiffusorRightTwoGain = 0.5

    decayDiffusorRightTwo = delay.VariableAllpass(100, decayDiffusorRightTwoTime, sampleRate, decayDiffusorRightTwoGain)
    decayDiffusorRightTwo.registerControl(slowCos)

    # Left 2
    decayDiffusorLeftTwoTime = 60.48
    decayDiffusorLeftTwoGain = 0.5

    decayDiffusorLeftTwo = delay.VariableAllpass(100, decayDiffusorLeftTwoTime, sampleRate, decayDiffusorLeftTwoGain)
    decayDiffusorLeftTwo.registerControl(slowCos)

    # first tank delays
    delayRightTankOne = delay.VariableAllpass(200, 141.69, sampleRate)
    delayLeftTankOne  = delay.VariableAllpass(200, 149.62, sampleRate)

    # delayRightTankOne.registerControl(slowCos)
    # delayLeftTankOne.registerControl(slowCos)

    # final delays that crisscross
    tankDelayRight = delay.VariableAllpass(400, 106.28 + tankOffset, sampleRate)                    # right side goes into here and crisscrosses over to left (output goes to left)
    tankDelayLeft  = delay.VariableAllpass(400, 125 + tankOffset, sampleRate)

    # tankDelayLeft.registerControl(slowCos)
    # tankDelayRight.registerControl(slowCos)

    leftSplit = zeros(BLOCK_SIZE)
    rightSplit = zeros(BLOCK_SIZE)
    x = zeros(BLOCK_SIZE)

    # Tap structures
    # time in ms, buffer, position
    # DelayA
    delA = [(8.9, zeros(BLOCK_SIZE), 0.0), (70.8, zeros(BLOCK_SIZE), 0.0), (99.8, zeros(BLOCK_SIZE), 0.0)]
    delB = [(11.2, zeros(BLOCK_SIZE), 0.0), (64.2, zeros(BLOCK_SIZE), 0.0)]
    delC = [(4.1, zeros(BLOCK_SIZE), 0.0), (67, zeros(BLOCK_SIZE), 0.0)]
    delD = [(11.8, zeros(BLOCK_SIZE), 0.0), (66.8, zeros(BLOCK_SIZE), 0.0), (121.7, zeros(BLOCK_SIZE), 0.0)]
    delE = [(6.3, zeros(BLOCK_SIZE), 0.0), (41.2, zeros(BLOCK_SIZE), 0.0)]
    delF = [(35.8, zeros(BLOCK_SIZE), 0.0), (89.7, zeros(BLOCK_SIZE), 0.0)]

    blockRange = range(100, numBlocks - 100)

    for block in range(numBlocks):
        pos = block * BLOCK_SIZE
        endPos = pos + BLOCK_SIZE

        if endPos > len(audioBuffer):
            endPos = len(audioBuffer)

        # Scale input audio
        if pos > len(audioBuffer):
            x = array([])
        else:
            x = audioBuffer[pos:endPos]

        if len(x) < BLOCK_SIZE:
            x = append(x, zeros(BLOCK_SIZE - len(x)))

        # Needed so multiply operation doesn't create copy
        for i in range(len(x)):
            x[i] = x[i] * scale

        # Now we process everything according to block diagram
        # Predelay
        preDelay.process(x, pos)

        # # Bandwidth
        rpole(x, bandWidth)

        # Diffusion
        diffusorOne.process(x, pos)
        diffusorTwo.process(x, pos)
        diffusorThree.process(x, pos)
        diffusorFour.process(x, pos)

        # Tank
        if freeze == True and block in blockRange:
            decay = 0.7
            leftSplit = rightSplit
            rightSplit = leftSplit
        else:
            leftSplit = rightSplit * gain + x
            rightSplit = leftSplit * gain + x

        # process each side
        decayDiffusorLeftOne.process(leftSplit, pos, delD)
        decayDiffusorRightOne.process(rightSplit, pos, delA)                  # delA

        delayLeftTankOne.process(leftSplit, pos)
        delayRightTankOne.process(rightSplit, pos)

        # damp each side
        if freeze == True and block in blockRange:
            pass
        else:
            rpole(leftSplit, 1.0 - damping)
            rpole(rightSplit, 1.0 - damping)

        # multiply by decay amount
        leftSplit = leftSplit * decay
        rightSplit = rightSplit * decay

        decayDiffusorLeftTwo.process(leftSplit, pos, delE)
        decayDiffusorRightTwo.process(rightSplit, pos, delB)

        tankDelayLeft.process(leftSplit, pos, delF)
        tankDelayRight.process(rightSplit, pos, delC)

        leftSplit = leftSplit * decay
        rightSplit = rightSplit * decay

        # Left channel
        finalOut[0][pos:pos + BLOCK_SIZE] = (delA[0][1] + delA[2][1] - delB[1][1] + delC[1][1] - delD[1][1] - delE[0][1] - delF[0][1]) * 0.125
        # Right Channel
        finalOut[1][pos:pos + BLOCK_SIZE] = (delD[0][1] + delD[2][1] - delE[1][1] + delF[1][1] - delA[1][1] - delB[0][1] - delC[0][1]) * 0.125

    return finalOut


def validInput(fileInput):
    """
    Checks if the input file is valid.
    """
    base, ext = os.path.splitext(fileInput)
    if ext.lower() != '.wav':
        raise argparse.ArgumentTypeError('File must be a .wav file')

    return fileInput


def dryWet(dry, wet, amount):
    """
    Mixes a signal, assuming the dry signal is mono and wet is stereo and the wet is longer, super specific
    """
    length = shape(wet)[1]
    output = zeros((2, length))

    dryGain = 1. - amount
    wetGain = amount

    for i in range(length):
        if i < len(dry):
            output[0][i] = dry[i] * dryGain + wet[0][i] * wetGain
            output[1][i] = dry[i] * dryGain + wet[1][i] * wetGain
        else:
            output[0][i] = wet[0][i]
            output[1][i] = wet[1][i]

    return output


def main():
    """
    Main function for processing the specified soundfile through this reverb.
    """

    parser = argparse.ArgumentParser(description='Artificial Reverb')
    parser.add_argument('soundfile', help='audio file to process', type=validInput)        # the soundfile is the first agument, with parameter values to follow
    parser.add_argument('outfile', help='path to output file', type=validInput)
    parser.add_argument('-w', '--wetdry', default=0.2, type=float, help='amount of wet signal in the mix')
    parser.add_argument('-da', '--damping', default=0.25, type=float, help='amount of high frequency damping')
    parser.add_argument('-de', '--decay', default=0.4, type=float, help='amount of attentuation applied to signal to make it decay')
    parser.add_argument('-pd', '--predelay', default=30, type=float, help='amount of time before starting reverb')
    parser.add_argument('-b', '--bandwidth', default=0.6, type=float, help='amount of high frequency attentuation on input')
    parser.add_argument('-t', '--tankoffset', default=0, type=float, help='amount of time (ms) to increase the last tank delay time')

    # Parse the commandline arguments
    args = parser.parse_args()

    # Get the entire path and assign soundfile
    soundfilePath = os.path.join(os.getcwd(), args.soundfile)
    
    # From here on, x refers to the input signal
    x, sampleRate, wavType = wavread(soundfilePath)
    dry = x.copy()

    y = reverbTest(x, sampleRate, args.damping, args.decay, args.predelay, args.bandwidth, args.tankoffset)

    # Apply wet/dry mix
    output = dryWet(dry, y, args.wetdry)

    # Finally write the output file
    wavwrite(transpose(output), args.outfile, sampleRate)


if __name__ == '__main__':
    main()
