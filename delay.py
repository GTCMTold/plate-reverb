#
#
#
#       delay.py
#
#       description:
#       this module provides basic classes for delayline based dsp structures
#
#
#

from pylab import *
import pdb


class VariableAllpass:
    """
    Variable length allpass filter

    This class allows users to create a normal variable length delay line or specify a gain to turn
    it into an allpass filter.

    Additionally, it's possible to register a function callback for modulating the delay line length.
    This allows the possibility of user defined modulation functions at the control rate.
    """
    def __init__(self, maxDelaySize, delayTime, sampleRate, gain=0.0):
        """
        delayTime   - time of delay (ms)
        size        - size of delay line (ms)
        sampleRate  - sample rate of system
        """

        self.sr = sampleRate / 1000.
        self.size = maxDelaySize * self.sr
        self.delayTime = delayTime * self.sr        # The actual delay time, which is less than the max
        self.gain = gain

        # Create internal buffer for delay samples
        self.delay = zeros(self.size)

        # This is a reference to a function
        self.control = None

        # The read position is a float because a specific time in ms might be between samples
        self.readPosition = 0.
        # The write position is always an integer
        self.writePosition = 0
        # The previous output of the allpass interpolation
        self.y_prev = 0.

    def process(self, audioBuffer, position, tapBuffers=None):
        """
        Processes the samples in place (i.e. doesn't return an array, just modifies input array)
        """
        newDelayTime = self.delayTime

        for index, sample in enumerate(audioBuffer):

            newDelayTime = self.delayTime

            if self.control != None:
                controlValue = self.control(position + index, self.sr)       # this is initially in ms and needs to be converted to samples
                newDelayTime = self.delayTime + (controlValue * self.sr)

            if newDelayTime > self.size:
                newDelayTime = self.size

            ######################
            self.readPosition = self.writePosition - newDelayTime

            if self.readPosition >= 0:
                if self.readPosition >= self.size:
                    self.readPosition = self.readPosition - self.size
            else:
                self.readPosition += self.size

            readPositionInt = int(self.readPosition)
            fraction = self.readPosition - readPositionInt

            if readPositionInt <= (self.size - 2):
                nextValue = self.delay[readPositionInt + 1]
            else:
                nextValue = self.delay[0]

            # use allpass interpolation - because it said so in the paper
            out = nextValue + (1 - fraction) * self.delay[readPositionInt] - (1 - fraction) * self.y_prev
            self.y_prev = out

            ####################
            self.delay[self.writePosition] = sample + out * self.gain
            audioBuffer[index] = out - self.gain * sample

            if tapBuffers is not None:
                for tap in tapBuffers:
                    self.getTap(tap, index)

            # advance write postion
            self.writePosition += 1
            if self.writePosition > self.size - 1:
                self.writePosition = 0

    def getTap(self, tapInfo, position):
        """
        Returns samples that are currently in the buffer according to the position
        """
        delayTime = tapInfo[0]
        tapBuffer = tapInfo[1]
        y_prev = tapInfo[2]

        readPosition = self.writePosition - delayTime
        if readPosition >= 0:
            if readPosition >= self.size:
                readPosition = readPosition - self.size
        else:
            readPosition += self.size

        readPositionInt = int(readPosition)
        fraction = readPosition - readPositionInt

        if readPositionInt <= (self.size - 2):
            nextValue = self.delay[readPositionInt + 1]
        else:
            nextValue = self.delay[0]

        out = nextValue + (1 - fraction) * self.delay[readPositionInt] - (1 - fraction) * y_prev
        y_prev = out

        tapBuffer[position] = out

    def registerControl(self, controlFunction):
        """
        This method allows you to register a callback
        function that's called before each processing block.

        This should allow for user defined functions that are external
        to this specific class (weird lfos, break point files, etc.)
        """

        self.control = controlFunction
