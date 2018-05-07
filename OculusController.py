#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:00:14 2018

@author: oculus
"""

import re, time
import oculusprimesocket as oc


class OculusController(object):
    def __init__(self):
        #connect to the oculus server
        result = oc.connect()
        return result
    
    def moveLeft(self):
        oc.sendString('move left')
        
    def moveRight(self):
        oc.sendString('move right')
        
    def moveForward(self):
        oc.sendString('move forward')
        
    def moveBackward(self):
        oc.sendString('move backward')
        
    def stop(self):
        oc.sendString('move stop')
        
    def nudgeLeft(self):
        oc.sendString('nudge left')
        
    def nudgeRight(self):
        oc.sendString('nudge right')
    
    def nudgeForward(self):
        oc.sendString('nudge right')
        
    def nudgeBackward(self):
        oc.sendString('nudge right')

    def speak(self, msg):
        oc.sendString('speech '  + msg)
        
