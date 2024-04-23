import xml.etree.ElementTree as ET
import numpy as np

def read_bec(xmlfile):
   root = ET.parse(xmlfile).getroot()
   array_vectors = root.findall("./calculation/*[@name='born_charges']/set/v")
   bec = np.array([np.fromstring(element.text, sep=' ') for element in array_vectors])
   bec.reshape(-1,3,3)
   return bec
