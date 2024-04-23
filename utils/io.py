import xml.etree.ElementTree as ET
import numpy as np

def read_vasp_bec(xmlfile: str):
   """Given a vasprun.xml file location read and parse bec as np array with 
      indices(k, j, i): k over atoms, j over E-field x,y,z, i over forces along x,y,z
      Unit is |e| [(eV/A) / (V/A)]  
   """
   root = ET.parse(xmlfile).getroot()
   array_vectors = root.findall("./calculation/*[@name='born_charges']/set/v")
   bec = np.array([np.fromstring(element.text, sep=' ') for element in array_vectors])
   bec.reshape(-1,3,3)
   return bec
   
