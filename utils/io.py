import numpy as np

def read_vasp_bec(xmlfile: str):
   """Given a vasprun.xml file location read and parse bec as np array with 
      indices(k, j, i): k over atoms, j over E-field x,y,z, i over forces along x,y,z
      Unit is |e| [(eV/A) / (V/A)]  
   """
   import xml.etree.ElementTree as ET
   root = ET.parse(xmlfile).getroot()
   array_vectors = root.findall("./calculation/*[@name='born_charges']/set/v")
   bec = np.array([np.fromstring(element.text, sep=' ') for element in array_vectors])
   bec = bec.reshape(-1,3,3)
   return bec

def ase_db_to_csv(db_file):
   from ase.db import connect
   import panda as pd
   db = connect(db_file)
   all_atoms = [row.toatoms().todict() for row in db.select('gap>0,converged=True')]
   all_bec = [row.data.bed for row in db.select('gap>0,converged=True')]
   df = pd.DataFrame(data={'structure': all_atoms, 'bec': all_bec})
   pd.DataFrame.to_csv('data.csv', df)
