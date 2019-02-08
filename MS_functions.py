""" Functions specific to MS data
(e.g. importing and data processing functions)

Florian Huber
Netherlands eScience Center, 2019

TODO: add liscence
"""

import os
import helper_functions as functions
import fnmatch
import copy
import numpy as np
from scipy.optimize import curve_fit


## ----------------------------------------------------------------------------
## ---------------------------- Spectrum class --------------------------------
## ----------------------------------------------------------------------------

class Spectrum(object):
    """ Spectrum class to store key information
    
    Functions include:
        - Import data from mass spec files (protoype so far, works with only few formats)
        - Calculate losses from peaks.
        - Process / filter peaks
        
    Args:
    -------
    min_frag: float
        Lower limit of m/z to take into account (Default = 0.0). 
    max_frag: float
        Upper limit of m/z to take into account (Default = 1000.0).
    min_loss: float
        Lower limit of losses to take into account (Default = 10.0).
    max_loss: float
        Upper limit of losses to take into account (Default = 200.0).
    min_intensity_perc: float
        Filter out peaks with intensities lower than the min_intensity_perc percentage
        of the highest peak intensity (Default = 0.0, essentially meaning: OFF).
    exp_intensity_filter: float
        Filter out peaks by applying an exponential fit to the intensity histogram.
        Intensity threshold will be set at where the exponential function will have dropped 
        to exp_intensity_filter (Default = 0.01).
    min_peaks: int
        Minimum number of peaks to keep, unless less are present from the start (Default = 10).
    merge_energies: bool
        Merge close peaks or not (False | True, Default is True).
    merge_ppm: int
        Merge peaks if their m/z is <= 1e6*merge_ppm (Default = 10).
    replace: 'max' or None
        If peaks are merged, either the heighest intensity of both is taken ('max'), 
        or their intensitites are added (None). 
    """
    def __init__(self, min_frag = 0.0, max_frag = 1000.0,
                 min_loss = 10.0, max_loss = 200.0,
                 min_intensity_perc = 0.0,
                 exp_intensity_filter = 0.01,
                 min_peaks = 10,
                 merge_energies = True,
                 merge_ppm = 10,
                 replace = 'max'):

        self.id = []
        self.filename = []
        self.peaks = []
        self.precursor_mz = []
        self.parent_mz = []
        self.metadata = {}
        self.family = None
        self.annotations = []
        self.smiles = []
        
        self.min_frag = min_frag 
        self.max_frag = max_frag
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.min_intensity_perc = min_intensity_perc
        self.exp_intensity_filter = exp_intensity_filter
        self.min_peaks = min_peaks
        self.merge_energies = merge_energies
        self.merge_ppm = merge_ppm
        self.replace = replace
    
    def read_spectrum(self, path, file, id):
        """ Read .ms file and extract most relevant information
        """
    
        with open(os.path.join(path, file),'r') as f:            
            temp_mass = []
            temp_intensity = []
            doc_name = file.split('/')[-1]
            self.filename = doc_name
            self.id = id
            for line in f:
                rline = line.rstrip()
                if len(rline) > 0:
                    if rline.startswith('>') or rline.startswith('#'):
                        keyval = rline[1:].split(' ')[0]
                        valval = rline[len(keyval)+2:]
                        if not keyval == 'ms2peaks':
                            self.metadata[keyval] = valval
                        if keyval == 'compound':
                            self.annotation = valval
                        if keyval == 'parentmass':
                            self.parent_mz = float(valval)
                        if keyval == 'intensity':
                            self.intensity = float(valval)
                        if keyval == 'smiles':
                            self.smiles = valval
                    else:
                        # If it gets here, its a fragment peak (MS2 level peak)
                        sr = rline.split(' ')
                        mass = float(sr[0])
                        intensity = float(sr[1])                
                        if self.merge_energies and len(temp_mass)>0:
                            # Compare to other peaks
                            errs = 1e6*np.abs(mass-np.array(temp_mass))/mass
                            if errs.min() < self.merge_ppm:
                                # Don't add, but merge the intensity
                                min_pos = errs.argmin()
                                if self.replace == 'max':
                                    temp_intensity[min_pos] = max(intensity,temp_intensity[min_pos])
                                else:
                                    temp_intensity[min_pos] += intensity
                            else:
                                temp_mass.append(mass)
                                temp_intensity.append(intensity)
                        else:
                            temp_mass.append(mass)
                            temp_intensity.append(intensity)
        
        peaks = list(zip(temp_mass, temp_intensity))
#        peaks = self.process_peaks(peaks)
        peaks = process_peaks(peaks, self.min_frag, self.max_frag,
                              self.min_intensity_perc, self.exp_intensity_filter,
                              self.min_peaks)
        
        self.peaks = peaks


    def get_losses(self):
        """ Use spectrum class and extract peaks and losses
        Losses are here the differences between the spectrum precursor mz and the MS2 level peaks.
        
        Remove losses outside window min_loss <-> max_loss.
        """ 
        
        MS1_peak = self.parent_mz
        losses = np.array(self.peaks.copy())
        losses[:,0] = MS1_peak - losses[:,0]
        keep_idx = np.where((losses[:,0] > self.min_loss) & (losses[:,0] < self.max_loss))[0]
        
        # TODO: now array is tranfered back to list (to be able to store as json later). Seems weird.
        losses_list = [(x[0], x[1]) for x in losses[keep_idx,:]]
        self.losses = losses_list


def process_peaks(peaks, min_frag, max_frag, 
                  min_intensity_perc,
                  exp_intensity_filter,
                  min_peaks):
    """ Process peaks
    
    Remove peaks outside window min_frag <-> max_frag.
    Remove peaks with intensities < min_intensity_perc/100*max(intensities)
    
    Uses exponential fit to intensity histogram. Threshold for maximum allowed peak
    intensity will be set where the exponential fit reaches exp_intensity_filter.
    
    Args:
    -------
    min_frag: float
        Lower limit of m/z to take into account (Default = 0.0). 
    max_frag: float
        Upper limit of m/z to take into account (Default = 1000.0).
    min_intensity_perc: float
        Filter out peaks with intensities lower than the min_intensity_perc percentage
        of the highest peak intensity (Default = 0.0, essentially meaning: OFF).
    exp_intensity_filter: float
        Filter out peaks by applying an exponential fit to the intensity histogram.
        Intensity threshold will be set at where the exponential function will have dropped 
        to exp_intensity_filter (Default = 0.01).
    min_peaks: int
        Minimum number of peaks to keep, unless less are present from the start (Default = 10).
   
    """
    def exponential_func(x, a, b):
        return a*np.exp(-b*x)
   
    if isinstance(peaks, list):
        peaks = np.array(peaks)
        if peaks.shape[1] != 2:
            print("Peaks were given in unexpected format...")
    
    intensity_thres = np.max(peaks[:,1]) * min_intensity_perc/100
    keep_idx = np.where((peaks[:,0] > min_frag) & (peaks[:,0] < max_frag) & (peaks[:,1] > intensity_thres))[0]
    if (len(keep_idx) < min_peaks):
        peaks = peaks[np.lexsort((peaks[:,0], peaks[:,1])),:][-min(min_peaks, len(peaks)):]
    else:
        peaks = peaks[keep_idx,:]

    if (exp_intensity_filter is not None) and len(peaks) > 2*min_peaks:
        # Fit exponential to peak intensity distribution 
        num_bins = 100  # number of bins for histogram

        # Ignore highest peak for further analysis 
        peaks2 = peaks.copy()
        peaks2[np.where(peaks2[:,1] == np.max(peaks2[:,1])),:] = 0
        hist, bins = np.histogram(peaks2[:,1], bins=num_bins)
        start = np.where(hist == np.max(hist))[0][0]  # take maximum intensity bin as starting point
        last = int(num_bins/2)
        x = bins[start:last]
        y = hist[start:last]
        try:
            popt, pcov = curve_fit(exponential_func, x , y, p0=(peaks.shape[0], 1e-4)) 
            threshold = -np.log(exp_intensity_filter)/popt[1]
        except RuntimeError:
            print("RuntimeError for ", len(peaks), " peaks. Use mean intensity as threshold.")
            threshold = np.mean(peaks2[:,1])
        except TypeError:
            print("Unclear TypeError for ", len(peaks), " peaks. Use mean intensity as threshold.")
            print(x, "and y: ", y)
            threshold = np.mean(peaks2[:,1])

        keep_idx = np.where(peaks[:,1] > threshold)[0]
        if len(keep_idx) < min_peaks:
            peaks = peaks[np.lexsort((peaks[:,0], peaks[:,1])),:][-min_peaks:]
        else:
            peaks = peaks[keep_idx, :]
        return [(x[0], x[1]) for x in peaks] # TODO: now array is transfered back to list (to be able to store as json later). Seems weird.
    else:
        return [(x[0], x[1]) for x in peaks]


## ----------------------------------------------------------------------------
## -------------------------- Functions to load MS data------------------------
## ----------------------------------------------------------------------------

def load_MS_data(path_data, path_json,
                 filefilter="*.*", 
                 results_file = None,
                 num_decimals = 3,
                 min_frag = 0.0, max_frag = 1000.0,
                 min_loss = 10.0, max_loss = 200.0,
                 min_intensity_perc = 0.0,
                 exp_intensity_filter = 0.01,
                 min_peaks = 10,
                 merge_energies = True,
                 merge_ppm = 10,
                 replace = 'max'):        
    """ Collect spectra from set of files
    Partly taken from ms2ldaviz.
    Prototype. Needs to be replaces by more versatile parser, accepting more MS data formats.
    """

    spectra = []
    spectra_dict = {}
    MS_documents = []
    MS_documents_intensity = []

    dirs = os.listdir(path_data)
    spectra_files = fnmatch.filter(dirs, filefilter)
        
    if results_file is not None:
        try: 
            spectra_dict = functions.json_to_dict(path_json + results_file)
            print("Spectra json file found and loaded.")
            collect_new_data = False
            
            with open(path_json + results_file[:-4] + "txt", "r") as f:
                for line in f:
                    line = line.replace('"', '').replace("'", "").replace("[", "").replace("]", "").replace("\n", "")
                    MS_documents.append(line.split(", "))
                    
            with open(path_json + results_file[:-5] + "_intensity.txt", "r") as f:
                for line in f:
                    line = line.replace("[", "").replace("]", "")
                    MS_documents_intensity.append([int(x) for x in line.split(", ")])
                
        except FileNotFoundError: 
            print("Could not find file ", path_json,  results_file) 
            print("New data from ", path_data, " will be imported.")
            collect_new_data = True

    # Read data from files if no pre-stored data is found:
    if spectra_dict == {} or results_file is None:

        # Run over all spectrum files:
        for i, filename in enumerate(spectra_files):
            
            # Show progress
            if (i+1) % 10 == 0 or i == len(spectra_files)-1:  
                print('\r', ' Load spectrum ', i+1, ' of ', len(spectra_files), ' spectra.', end="")
            
            spectrum = Spectrum(min_frag = min_frag, 
                                max_frag = max_frag,
                                min_loss = min_loss, 
                                max_loss = max_loss,
                                min_intensity_perc = min_intensity_perc,
                                exp_intensity_filter = exp_intensity_filter,
                                min_peaks = min_peaks,
                                merge_energies = merge_energies,
                                merge_ppm = merge_ppm,
                                replace = replace)
            
            # Load spectrum data from file:
            spectrum.read_spectrum(path_data, filename, i)
            
            # Calculate losses:
            spectrum.get_losses()
            
            # Collect in form of list of spectrum objects, and as dictionary
            spectra.append(spectrum)
            spectra_dict[filename] = spectrum.__dict__
        
        MS_documents, MS_documents_intensity = create_MS_documents(spectra, num_decimals)

        # Save collected data
        if collect_new_data == True:
            
            functions.dict_to_json(spectra_dict, path_json + results_file)     
            # Store documents
            with open(path_json + results_file[:-4] + "txt", "w") as f:
                for s in MS_documents:
                    f.write(str(s) +"\n")
                    
            with open(path_json + results_file[:-5] + "_intensity.txt", "w") as f:
                for s in MS_documents_intensity:
                    f.write(str(s) +"\n")

    return spectra, spectra_dict, MS_documents, MS_documents_intensity



def load_MGF_data(path_json,
                  mgf_file, 
                 results_file = None,
                 num_decimals = 3,
                 min_frag = 0.0, max_frag = 1000.0,
                 min_loss = 10.0, max_loss = 200.0,
                 exp_intensity_filter = 0.01,
                 min_peaks = 10,
                 peaks_per_mz = 20/200):        
    """ Collect spectra from MGF file
    Partly taken from ms2ldaviz.
    Prototype. Needs to be replaces by more versatile parser, accepting more MS data formats.
    """
    from parsers import LoadMGF
    from metabolomics import load_spectra
    
    spectra = []
    spectra_dict = {}
    MS_documents = []
    MS_documents_intensity = []
    collect_new_data = True
        
    if results_file is not None:
        try: 
            spectra_dict = functions.json_to_dict(path_json + results_file)
            print("Spectra json file found and loaded.")
            collect_new_data = False
            
            with open(path_json + results_file[:-4] + "txt", "r") as f:
                for line in f:
                    line = line.replace('"', '').replace("'", "").replace("[", "").replace("]", "").replace("\n", "")
                    MS_documents.append(line.split(", "))
                    
            with open(path_json + results_file[:-5] + "_intensity.txt", "r") as f:
                for line in f:
                    line = line.replace("[", "").replace("]", "")
                    MS_documents_intensity.append([int(x) for x in line.split(", ")])
                
        except FileNotFoundError: 
            print("Could not find file ", path_json,  results_file) 
            print("Data will be impted from ", results_file)

    # Read data from files if no pre-stored data is found:
    if spectra_dict == {} or results_file is None:

        # Load mgf file
        spectra = load_spectra(mgf_file)
        # Load metadata
        ms1, ms2, metadata = LoadMGF(name_field='scans').load_spectra([mgf_file])

        for spec in spectra:
            id = spec.spectrum_id
            spec.metadata = metadata[id]

        min_peak_absolute = min_peaks
        
        # Scale the min_peak filter
        def min_peak_scaling(x, A, B):
            return int(A + B * x)
        
        for spec in spectra:
            peaks = spec.peaks.copy()
            
            # Scale the min_peak filter
            min_peaks = min_peak_scaling(spec.precursor_mz, min_peak_absolute, peaks_per_mz)
            
            if len(peaks) > min_peaks:
                peaks = process_peaks(peaks, 
                                       min_frag = min_frag, 
                                       max_frag = max_frag,
                                       min_intensity_perc = 0.0,
                                       exp_intensity_filter = exp_intensity_filter,
                                       min_peaks = min_peaks)
            spec.peaks = peaks
            
            
        # filter out spectra with few peaks
        min_peaks_absolute = 10
        num_peaks_initial = len(spectra)
        spectra = [copy.deepcopy(x) for x in spectra if len(x.peaks)>=min_peaks_absolute]
        print("Take ", len(spectra), "peaks out of ", num_peaks_initial, ".")


        # Transfer object to dictionary
        spectra_dict = {}
        for spec in spectra:
            spectra_dict[spec.spectrum_id] = spec.__dict__
            spectra_dict[spec.spectrum_id]["smiles"] = spec.metadata["smiles"]
                    
        #            # Calculate losses
        #            losses = np.array(peaks.copy())
        #            losses[:,0] = spec.precursor_mz - losses[:,0]
        #            
        #            keep_idx = np.where((losses[:,0] > min_loss) & (losses[:,0] < max_loss))[0]                        
        #            losses = losses[keep_idx,:]
        #            
        #            spec.losses = [(x[0], x[1]) for x in losses]
        #            

        # Create documents from peaks (and losses)
        MS_documents, MS_documents_intensity = create_MS_documents(spectra, num_decimals,
                                                                   min_loss, max_loss)

        # Save collected data
        if collect_new_data == True:
            
            functions.dict_to_json(spectra_dict, path_json + results_file)     
            # Store documents
            with open(path_json + results_file[:-4] + "txt", "w") as f:
                for s in MS_documents:
                    f.write(str(s) +"\n")
                    
            with open(path_json + results_file[:-5] + "_intensity.txt", "w") as f:
                for s in MS_documents_intensity:
                    f.write(str(s) +"\n")

    return spectra, spectra_dict, MS_documents, MS_documents_intensity



def create_MS_documents(spectra, num_decimals,
                        min_loss = 10.0, max_loss = 200.0):
    """ Create documents from peaks and losses.
    
    Every peak and every loss will be transformed into a WORD.
    Words then look like this: "peak_100.038" or "loss_59.240" 
    
    Args:
    --------
    spectra: list
        List of all spectrum class elements = all spectra to be in corpus
    num_decimals: int
        Number of decimals to take into account
    min_loss: float
        Lower limit of losses to take into account (Default = 10.0).
    max_loss: float
        Upper limit of losses to take into account (Default = 200.0).
    """
    MS_documents = []
    MS_documents_intensity = []
    
    for i, spectrum in enumerate(spectra):
        doc = []
        doc_intensity = []
        losses = np.array(spectrum.losses)
        keep_idx = np.where((losses[:,0] > min_loss) & (losses[:,0] < max_loss))[0]                        
        losses = losses[keep_idx,:]
        peaks = np.array(spectrum.peaks)
        
        # Sort peaks and losses by m/z 
        peaks = peaks[np.lexsort((peaks[:,1], peaks[:,0])),:]
        losses = losses[np.lexsort((losses[:,1], losses[:,0])),:]

        if (i+1) % 100 == 0 or i == len(spectra)-1:  # show progress
                print('\r', ' Created documents for ', i+1, ' of ', len(spectra), ' spectra.', end="")
                
        for i in range(len(peaks)):
            doc.append("peak_" + "{:.{}f}".format(peaks[i,0], num_decimals))
            doc_intensity.append(int(peaks[i,1]))
            
        for i in range(len(losses)):
            doc.append("loss_"  + "{:.{}f}".format(losses[i,0], num_decimals))
            doc_intensity.append(int(losses[i,1]))

        MS_documents.append(doc)
        MS_documents_intensity.append(doc_intensity)
         
    return MS_documents, MS_documents_intensity




## ----------------------------------------------------------------------------
## ---------------------------- Plotting functions ----------------------------
## ----------------------------------------------------------------------------

from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib import pyplot as plt

def plot_smiles(query_id, spectra, MS_measure, num_candidates = 10,
                   sharex=True, labels=False, dist_method = "centroid",
                   plot_type = "single"):
    """ Plot molecules for closest candidates
    
    """

    # Select chosen distance methods
    if dist_method == "centroid":
        candidates_idx = MS_measure.Cdistances_ctr_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.Cdistances_ctr[query_id, :num_candidates]
    elif dist_method == "pca":
        candidates_idx = MS_measure.Cdistances_pca_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.Cdistances_pca[query_id, :num_candidates]
    elif dist_method == "autoencoder":
        candidates_idx = MS_measure.Cdistances_ae_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.Cdistances_ae[query_id, :num_candidates]
    elif dist_method == "lda":
        candidates_idx = MS_measure.Cdistances_lda_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.Cdistances_lda[query_id, :num_candidates]
    elif dist_method == "lsi":
        candidates_idx = MS_measure.Cdistances_lsi_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.Cdistances_lsi[query_id, :num_candidates]
    elif dist_method == "doc2vec":
        candidates_idx = MS_measure.Cdistances_d2v_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.Cdistances_d2v[query_id, :num_candidates]
    else:
        print("Chosen distance measuring method not found.")

    size = (200, 200)  # Smaller figures than the default

    if isinstance(spectra, dict):
        # If spectra is given as a dictionary
        keys = []
        for key, value in spectra.items():
            keys.append(key)  
            
        smiles = []  
        molecules = []
        
        for i, candidate_id in enumerate(candidates_idx):
            key = keys[candidate_id]
            smiles.append(spectra[key]["smiles"])
            mol = Chem.MolFromSmiles(smiles[i])
            if mol != None:
                mol.SetProp('_Name', smiles[i])
                if plot_type == 'single':
                    Draw.MolToMPL(mol, size=size)
        
        if plot_type != "single":    # this will only work if there's no conflict with rdkit and pillow...       
            Chem.Draw.MolsToGridImage(molecules,legends=[mol.GetProp('_Name') for mol in molecules])
            
    elif isinstance(spectra, list):
        # Assume that it is then a list of Spectrum objects
        
        smiles = []  
        for i, candidate_id in enumerate(candidates_idx):
            smiles.append(spectra[candidate_id].metadata["smiles"])
            mol = Chem.MolFromSmiles(smiles[i])
#            mol.SetProp('_Name', smiles[i])
            if plot_type == 'single':
                Draw.MolToMPL(mol, size=size)
        
        if plot_type != "single":    # this will only work if there's no conflict with rdkit and pillow...       
            Chem.Draw.MolsToGridImage(molecules,legends=[mol.GetProp('_Name') for mol in molecules])


