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
from scipy.optimize import linear_sum_assignment
import random
import pandas as pd

from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem

# Add multi core parallelization
from concurrent.futures import ThreadPoolExecutor, as_completed


## --------------------------------------------------------------------------------------------------
## ---------------------------- Spectrum class ------------------------------------------------------
## --------------------------------------------------------------------------------------------------

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
        if exp_intensity_filter == 0:
            self.exp_intensity_filter = None
        else:
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
        self.n_peaks = len(peaks)

    def read_spectrum_mgf(self, spectrum_mgf, id):
        """ Translate to THIS spectrum object given that we have a metabolomics.py spectrum object
        """
        self.id = id
#        self.filename = doc_name
        
        if spectrum_mgf.parent_mz is not None:
            self.parent_mz = spectrum_mgf.parent_mz
        elif 'parentmass' in spectrum_mgf.metadata:
            self.parent_mz = float(spectrum_mgf.metadata['parentmass'])
        elif spectrum_mgf.precursor_mz is not None:
            print("Only found precursor mass, not parent mass.")
            self.parent_mz = spectrum_mgf.precursor_mz
        elif 'precursormass' in spectrum_mgf.metadata:
            print("Only found precursor mass, not parent mass.")
            self.parent_mz = float(spectrum_mgf.metadata['precursormass'])
        else:
            print(id, spectrum_mgf.parent_mz) 
        
        if spectrum_mgf.metadata:
            self.metadata = spectrum_mgf.metadata

        if 'smiles' in spectrum_mgf.metadata:
            self.smiles = spectrum_mgf.metadata['smiles']

        peaks = spectrum_mgf.peaks
        peaks = process_peaks(peaks, self.min_frag, self.max_frag,
                              self.min_intensity_perc, self.exp_intensity_filter,
                              self.min_peaks)
        
        self.peaks = peaks
        self.n_peaks = len(peaks)


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
        

        
def dict_to_spectrum(spectra_dict): 
    """ Create spectrum object from spectra_dict.
    """
    spectra = []
    keys = []
    for key, value in spectra_dict.items():
        keys.append(key) 
            
        spectrum = Spectrum(min_frag = value["min_frag"], 
                            max_frag = value["max_frag"],
                            min_loss = value["min_loss"], 
                            max_loss = value["max_loss"],
                            min_intensity_perc = 0,
                            exp_intensity_filter = value["exp_intensity_filter"],
                            min_peaks = value["min_peaks"])
        
        for key2, value2 in value.items():
            setattr(spectrum, key2, value2)
         
        spectrum.peaks = [(x[0],x[1]) for x in spectrum.peaks]  # convert to tuples
        
        # Collect in form of list of spectrum objects
        spectra.append(spectrum)
        
    return spectra


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
        # if not enough peaks selected, try again without intensity threshold
        keep_idx2 = np.where((peaks[:,0] > min_frag) & (peaks[:,0] < max_frag))[0]
        peaks = peaks[keep_idx2,:]
#        peaks = peaks[np.lexsort((peaks[:,0], peaks[:,1])),:][-min(min_peaks, len(peaks)):]  # former way --> problem: leads to peaks >max_frag
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
        
        # Sort by peak m/z
        peaks = peaks[np.lexsort((peaks[:,1], peaks[:,0])),:]
        return [(x[0], x[1]) for x in peaks] # TODO: now array is transfered back to list (to be able to store as json later). Seems weird.
    else:
        # Sort by peak m/z
        peaks = peaks[np.lexsort((peaks[:,1], peaks[:,0])),:]
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
            spectra = dict_to_spectrum(spectra_dict)
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
                 min_intensity_perc = 0.0,
                 exp_intensity_filter = 0.01,
                 min_peaks = 10,
                 peaks_per_mz = 20/200,
                 peak_loss_words = ['peak_', 'loss_'],
                 sub_spectra = False):        
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
            sub_spectra_metadata = pd.read_csv(path_json + results_file[:-5] + "_metadata.csv")
            print("Spectra json file found and loaded.")
            spectra = dict_to_spectrum(spectra_dict)
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
            print("Data will be imported from ", results_file)

    # Read data from files if no pre-stored data is found:
    if spectra_dict == {} or results_file is None:

        # Load mgf file
        spectra_mgf = load_spectra(mgf_file)
        # Load metadata
        ms1, ms2, metadata = LoadMGF(name_field='scans').load_spectra([mgf_file])

        for spec in spectra_mgf:
            spec.metadata = metadata[spec.spectrum_id]

        # Make conform with spectrum class as defined in MS_functions.py
        #--------------------------------------------------------------------
        
        # Scale the min_peak filter
        def min_peak_scaling(x, A, B):
            return int(A + B * x)
        
        for spec in spectra_mgf:
            # Scale the min_peak filter
            min_peaks_scaled = min_peak_scaling(spec.precursor_mz, min_peaks, peaks_per_mz)
            
            spectrum = Spectrum(min_frag = min_frag, 
                                max_frag = max_frag,
                                min_loss = min_loss, 
                                max_loss = max_loss,
                                min_intensity_perc = min_intensity_perc,
                                exp_intensity_filter = exp_intensity_filter,
                                min_peaks = min_peaks_scaled)
            
            id = spec.spectrum_id
            spectrum.read_spectrum_mgf(spec, id)
            spectrum.get_losses

            # Calculate losses:
            if len(spectrum.peaks) >= min_peaks: 
                spectrum.get_losses()
            
            # Collect in form of list of spectrum objects
            spectra.append(spectrum)

            
        # filter out spectra with few peaks
        min_peaks_absolute =  min_peaks
        num_spectra_initial = len(spectra)
        spectra = [copy.deepcopy(x) for x in spectra if len(x.peaks)>=min_peaks_absolute]
        print("Take ", len(spectra), "spectra out of ", num_spectra_initial, ".")


        # Collect dictionary
        for spec in spectra:
            id = spec.id
            spectra_dict[id] = spec.__dict__

        # Create documents from peaks (and losses)
        if sub_spectra == True:
            MS_documents, MS_documents_intensity, sub_spectra_metadata = create_subspectra_documents(spectra, num_decimals,
                                                                               min_loss, max_loss,
                                                                               main_peak_cutoff = 0.2, min_words = 10)
        else:
            MS_documents, MS_documents_intensity, sub_spectra_metadata = create_MS_documents(spectra, num_decimals, 
                                                                                             peak_loss_words, 
                                                                                             min_loss, max_loss)

        # Save collected data
        if collect_new_data == True:
            sub_spectra_metadata.to_csv(path_json + results_file[:-5] + "_metadata.csv", index=False)
            
            functions.dict_to_json(spectra_dict, path_json + results_file)     
            # Store documents
            with open(path_json + results_file[:-4] + "txt", "w") as f:
                for s in MS_documents:
                    f.write(str(s) +"\n")
                    
            with open(path_json + results_file[:-5] + "_intensity.txt", "w") as f:
                for s in MS_documents_intensity:
                    f.write(str(s) +"\n")

    return spectra, spectra_dict, MS_documents, MS_documents_intensity, sub_spectra_metadata


## --------------------------------------------------------------------------------------------------
## ---------------------- Functions to analyse MS data ----------------------------------------------
## --------------------------------------------------------------------------------------------------


def create_MS_documents(spectra, num_decimals, peak_loss_words = ['peak_', 'loss_'],
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
    spectra_metadata = pd.DataFrame(columns=['doc_ID', 'spectrum_ID', 'sub_ID', 'parent_mz', 'parent_intensity', 'no_peaks_losses'])
    
    for spec_id, spectrum in enumerate(spectra):
        doc = []
        doc_intensity = []
        losses = np.array(spectrum.losses)
        if len(losses) > 0: 
            keep_idx = np.where((losses[:,0] > min_loss) & (losses[:,0] < max_loss))[0]
            losses = losses[keep_idx,:]
        else:
            print("No losses detected for: ", spec_id, spectrum.id)
        peaks = np.array(spectrum.peaks)
        
        # Sort peaks and losses by m/z 
        peaks = peaks[np.lexsort((peaks[:,1], peaks[:,0])),:]
        if len(losses) > 0: 
            losses = losses[np.lexsort((losses[:,1], losses[:,0])),:]

        if (spec_id+1) % 100 == 0 or spec_id == len(spectra)-1:  # show progress
                print('\r', ' Created documents for ', spec_id+1, ' of ', len(spectra), ' spectra.', end="")
                
        for i in range(len(peaks)):
            doc.append(peak_loss_words[0] + "{:.{}f}".format(peaks[i,0], num_decimals))
            doc_intensity.append(int(peaks[i,1]))
            
        for i in range(len(losses)):
            doc.append(peak_loss_words[1]  + "{:.{}f}".format(losses[i,0], num_decimals))
            doc_intensity.append(int(losses[i,1]))

        MS_documents.append(doc)
        MS_documents_intensity.append(doc_intensity)
        spectra_metadata.loc[spec_id] = [spec_id, int(spectrum.id), 0, spectrum.parent_mz, 1, len(doc)]
         
    return MS_documents, MS_documents_intensity, spectra_metadata


def create_subspectra_documents(spectra, num_decimals, 
                                min_loss = 10.0, max_loss = 200.0,
                                main_peak_cutoff = 0.2,
                                min_words = 10):
    """ Create documents from peaks and losses. AND documents for subspectra (from main peaks)
    
    Every peak and every loss will be transformed into a WORD.
    Words then look like this: "peak_100.038" or "loss_59.240" 
    
    Args:
    --------
    spectra: list
        List of all spectrum class elements = all spectra to be in corpus
    num_decimals: int
        Number of decimals to take into account
    main_peak_cutoff: float
    
    min_loss: float
        Lower limit of losses to take into account (Default = 10.0).
    max_loss: float
        Upper limit of losses to take into account (Default = 200.0).
    """
    MS_documents = []
    MS_documents_intensity = []
#    sub_spectra_reg = {}
    sub_spectra_metadata = pd.DataFrame(columns=['doc_ID', 'spectrum_ID', 'sub_ID', 'parent_mz', 'parent_intensity', 'no_peaks_losses'])
    doc_counter = 0
    
    for spec_id, spectrum in enumerate(spectra):
        doc = []
        doc_intensity = []
        losses = np.array(spectrum.losses)
        if len(losses) > 0: 
            keep_idx = np.where((losses[:,0] > min_loss) & (losses[:,0] < max_loss))[0]
            losses = losses[keep_idx,:]
        else:
            print("No losses detected for: ", spec_id, spectrum.id)
        peaks = np.array(spectrum.peaks).astype(float)
        
        # Sort peaks and losses by m/z 
        peaks = peaks[np.lexsort((peaks[:,1], peaks[:,0])),:]
        if len(losses) > 0: 
            losses = losses[np.lexsort((losses[:,1], losses[:,0])),:]
            
        # Normalize intensities
        max_peak_intens = np.max(peaks[:,1])
        peaks[:,1] = peaks[:,1]/max_peak_intens
        if len(losses) > 0:
            losses[:,1] = losses[:,1]/max_peak_intens 
        
        # Identify main peaks for createing sub-spectra:
        main_peak_idx = np.where(peaks[:,1] > main_peak_cutoff)[0]
#        if peaks[main_peak_idx[-1],0] > spectrum.parent_mz
        main_peaks = peaks[main_peak_idx,:]
        
        sub_peaks_lst = []
        sub_losses_lst = []
        if len(main_peaks) > 0:          
#            sub_spectra_reg[str(spec_id)] = peaks[main_peak_idx,:]
            for m in range(len(main_peaks)): 
                sub_peaks = peaks[(peaks[:,0] < main_peaks[m,0]),:]
                sub_peaks_lst.append(sub_peaks)
                # Calculate new losses with respect to main peak!
                sub_losses = sub_peaks
                sub_losses[:,0] = main_peaks[m,0] - sub_losses[:,0]
                if len(sub_losses) > 0: 
                    keep_idx = np.where((sub_losses[:,0] > min_loss) & (sub_losses[:,0] < max_loss))[0]
                    sub_losses = sub_losses[keep_idx,:]
                sub_losses_lst.append(sub_losses)

        if (spec_id+1) % 100 == 0 or spec_id == len(spectra)-1:  # show progress
                print('\r', ' Created documents for ', spec_id+1, ' of ', len(spectra), ' spectra.', end="")
        
        # Create document from whole spectra
        for i in range(len(peaks)):
            doc.append("peak_" + "{:.{}f}".format(peaks[i,0], num_decimals))
            doc_intensity.append(peaks[i,1])
#            doc_intensity.append(int(peaks[i,1]))
            
        for i in range(len(losses)):
            doc.append("loss_"  + "{:.{}f}".format(losses[i,0], num_decimals))
            doc_intensity.append(losses[i,1])
#            doc_intensity.append(int(losses[i,1]))

        MS_documents.append(doc)
        MS_documents_intensity.append(doc_intensity)
#        sub_spectra_metadata.loc[doc_counter] = [doc_counter, spec_id, 0, spectrum.parent_mz, 1, len(doc)]
        sub_spectra_metadata.loc[doc_counter] = [doc_counter, int(spectrum.id), 0, spectrum.parent_mz, 1, len(doc)]
        doc_counter += 1
        
        
        # Create document from sub-spectra
        for m, sub_peaks in enumerate(sub_peaks_lst):
            subdoc = []
            subdoc_intensity = []
            for i in range(len(sub_peaks)):
                subdoc.append("peak_" + "{:.{}f}".format(sub_peaks[i,0], num_decimals))
                subdoc_intensity.append(sub_peaks[i,1])
#                subdoc_intensity.append(int(sub_peaks[i,1]))
    
            sub_losses = sub_losses_lst[m]
            for i in range(len(sub_losses)):
                subdoc.append("loss_"  + "{:.{}f}".format(sub_losses[i,0], num_decimals))
                subdoc_intensity.append(sub_losses[i,1])
#                subdoc_intensity.append(int(sub_losses[i,1]))
            
            if len(subdoc) >= min_words:
                MS_documents.append(subdoc)
                MS_documents_intensity.append(subdoc_intensity)
                sub_spectra_metadata.loc[doc_counter] = [doc_counter, spec_id, m+1, main_peaks[m,0], main_peaks[m,1], len(subdoc)] #sub_spectra_reg[str(i)][m,0]]
                doc_counter += 1
         
    return MS_documents, MS_documents_intensity, sub_spectra_metadata



def get_mol_fingerprints(spectra_dict, method = "daylight"):
    """ Calculate molecule fingerprints based on given smiles.
    (using RDkit)
    
    Output: exclude_IDs list with spectra that had no smiles or problems when deriving fingerprint
    
    Args:
    --------
    spectra_dict: dict
        Dictionary containing all spectrum objects information (peaks, losses, metadata...).
    method: str
        
    """
    # If spectra is given as a dictionary
    keys = []
    smiles = []
    exclude_IDs = []
    for key, value in spectra_dict.items():
        if "smiles" in value:
            keys.append(key) 
            smiles.append(value["smiles"])
        else:
            print("No smiles found for spectra ", key, ".")
            smiles.append("H20")  # just have some water when you get stuck
            exclude_IDs.append(int(value["id"]))
   
    molecules = [Chem.MolFromSmiles(x) for x in smiles]
    fingerprints = []
    for i in range(len(molecules)):
        if molecules[i] is None:
            print("Problem with molecule " + spectra_dict[keys[i]]["id"])
            fp = 0
            exclude_IDs.append(int(spectra_dict[keys[i]]["id"]))
        else:
            if method == "daylight":
                fp = FingerprintMols.FingerprintMol(molecules[i])
            elif method == "morgan1":
                fp = AllChem.GetMorganFingerprint(molecules[i],1)
            elif method == "morgan2":
                fp = AllChem.GetMorganFingerprint(molecules[i],2)
            elif method == "morgan3":
                fp = AllChem.GetMorganFingerprint(molecules[i],3)
            else:
                print("Unkown fingerprint method given...")

        fingerprints.append(fp)
                
#    fingerprints = [FingerprintMols.FingerprintMol(x) for x in molecules]
    
    return molecules, fingerprints, exclude_IDs


def compare_molecule_selection(query_id, spectra_dict, MS_measure, 
                               fingerprints,
                               num_candidates = 25, 
                               similarity_method = "centroid"):
    """ Compare spectra-based similarity with smile-based similarity
    
    Args:
    -------
    query_id: int
        Number of spectra to use as query.
    spectra_dict: dict
        Dictionary containing all spectra peaks, losses, metadata.
    MS_measure: object
        Similariy object containing the model and distance matrices.
    fingerprints: object
        Fingerprint objects for all molecules (if smiles exist for the spectra).
    num_candidates: int
        Number of candidates to list (default = 25) .
    similarity_method: str
        Define method to use (default = "centroid").
    """
    
    # Select chosen similarity methods
    if similarity_method == "centroid":
        candidates_idx = MS_measure.list_similars_ctr_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_ctr[query_id, :num_candidates]
    elif similarity_method == "pca":
        candidates_idx = MS_measure.list_similars_pca_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_pca[query_id, :num_candidates]
    elif similarity_method == "autoencoder":
        candidates_idx = MS_measure.list_similars_ae_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_ae[query_id, :num_candidates]
    elif similarity_method == "lda":
        candidates_idx = MS_measure.list_similars_lda_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_lda[query_id, :num_candidates]
    elif similarity_method == "lsi":
        candidates_idx = MS_measure.list_similars_lsi_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_lsi[query_id, :num_candidates]
    elif similarity_method == "doc2vec":
        candidates_idx = MS_measure.list_similars_d2v_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_d2v[query_id, :num_candidates]
    else:
        print("Chosen similarity measuring method not found.")
        
    mol_sim = np.zeros((len(fingerprints)))
    if fingerprints[query_id] != 0:
        for j in range(len(fingerprints)):
            if fingerprints[j] != 0:     
                mol_sim[j] = DataStructs.FingerprintSimilarity(fingerprints[query_id], fingerprints[j])
                
    smiles_similarity = np.array([np.arange(0, len(mol_sim)), mol_sim]).T
    smiles_similarity = smiles_similarity[np.lexsort((smiles_similarity[:,0], smiles_similarity[:,1])),:]
    
    print("Selected candidates based on spectrum: ")
    print(candidates_idx)
    print("Selected candidates based on smiles: ")
    print(smiles_similarity[:num_candidates,0])
    print("Selected candidates based on spectrum: ")
    for i in range(num_candidates):
        print("id: "+ str(candidates_idx[i]) + " (similarity: " +  str(candidates_dist[i]) + " | Tanimoto: " + str(mol_sim[candidates_idx[i]]) +")")



def evaluate_measure(spectra_dict, 
                     spectra,
                     MS_measure, 
                       fingerprints,
                       num_candidates = 25,
                       num_of_molecules = "all", 
                       similarity_method = "centroid",
                       molnet_sim = None,
                       reference_list = None):
    """ Compare spectra-based similarity with smile-based similarity.
    
    Output:
    -------
    mol_sim: matrix with molecule similarity scores for TOP 'num_candidates' for 'num_of_molecules'.
    spec_sim: matrix with spectra similarity for TOP 'num_candidates' for 'num_of_molecules' (using 'similarity_method').
    spec_idx: matrix with spectra IDs corresponding to spec_sim values.
    reference_list: list of selected 'num_of_molecules'. Will contain all IDs if num_of_molecules = "all".
        
    Args:
    -------
    spectra_dict: dict
        Dictionary containing all spectra peaks, losses, metadata.
    MS_measure: object
        Similariy object containing the model and distance matrices.
    fingerprints: object
        Fingerprint objects for all molecules (if smiles exist for the spectra).
    num_candidates: int
        Number of candidates to list (default = 25) .
    num_of_molecules: int
        Number of molecules to test method on (default= 100)
    similarity_method: str
        Define method to use (default = "centroid").
    """
    num_spectra = len(MS_measure.corpus)
    
    # Create reference list if not given as args:
    if reference_list is None:
        if num_of_molecules == "all":
            reference_list = np.arange(num_spectra)
        elif isinstance(num_of_molecules, int): 
            reference_list = np.array(random.sample(list(np.arange(len(fingerprints))),k=num_of_molecules))
        else:
            print("num_of_molecules needs to be integer or 'all'.")
        
    mol_sim = np.zeros((len(reference_list), num_candidates))
    spec_sim = np.zeros((len(reference_list), num_candidates))
    spec_idx = np.zeros((len(reference_list), num_candidates))
#    molnet_sim = np.zeros((len(reference_list), num_candidates))
    
    candidates_idx = np.zeros((num_candidates), dtype=int)
    candidates_sim = np.zeros((num_candidates))
    
    for i, query_id in enumerate(reference_list):
        # Show progress:
        
        if (i+1) % 10 == 0 or i == len(reference_list)-1:  
                print('\r', ' Evaluate spectrum ', i+1, ' of ', len(reference_list), ' spectra.', end="")

        # Select chosen similarity methods
        if similarity_method == "centroid":
            candidates_idx = MS_measure.list_similars_ctr_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_ctr[query_id, :num_candidates]
        elif similarity_method == "pca":
            candidates_idx = MS_measure.list_similars_pca_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_pca[query_id, :num_candidates]
        elif similarity_method == "autoencoder":
            candidates_idx = MS_measure.list_similars_ae_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_ae[query_id, :num_candidates]
        elif similarity_method == "lda":
            candidates_idx = MS_measure.list_similars_lda_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_lda[query_id, :num_candidates]
        elif similarity_method == "lsi":
            candidates_idx = MS_measure.list_similars_lsi_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_lsi[query_id, :num_candidates]
        elif similarity_method == "doc2vec":
            candidates_idx = MS_measure.list_similars_d2v_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_d2v[query_id, :num_candidates]
            
        elif similarity_method == "molnet":      
            candidates_idx = molnet_sim[i,:].argsort()[-num_candidates:][::-1]
            candidates_sim = molnet_sim[i, candidates_idx]
            
#            molnet_sim = np.zeros((num_spectra))
#            for cand_id in range(num_spectra):
#                molnet_sim[cand_id] = fast_cosine_shift(spectra[query_id], spectra[cand_id], 0.2, 2)
#            candidates_idx = molnet_sim.argsort()[-num_candidates:][::-1]
#            candidates_sim = molnet_sim[candidates_idx]                
        else:
            print("Chosen similarity measuring method not found.")

        # Check type of fingerprints given as input:
        try: 
            DataStructs.FingerprintSimilarity(fingerprints[0], fingerprints[0])
            fingerprint_type = "daylight" # at least assumed here
        
        except AttributeError:
            fingerprint_type = "morgan" # at least assumed here

        # Calculate Tanimoto similarity for selected candidates
        if fingerprints[query_id] != 0:
            for j, cand_id in enumerate(candidates_idx): 
                if fingerprints[cand_id] != 0:     
                    if fingerprint_type == "daylight":
                        mol_sim[i, j] = DataStructs.FingerprintSimilarity(fingerprints[query_id], fingerprints[cand_id])
                    elif fingerprint_type == "morgan":
                        mol_sim[i, j] = DataStructs.DiceSimilarity(fingerprints[query_id], fingerprints[cand_id])

        spec_sim[i,:] = candidates_sim
        spec_idx[i,:] = candidates_idx

    return mol_sim, spec_sim, spec_idx, reference_list


"""
From Simon Rogers
molnet
"""
def fast_cosine_shift(spectrum1, spectrum2, tol, min_match, min_intens = 0):
    if len(spectrum1.peaks) == 0 or len(spectrum2.peaks) == 0:
        return 0.0,[]

    spec1 = np.array(spectrum1.peaks, dtype=float)
    spec2 = np.array(spectrum2.peaks, dtype=float)
    
    # normalize intensities:
    spec1[:,1] = spec1[:,1]/max(spec1[:,1])
    spec2[:,1] = spec2[:,1]/max(spec2[:,1])
    
    # filter, if wanted:
    spec1 = spec1[spec1[:,1] > min_intens,:]
    spec2 = spec2[spec2[:,1] > min_intens,:]
    
#    # Sort by peak m/z:
#    spec1 = spec1[np.lexsort((spec1[:,1], spec1[:,0])),:]
#    spec2 = spec2[np.lexsort((spec2[:,1], spec2[:,0])),:]
    
    zero_pairs = find_pairs(spec1, spec2, tol, shift=0.0)

    shift = spectrum1.parent_mz - spectrum2.parent_mz

    nonzero_pairs = find_pairs(spec1, spec2, tol, shift = shift)

    matching_pairs = zero_pairs + nonzero_pairs

    matching_pairs = sorted(matching_pairs,key = lambda x: x[2], reverse = True)

    used1 = set()
    used2 = set()
    score = 0.0
    used_matches = []
    for m in matching_pairs:
        if not m[0] in used1 and not m[1] in used2:
            score += m[2]
            used1.add(m[0])
            used2.add(m[1])
            used_matches.append(m)
    
    if len(used_matches) < min_match:
        score = 0.0
    else:     
        # Normalize score:
        score = score/max(np.sum(spec1[:,1]**2), np.sum(spec2[:,1]**2))
    
    return score #, used_matches


def fast_cosine_shift2(spectrum1, spectrum2, tol, max_mz):
    """ try to speed up things
    DOESN't work well!!
    """
    if len(spectrum1.peaks) == 0 or len(spectrum2.peaks) == 0:
        return 0.0,[]

    spec1 = np.array(spectrum1.peaks.copy(), dtype=float)
    spec2 = np.array(spectrum2.peaks.copy(), dtype=float)
    
    # Normalize intensities:
    spec1[:,1] = spec1[:,1]/np.max(spec1[:,1])
    spec2[:,1] = spec2[:,1]/np.max(spec2[:,1])
    
#    # Sort by peak m/z: SOULD BE DONE ALREADY in spectrum object...
#    spec1 = spec1[np.lexsort((spec1[:,1], spec1[:,0])),:]
#    spec2 = spec2[np.lexsort((spec2[:,1], spec2[:,0])),:]
    
    spec1_onehot = one_hot_spectrum(spec1, tol, max_mz, shift=0, min_mz = 0)
    spec2_onehot = one_hot_spectrum(spec2, tol, max_mz, shift=0, min_mz = 0)
    
    shift = spectrum1.parent_mz - spectrum2.parent_mz
    spec2_onehot_shift = one_hot_spectrum(spec2, tol, max_mz, shift=shift, min_mz = 0)
    
    peak_pairs = np.zeros((len(spec1_onehot),2))
    peak_pairs[:,0] = spec1_onehot * spec2_onehot
    peak_pairs[:,1] = spec1_onehot * spec2_onehot_shift

    matching_pairs = np.amax(peak_pairs, axis=1)

#    #score (not normalized)
#    score = np.sum(matching_pairs)

    # Score (normalized)
    score = np.sum(matching_pairs)/max(np.sum(spec1_onehot**2), np.sum(spec2_onehot**2))
  
    return score

    
def fast_cosine_shift_hungarian(spectrum1, spectrum2, tol, min_match, min_intens=0):
    """ Taking full care of weighted bipartite matching problem:
        Use Hungarian algorithm (slow...)
    """
    if len(spectrum1.peaks) == 0 or len(spectrum2.peaks) == 0:
        return 0.0,[]

    spec1 = np.array(spectrum1.peaks, dtype=float)
    spec2 = np.array(spectrum2.peaks, dtype=float)
    
    # normalize intensities:
    spec1[:,1] = spec1[:,1]/max(spec1[:,1])
    spec2[:,1] = spec2[:,1]/max(spec2[:,1])
    
    # filter, if wanted:
    spec1 = spec1[spec1[:,1] > min_intens,:]
    spec2 = spec2[spec2[:,1] > min_intens,:]
    
    zero_pairs = find_pairs(spec1, spec2, tol, shift=0.0)

    shift = spectrum1.parent_mz - spectrum2.parent_mz

    nonzero_pairs = find_pairs(spec1, spec2, tol, shift = shift)

    matching_pairs = zero_pairs + nonzero_pairs

    # Use Hungarian_algorithm:
    set1 = set()
    set2 = set()
    for m in matching_pairs:
        set1.add(m[0])
        set2.add(m[1])
    
    list1 = list(set1)
    list2 = list(set2)
    matrix_size = max(len(set1), len(set2))    
    matrix = np.ones((matrix_size, matrix_size))
    
    for m in matching_pairs:
        matrix[list1.index(m[0]),list2.index(m[1])] = 1 - m[2]

    row_ind, col_ind = linear_sum_assignment(matrix)
    score = matrix.shape[0] - matrix[row_ind, col_ind].sum()
        
    # normalize score:
    score = score/max(np.sum(spec1[:,1]**2), np.sum(spec2[:,1]**2))
    
    return score


def molnet_matrix(spectra, 
                  tol, 
                  max_mz, min_mz = 0, 
                  min_match = 2, min_intens = 0.01,
                  filename = None,
                  method='fast',
                  num_workers = 4,
                  safety_points = 50):
    """ Create Matrix of all mol.networking similarities.
    Takes some time to calculate, so better only do it once and save as npy.
    Now implemented: parallelization of code using concurrent.futures.
    
    spectra: list
        List of spectra (of Spectrum class)
    tol: float
        Tolerance to still count peaks a match (mz +- tolerance).
    max_mz: float
        Maxium m-z mass to take into account
    min_mz: float 
        Minimum m-z mass to take into account
    min_match: int
        Minimum numbe of peaks that need to be matches. Otherwise score will be set to 0
    min_intens: float
        Sets the minimum relative intensity peaks must have to be looked at for potential matches.
    filename: str/ None
        Filename to look for existing npy-file with molent matrix. Or, if not found, to 
        use to save the newly calculated matrix.
    method: 'fast' | 'hungarian'
        "Fast" will use Simon's molnet scoring which is much faster, but not 100% accurate
        regarding the weighted bipartite matching problem.
        "hungarian" will use the Hungarian algorithm, which is slower but more accurate.
    num_workers: int
        Number of threads to use for calculation. 
    safety_points: int
        Number of safety points, i.e. number of times the molnet-matrix is saved during process.
    """  
    if filename is not None:
        try: 
            molnet_sim = np.load(filename)
            print("MolNet similarity scores found and loaded.")
            collect_new_data = False
                
        except FileNotFoundError: 
            print("Could not find file ", filename) 
            print("MolNet scores will be calculated from scratch.")
            collect_new_data = True
    
    if collect_new_data == True:      
        molnet_sim = np.zeros((len(spectra), len(spectra)))
        
        counter = 0
        safety_save = int(((len(spectra)**2)/2)/safety_points)  # Save molnet-matrix along process
        print("Calculate pairwise MolNet scores by ", num_workers, "number of workers.")
        for i in range(len(spectra)):
            parameter_collection = []    
            for j in range(i,len(spectra)):
                parameter_collection.append([spectra[i], spectra[j], i, j, tol, min_match, min_intens, method, counter])
                counter += 1
# OLD code w/o parallelization:
            # Show progress
#            if (i+1) % 10 == 0 or i == len(spectra)-1:  
#                print('\r', ' Molnet for spectrum ', i+1, ' of ', len(spectra), ' spectra.', end="")
#        
#            for j in range(i,len(spectra)):
#                if method == 'fast':
#                    molnet_sim[i,j] = fast_cosine_shift(spectra[i], spectra[j], tol, min_match, min_intens = min_intens)
#                elif method == 'hungarian':
#                    molnet_sim[i,j] = fast_cosine_shift_hungarian(spectra[i], spectra[j], tol, min_match, min_intens = min_intens)
#                else:
#                    print("Given method does not exist...")

            # Create a pool of processes. For instance one for each CPU in your machine.
            molnet_pairs = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(molnet_pair, X, len(spectra)) for X in parameter_collection]
                molnet_pairs.append(futures)
             
            for m, future in enumerate(molnet_pairs):
                spec_i, spec_j, ind_i, ind_j, _, _, _, _, _ = parameter_collection[m]
                molnet_sim[ind_i,ind_j] = future[0].result()
                
            if (counter+1) % safety_save == 0:
                np.save(filename, molnet_sim)

        # Symmetric matrix --> fill        
        for i in range(1,len(spectra)):
            for j in range(i):  
                molnet_sim[i,j] = molnet_sim[j,i]      
    
        if filename is not None:
            np.save(filename, molnet_sim)
            
    return molnet_sim


def molnet_pair(X, len_spectra):
    """ Single molnet pair calculation
    """ 
    spectra_i, spectra_j, i, j, tol, min_match, min_intens, method, counter = X
    if method == 'fast':
        molnet_pair = fast_cosine_shift(spectra_i, spectra_j, tol, min_match, min_intens = min_intens)
    elif method == 'hungarian':
        molnet_pair = fast_cosine_shift_hungarian(spectra_i, spectra_j, tol, min_match, min_intens = min_intens)
    else:
        print("Given method does not exist...")

    if (counter+1) % 1000 == 0 or counter == len_spectra-1:  
        print('\r', ' Calculated MolNet for pair ', i, '--', j, '. ( ', np.round(200*(counter+1)/len_spectra**2, 2), ' % done).', end="")

    return molnet_pair


def tanimoto_matrix(spectra, 
                  fingerprints,
                  filename = None):
    """ Create Matrix of all Tanimoto molecule similarities (based on annotated SMILES).
    Takes some time to calculate, so better only do it once and save as npy.
    """  
    
    if filename is not None:
        try: 
            tanimoto_similarities = np.load(filename)
            print("Tanimoto similarity scores found and loaded.")
            collect_new_data = False
                
        except FileNotFoundError: 
            print("Could not find file ", filename) 
            print("Tanimoto scores will be calculated from scratch.")
            collect_new_data = True
    
    if collect_new_data == True:      
        
        # Check type of fingerprints given as input:
        try: 
            DataStructs.FingerprintSimilarity(fingerprints[0], fingerprints[0])
            fingerprint_type = "daylight" # at least assumed here
        
        except AttributeError:
            fingerprint_type = "morgan" # at least assumed here
        
        tanimoto_similarities = np.zeros((len(spectra), len(spectra)))
        for i in range(len(spectra)):
            # Show progress
            if (i+1) % 10 == 0 or i == len(spectra)-1:  
                print('\r', ' Tanimoto for spectrum ', i+1, ' of ', len(spectra), ' spectra.', end="")
            if fingerprints[i] != 0:
                for j in range(i,len(spectra)):
                    if fingerprints[j] != 0: 
                        if fingerprint_type == "daylight":
                            tanimoto_similarities[i,j] = DataStructs.FingerprintSimilarity(fingerprints[i], fingerprints[j])
                        elif fingerprint_type == "morgan":
                            tanimoto_similarities[i,j] = DataStructs.DiceSimilarity(fingerprints[i], fingerprints[j])
        
        # Symmetric matrix --> fill        
        for i in range(1,len(spectra)):
            for j in range(i):  
                tanimoto_similarities[i,j] = tanimoto_similarities[j,i]   
    
        if filename is not None:
            np.save(filename, tanimoto_similarities)

    return tanimoto_similarities


def one_hot_spectrum(spec, tol, max_mz, shift = 0, min_mz = 0):
    """ Convert spectrum peaks into on-hot-vector
    """
    dim_vector = int((max_mz - min_mz)/tol)
    one_hot_spec = np.zeros((dim_vector))
    idx = ((spec[:,0] + shift)*1/tol).astype(int)
    idx[idx>=dim_vector] = 0
    idx[idx<0] = 0
    one_hot_spec[idx] = spec[:,1]
    
    return one_hot_spec
    

def find_pairs(spec1, spec2, tol, shift=0):
    matching_pairs = []
    spec2lowpos = 0
    spec2length = len(spec2)
    
    for idx in range(len(spec1)):
#        ,(mz,intensity) in enumerate(spec1):
        mz = spec1[idx,0]
        intensity = spec1[idx,1]
        # do we need to increase the lower idx?
        while spec2lowpos < spec2length and spec2[spec2lowpos][0] + shift < mz - tol:
            spec2lowpos += 1
        if spec2lowpos == spec2length:
            break
        spec2pos = spec2lowpos
        while(spec2pos < spec2length and spec2[spec2pos][0] + shift < mz + tol):
            matching_pairs.append((idx, spec2pos, intensity*spec2[spec2pos][1]))
            spec2pos += 1
        
    return matching_pairs    



def find_pairs2(spec1, spec2, tol, shift=0):
    """ Alternative numpy based approach (check what's faster)
    TODO: seems slower, check again?
    """
    matching_pairs = []
    
    cand_idx = []
    for idx in range(len(spec1)):
        cands = np.where(np.abs(spec2[:,0] + shift - spec1[idx,0]) < tol)[0]
        for cand in cands:
            cand_idx.append((idx, cand))

    for (idx, idx2) in cand_idx:
        matching_pairs.append((idx, idx2, spec1[idx,1]*spec2[idx2,1])) 
        
    return matching_pairs  


def find_all_pairs(spectra, query_id, tol, min_match = 2):
    """ Alternative numpy based approach (check what's faster)
    TODO: so far seem to be much slower!!!
    """
 
    similarities = np.zeros((len(spectra)))

    max_peaks = 0
    parent_mzs = np.zeros((len(spectra)))
    for i, spec in enumerate(spectra):
        if len(spec.peaks) > max_peaks:
            max_peaks = len(spec.peaks)
        parent_mzs[i] = spec.parent_mz
    
    # Create two numpy arrays for all peaks and intensities        
    peaks_all = np.zeros((len(spectra), max_peaks))      
    intensities_all = np.zeros((len(spectra), max_peaks))       
    
    for i, spec in enumerate(spectra):
        peaks = np.array(spec.peaks)
        peaks_all[i,:len(peaks)] = peaks[:,0]
        intensities_all[i,:len(peaks)] = peaks[:,1]/max(peaks[:,1])
               
#    found_pairs = []

    for idx in range(len(spectra[query_id].peaks)):
        cands = np.where(np.abs(peaks_all - peaks_all[query_id, idx]) < tol)
        cands_arr = np.array(cands)
        pairs = np.zeros((5, cands_arr.shape[1]))
        
        pairs[0,:] = query_id
        pairs[1,:] = idx
        pairs[2:4,:] = cands_arr
        pairs[4,:] = intensities_all[query_id, idx] * intensities_all[cands]
        if idx == 0:
            found_pairs = pairs
        else:
            found_pairs = np.concatenate((found_pairs, pairs), axis=1)
        
    found_pairs = found_pairs[:,np.lexsort((found_pairs[3,:], 
                                            found_pairs[1,:], 
                                            found_pairs[2,:]))]
    
#        shift = spectrum1.parent_mz - spectrum2.parent_mz
    # Calculate shift matrix   
    shift_matrix = np.tile((spectra[query_id].parent_mz - parent_mzs), (max_peaks,1)).T
    

    # SAME with parent m/z shift ----------------------------------------------
    for idx in range(len(spectra[query_id].peaks)):
        cands_shift = np.where(np.abs(peaks_all + shift_matrix - peaks_all[query_id, idx]) < tol)
        cands_arr_shift = np.array(cands_shift)
        pairs_shift = np.zeros((5, cands_arr_shift.shape[1]))
        
        pairs_shift[0,:] = query_id
        pairs_shift[1,:] = idx
        pairs_shift[2:4,:] = cands_arr_shift
        pairs_shift[4,:] = intensities_all[query_id, idx] * intensities_all[cands_shift]
        if idx == 0:
            found_pairs_shift = pairs_shift
        else:
            found_pairs_shift = np.concatenate((found_pairs_shift, pairs_shift), axis=1)
        
    found_pairs_shift = found_pairs_shift[:,np.lexsort((found_pairs_shift[3,:], 
                                                        found_pairs_shift[1,:], 
                                                        found_pairs_shift[2,:]))]
    # Select all candidates with matching peaks:
    set_cands = set()
    set_cands.update(found_pairs[2,:].astype(int))
    set_cands.update(found_pairs_shift[2,:].astype(int))

    matching_pairs = []
    matching_pairs_shift = []
    for x in list(set_cands):  
        idx = np.where(found_pairs[2,:] == x)[0]
        matching_pairs = list(zip(found_pairs[1, idx].astype(int), 
                                  found_pairs[3, idx].astype(int), 
                                  found_pairs[4, idx]))

        idx = np.where(found_pairs_shift[2,:] == x)[0]
        matching_pairs_shift = list(zip(found_pairs_shift[1, idx].astype(int), 
                                        found_pairs_shift[3, idx].astype(int), 
                                        found_pairs_shift[4, idx]))


        matching_pairs_total = matching_pairs + matching_pairs_shift
        matching_pairs_total = sorted(matching_pairs_total, key = lambda x: x[2], reverse = True)
    
        used1 = set()
        used2 = set()
        score = 0.0
        used_matches = []
        for m in matching_pairs:
            if not m[0] in used1 and not m[1] in used2:
                score += m[2]
                used1.add(m[0])
                used2.add(m[1])
                used_matches.append(m)
        if len(used_matches) < min_match:
            score = 0.0
            
        # normalize score:
        score = score/max(np.sum(intensities_all[query_id,:]**2), 
                          np.sum(intensities_all[x,:]**2))
        similarities[x] = score
        
    return similarities 



## --------------------------------------------------------------------------------------------------
## ---------------------------- Plotting functions --------------------------------------------------
## --------------------------------------------------------------------------------------------------
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def get_spaced_colors_hex(n):
    """ Create set of 'n' well-distinguishable colors
    """
    spaced_colors = ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "000000", 
        "800000", "008000", "000080", "808000", "800080", "008080", "808080", 
        "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0", "C0C0C0", 
        "400000", "004000", "000040", "404000", "400040", "004040", "404040", 
        "200000", "002000", "000020", "202000", "200020", "002020", "202020", 
        "600000", "006000", "000060", "606000", "600060", "006060", "606060", 
        "A00000", "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0", 
        "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0"]
    
    RGB_colors = ["#"+x for x in spaced_colors[:n] ]
    

    return RGB_colors


def plot_spectra(spectra, compare_ids, min_mz = 50, max_mz = 500):
    """ Plot different spectra together to compare.
    """
    plt.figure(figsize=(10,10))

    peak_number = []
    RGB_colors = get_spaced_colors_hex(len(compare_ids))
    for i, id in enumerate(compare_ids):
        peaks = np.array(spectra[id].peaks.copy())
        peak_number.append(len(peaks))
        peaks[:,1] = peaks[:,1]/np.max(peaks[:,1]); 

        markerline, stemlines, baseline = plt.stem(peaks[:,0], peaks[:,1], linefmt='-', markerfmt='.', basefmt='r-')
        plt.setp(stemlines, 'color', RGB_colors[i])
    
    plt.xlim((min_mz, max_mz))
    plt.grid(True)
    plt.title('Spectrum')
    plt.xlabel('m/z')
    plt.ylabel('peak intensity')
    
    plt.show()
    
    print("Number of peaks: ", peak_number)


def plot_losses(spectra, compare_ids, min_loss = 0, max_loss = 500):
    """ Plot different spectra together to compare.
    """
    plt.figure(figsize=(10,10))

    losses_number = []
    RGB_colors = get_spaced_colors_hex(len(compare_ids)+5)
    for i, id in enumerate(compare_ids):
        losses = np.array(spectra[id].losses.copy())
        losses_number.append(len(losses))
        losses[:,1] = losses[:,1]/np.max(losses[:,1]); 

        markerline, stemlines, baseline = plt.stem(losses[:,0], losses[:,1], linefmt='-', markerfmt='.', basefmt='r-')
        plt.setp(stemlines, 'color', RGB_colors[i])
    
    plt.xlim((min_loss, max_loss))
    plt.grid(True)
    plt.title('Spectrum')
    plt.xlabel('m/z')
    plt.ylabel('peak intensity')
    
    plt.show()
    
    print("Number of peaks: ", losses_number)






from rdkit import Chem
from rdkit.Chem import Draw

def plot_smiles(query_id, spectra, MS_measure, num_candidates = 10,
                   sharex=True, labels=False, similarity_method = "centroid",
                   plot_type = "single", molnet_sim = None):
    """ Plot molecules for closest candidates
    
    """

    # Select chosen similarity methods
    if similarity_method == "centroid":
        candidates_idx = MS_measure.list_similars_ctr_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_ctr[query_id, :num_candidates]
    elif similarity_method == "pca":
        candidates_idx = MS_measure.list_similars_pca_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_pca[query_id, :num_candidates]
    elif similarity_method == "autoencoder":
        candidates_idx = MS_measure.list_similars_ae_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_ae[query_id, :num_candidates]
    elif similarity_method == "lda":
        candidates_idx = MS_measure.list_similars_lda_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_lda[query_id, :num_candidates]
    elif similarity_method == "lsi":
        candidates_idx = MS_measure.list_similars_lsi_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_lsi[query_id, :num_candidates]
    elif similarity_method == "doc2vec":
        candidates_idx = MS_measure.list_similars_d2v_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_d2v[query_id, :num_candidates]
    elif similarity_method == "molnet":
        if molnet_sim is None:
            print("If 'molnet' is chosen as similarity measure, molnet-matrix needs to be provided.")
            print("Use molnet_matrix function.")
        else:
            candidates_idx = molnet_sim[query_id,:].argsort()[-num_candidates:][::-1]
            candidates_sim = molnet_sim[query_id, candidates_idx]
    else:
        print("Chosen similarity measuring method not found.")

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


def top_score_histogram(spec_sim, mol_sim, 
                        score_threshold, 
                        num_candidates, 
                        num_bins = 50, 
                        filename = None):
    """ Plot histogram of Tanimoto scores (mol_sim) of top selected candidates based on 
    spectrum similarity scores (spec_sim). 
    
    spec_sim, mol_sim : to be calculated with evaluate_measure function.
    
    filename: str
        If not none: save figure to file with given name.
    """
    
    fig, ax = plt.subplots(figsize=(10,10))

    selection = np.where(spec_sim[:,1:] > score_threshold)
    X = mol_sim[selection[0], selection[1]+1].reshape(len(selection[0]))
    n, bins, patches = plt.hist(X, num_bins, weights=np.ones(len(X))/len(X), facecolor='blue', edgecolor='white', alpha=0.9)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title("Tanimoto scores of TOP " + str(num_candidates-1) + " candidates with score > " + str(score_threshold))
    plt.xlabel("Tanimoto score (based on spectra annotated SMILES)")
    plt.ylabel("Percentage")

    test = spec_sim[:,1:].reshape(spec_sim.shape[0]*(spec_sim.shape[1]-1))
    test.sort()
    text1 = "Mean Tanimoto similarity is " + str(np.round(np.mean(mol_sim[selection[0], selection[1]+1]), 4))
    text2 = "Spectrum similarity score for TOP " + str(num_candidates-1) + ", top 20% is " + str(np.round(test[int(len(test)*0.8)], 4))
    text3 = ""
    plt.text(0, 0.96*np.max(n), text1, fontsize=12, backgroundcolor = "white")
    plt.text(0, 0.91*np.max(n), text2, fontsize=12, backgroundcolor = "white")
    plt.text(0, 0.86*np.max(n), text3, fontsize=12, backgroundcolor = "white")

    if filename is not None:
        plt.savefig(filename, dpi=600)
    
    plt.show()


def similarity_histogram(M_sim, M_sim_ref, 
                         score_threshold,
                         num_bins = 50, 
                         exclude_IDs = None,
                         filename = None,
                         exclude_diagonal = True):
    """ Plot histogram of Reference scores (from matrix M_sim_ref) for all pairs 
    with similarity score >= score_threshold. 
    
    M_sim: numpy array
        Matrix with similarities between pairs.
    M_sim_ref: numpy array
        Matrix with reference scores/similarity values between pairs.
    
    filename: str
        If not none: save figure to file with given name.
    """
    fig, ax = plt.subplots(figsize=(10,10))
    
    if exclude_IDs is not None:
        # Remove elements in exclude_IDs array
        IDs = np.arange(0,M_sim.shape[0])
        M_sim = np.delete(M_sim, IDs[exclude_IDs], axis=0)
        M_sim = np.delete(M_sim, IDs[exclude_IDs], axis=1)
        M_sim_ref = np.delete(M_sim_ref, IDs[exclude_IDs], axis=0)
        M_sim_ref = np.delete(M_sim_ref, IDs[exclude_IDs], axis=1)
        
        IDs = np.delete(IDs, IDs[exclude_IDs])
        
    if exclude_diagonal == True:
        # Exclude diagonal
        M_sim[np.arange(0,M_sim.shape[0]), np.arange(0,M_sim.shape[0])] = score_threshold - 1
    
    selection = np.where(M_sim[:,:] >= score_threshold)
    X = M_sim_ref[selection].reshape(len(selection[0]))
    n, bins, patches = plt.hist(X, num_bins, weights=np.ones(len(X))/len(X), facecolor='blue', edgecolor='white', alpha=0.9)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title("Total reference scores for all candidates with similarity score > " + str(score_threshold), fontsize = 16)
#    plt.set_xticklabels(rotation=0, fontsize=12)
    ax.tick_params(labelsize=12)   
    plt.xlabel("Reference score.", fontsize = 14)
    plt.ylabel("Percentage", fontsize = 14)

    if filename is not None:
        plt.savefig(filename, dpi=600)
    
    plt.show()

    return n, bins


def compare_best_results(spectra_dict, 
                         spectra,
                         MS_measure,
                         tanimoto_sim,
                         molnet_sim,
                         num_candidates = 25,
                         similarity_method = ["centroid"]):
    """ Compare spectra-based similarity with smile-based similarity and mol.networking.

    Args:
    -------
    spectra_dict: dict
        Dictionary containing all spectra peaks, losses, metadata.
    MS_measure: object
        Similariy object containing the model and distance matrices.
    tanimoto_sim: numpy array
        Matrix of Tanimoto similarities between SMILES of spectra.
    molnet_sim: numpy array
        Matrix of mol. networking similarities of spectra.
    num_candidates: int
        Number of candidates to list (default = 25) .
    similarity_method: str
        Define method to use (default = "centroid").
    """
    num_spectra = len(spectra)
        
    spec_best = np.zeros((num_spectra, num_candidates, len(similarity_method)))
#    spec_best_idx = np.zeros((num_spectra, num_candidates))
    mol_best = np.zeros((num_spectra, num_candidates))
    tanimoto_best = np.zeros((num_spectra, num_candidates))
    
    candidates_idx = np.zeros((num_candidates), dtype=int)
    candidates_sim = np.zeros((num_candidates))
    for k, method in enumerate(similarity_method):
        for i in range(num_spectra):
            # Select chosen similarity methods
            if method == "centroid":
                candidates_idx = MS_measure.list_similars_ctr_idx[i, :num_candidates]
            elif method == "pca":
                candidates_idx = MS_measure.list_similars_pca_idx[i, :num_candidates]
            elif method == "autoencoder":
                candidates_idx = MS_measure.list_similars_ae_idx[i, :num_candidates]
            elif method == "lda":
                candidates_idx = MS_measure.list_similars_lda_idx[i, :num_candidates]
            elif method == "lsi":
                candidates_idx = MS_measure.list_similars_lsi_idx[i, :num_candidates]
            elif method == "doc2vec":
                candidates_idx = MS_measure.list_similars_d2v_idx[i, :num_candidates]
            else:
                print("Chosen similarity measuring method not found.")

            candidates_sim = tanimoto_sim[i, candidates_idx]
            spec_best[i,:,k] = candidates_sim

    for i in range(num_spectra):        
        # Compare to molecular networking score
        molnet_candidates_idx = molnet_sim[i,:].argsort()[-num_candidates:][::-1]
        molnet_candidates_sim = tanimoto_sim[i, molnet_candidates_idx]
        
        # Compare to maximum possible Tanimoto score
        tanimoto_candidates_idx = tanimoto_sim[i,:].argsort()[-num_candidates:][::-1]
        tanimoto_candidates_sim = tanimoto_sim[i, tanimoto_candidates_idx]     
                
        mol_best[i,:] = molnet_candidates_sim
        tanimoto_best[i,:] = tanimoto_candidates_sim

    labels = []
    avg_best_scores = []
    labels.append('Tanimoto (best)')
    avg_best_scores.append(np.mean(tanimoto_best, axis=0))
    labels.append('Mol.networking score')
    avg_best_scores.append(np.mean(mol_best, axis=0))
    
    for k, method in enumerate(similarity_method):
        labels.append('Spectrum similarity (' + method + ')')
        avg_best_scores.append(np.mean(spec_best[:,:,k], axis=0))

    return avg_best_scores, labels


def plot_best_results(avg_best_scores,  
                      labels,
                      tanimoto_sim,
                      filename = None):
    """ Plot best candidate average results.
    """
    
    num_candidates = len(avg_best_scores[0])

    # These are the colors that will be used in the plot
    color_sequence = ['#003f5c','#882556', '#D65113', '#ffa600', '#58508d', '#bc5090', 
                      '#2651d1', '#2f4b7c', '#ff6361', '#a05195', '#d45087'] 
    markers = ['^', 'v', 'o']#, 'v']
                      
    fig, ax = plt.subplots(figsize=(10,16))
    plt.subplot(211)
    for i, label in enumerate(labels):
        plt.plot(np.arange(0,num_candidates), avg_best_scores[i], 
                 label=label, linewidth=1, markersize=12,
                 marker=markers[min(i,len(markers)-1)], linestyle=':', color=color_sequence[i])
    
    # Add mean Tanimoto baseline
    plt.plot(np.arange(0,num_candidates), np.mean(tanimoto_sim)*np.ones((num_candidates)),
             label='Average Tanimoto similarity', linewidth=2, color='black')    

    plt.legend(fontsize = 12)
    plt.xticks(range(0, num_candidates), fontsize=12)
    plt.xlabel("Top 'x' candidates")
    plt.ylabel("Average Tanimoto score.")
    
#    fig, ax = plt.subplots(figsize=(10,8))
    plt.subplot(212)
    for i, label in enumerate(labels[1:], start=1):
        plt.plot(np.arange(1,num_candidates), avg_best_scores[i][1:]/avg_best_scores[0][1:], 
                 label=label+'/Tanimoto max', linewidth=1, markersize=12,
                 marker=markers[min(i,len(markers)-1)], linestyle=':', color=color_sequence[i])

    # Add mean Tanimoto baseline
    plt.plot(np.arange(1,num_candidates), np.mean(tanimoto_sim)*np.ones((num_candidates-1))/avg_best_scores[0][1:],
             label='Baseline: random candidate selection', linewidth=2, color='black')  
    
    plt.legend(fontsize = 12)
    plt.xticks(range(1, num_candidates), fontsize=12)
    plt.xlabel("Top 'x' candidates")
    plt.ylabel("Fraction of max. possible average Tanimoto score")
    
    if filename is not None:
        plt.savefig(filename, dpi=600)