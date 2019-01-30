""" Functions specific to MS data
(e.g. importing and data processing functions)

Florian Huber
Netherlands eScience Center, 2019

TODO: add liscence
"""

import os
import helper_functions as functions
import fnmatch
import numpy as np
from scipy.optimize import curve_fit

class Spectrum(object):
    """ Spectrum class to store key information
    """
    def __init__(self, min_frag = 0.0, max_frag = 1000.0,
                 min_loss = 10.0, max_loss = 200.0,
                 min_intensity = 0.0, 
                 min_intensity_perc = 0.0,
                 exp_intensity_filter = 0.01,
                 min_peaks = 10,
                 merge_energies = True,
                 merge_ppm = 100,
                 replace = 'max'):

        self.id = id
        self.filename = []
        self.peaks = []
        self.precursor_mz = []
        self.parent_mz = []
        self.metadata = {}
        self.family = None
        self.annotations = []
#        self.InChiKey = []
        self.smiles = []
        
        self.min_frag = min_frag 
        self.max_frag = max_frag
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.min_intensity_perc = min_intensity_perc
        self.min_intensity = min_intensity
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
                        if intensity >= self.min_intensity:                   
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
        peaks = process_peaks(peaks, self.min_frag, self.max_frag,
                              self.min_intensity_perc, self.exp_intensity_filter, self.min_peaks)
        
        self.peaks = peaks
                
#                    functions.dict_to_json(self.BGC_data_dict, path_json + results_file)     
#                    # store documents (PFAM domains per BGC)
#                    with open(path_json + results_file[:-4] + "txt", "w") as f:
#                        for s in self.BGC_documents:
#                            f.write(str(s) +"\n")



    def get_losses(self):
        """ Process peaks
        Remove peaks outside window min_frag <-> max_frag.
        Remove peaks with intensities < min_intensity_perc/100*max(intensities)
        """
        """ Use spectrum class and extract peaks and losses
        Losses are here the differences between the spectrum precursor mz and the MS2 level peaks.
        
        Remove losses outside window min_loss <-> max_loss.
        """ 
        MS1_peak = self.parent_mz
#        peaks = self.peaks
        losses = self.peaks.copy()
        losses[:,0] = MS1_peak - losses[:,0]
        keep_idx = np.where((losses[:,0] > self.min_loss) & (losses[:,0] < self.max_loss))[0]
        
        self.losses = losses[keep_idx,:]


def process_peaks(peaks, min_frag = 0.0,max_frag = 1000.0,
                  min_intensity_perc = 0.0,
                  exp_intensity_filter = 0.01,
                  min_peaks = 10):
    """ Process peaks
    Remove peaks outside window min_frag <-> max_frag.
    Remove peaks with intensities < min_intensity_perc/100*max(intensities)
    
    Args:
    -------
    peaks: list, numpy array
        List or array of peaks (m/z, intensity).
    min_frag: float
        Minimum allowed fragment m/z.
    max_frag: float
        Maximum allowed fragment m/z.
    min_intensity_perc: float
        Minimium intensity (in precentage of the maximum intensity).
    exp_intensity_filter: float
        Uses exponential fit to intensity histogram. Threshold for maximum allowed peak
        intensity will be set where the exponential fit reaches exp_intensity_filter.
    min_peaks: int
        Minimum amount of peaks to keep (if so many exist in spectrum).
    
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
#    print("No of peaks after 1st filter: ", len(peaks))
    if (exp_intensity_filter is not None) and len(peaks) > 2*min_peaks:
        # Fit exponential to peak intensity distribution 
        num_bins = 100  # number of bins for histogram

        # remove highest peak
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
        
#        print("Set threshold at: ", threshold)
#        print("Would take ", np.sum(peaks[:,1] > threshold), " of ", len(peaks), "peaks.")
    
        keep_idx = np.where(peaks[:,1] > threshold)[0]
        if len(keep_idx) < min_peaks:
            peaks = peaks[np.lexsort((peaks[:,0], peaks[:,1])),:][-min_peaks:]
        else:
            peaks = peaks[keep_idx, :]
        return peaks
    else:
        return peaks
    

def load_MS_data(path_data, filefilter="*.*", 
                 min_frag = 0.0, max_frag = 1000.0,
                 min_loss = 10.0, max_loss = 200.0,
                 min_intensity = 0.0, 
                 min_intensity_perc = 0.0,
                 exp_intensity_filter = 0.01,
                 min_peaks = 10,
                 merge_energies = True,
                 merge_ppm = 10,
                 replace = 'max'):        
    """ Collect spectra from set of files
    Partly taken from ms2ldaviz 
    
    
    
    """
    
    dirs = os.listdir(path_data)
    spectra_files = fnmatch.filter(dirs, filefilter)
    
    spectra = []
    # run over all spectrum files:
    for i, file in enumerate(spectra_files):
        # Show progress
        if (i+1) % 10 == 0 or i == len(spectra_files)-1:  
            print('\r', ' Load spectrum ', i+1, ' of ', len(spectra_files), ' spectra.', end="")
        
        spectrum = Spectrum(min_frag = min_frag, 
                            max_frag = max_frag,
                            min_loss = min_loss, 
                            max_loss = max_loss,
                            min_intensity = min_intensity,
                            min_intensity_perc = min_intensity_perc,
                            exp_intensity_filter = exp_intensity_filter,
                            min_peaks = min_peaks,
                            merge_energies = merge_energies,
                            merge_ppm = merge_ppm,
                            replace = replace)
        spectrum.read_spectrum(path_data, file, i)
        spectrum.get_losses()
        spectra.append(spectrum)
        
    return spectra




def create_MS_documents(spectra, num_decimals):
    """ Create documents from peaks and losses.
    
    Every peak and every loss will be transformed into a WORD.
    Words then look like this: "peak_100.038" or "loss_59.240" 
    
    Args:
    --------
    spectra: list
        List of all spectrum class elements = all spectra to be in corpus
    num_decimals: int
        Number of decimals to take into account
    
    """
    MS_documents = []
    MS_documents_intensity = []
    
    for i, spectrum in enumerate(spectra):
        doc = []
        doc_intensity = []
        losses = spectrum.losses 
        peaks = spectrum.peaks

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




##
## ---------------------------- Plotting functions ----------------------------
## 

from rdkit import Chem
from rdkit.Chem import Draw


def plot_smiles(query_id, spectra, candidate_idx, candidate_dist, max_plot_dimension = 10,
                   sharex=True, labels=False, dist_method = "centroid"):
    """ Plot molecules for closest candidates
    
    """
    size = (200, 200)  # Smaller figures than the default
    idx = candidate_idx[query_id,:max_plot_dimension]

    for id in idx:
        m = Chem.MolFromSmiles(spectra[id].smiles)
        Draw.MolToMPL(m, size=size)

