""" Functions specific to MS data
(e.g. importing and data processing functions)

Florian Huber
Netherlands eScience Center, 2019

TODO: add liscence
"""

def get_losses(spectrum):
    """ Use spectrum class and extract peaks and losses
    Losses are here the differences between the spectrum precursor mz and the MS2 level peaks.
    """ 
    MS1_peak = spectrum.precursor_mz
    peaks = spectrum.peaks
    losses = []
    for i, peak in enumerate(peaks):
        losses.append((MS1_peak - peak[0], i))
    return losses


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
        losses = get_losses(spectrum)
        peaks = spectrum.peaks

        if (i+1) % 100 == 0 or i == len(spectra)-1:  # show progress
                print('\r', ' Created documents for ', i+1, ' of ', len(spectra), ' spectra.', end="")
                
        for peak in peaks:
            doc.append("peak_" + "{:.{}f}".format(peak[0], num_decimals))
            doc_intensity.append(int(peak[1]))
        
        for loss in losses:
            doc.append("loss_"  + "{:.{}f}".format(loss[0], num_decimals))
            
            # TODO: check what would be a good measure for intensity here!
            loss_intensity = peaks[loss[1]][1]
            doc_intensity.append(int(loss_intensity))
            
        MS_documents.append(doc)
        MS_documents_intensity.append(doc_intensity)
         
    return MS_documents, MS_documents_intensity