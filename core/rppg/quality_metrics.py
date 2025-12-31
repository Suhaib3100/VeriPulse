"""Signal quality assessment."""

import numpy as np

class QualityAnalyzer:
    def __init__(self):
        pass
    
    def analyze(self, signal, features):
        """
        Compute signal quality score based on features.
        
        Args:
            signal: The rPPG signal (unused here but kept for interface)
            features: Dictionary containing 'snr', 'periodicity', etc.
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        snr = features.get("snr", 0.0)
        periodicity = features.get("periodicity", 0.0)
        
        # Heuristic scoring
        # SNR > 3.0 is usually good
        # Periodicity > 0.5 is usually good (normalized autocorrelation)
        
        snr_score = np.clip(snr / 4.0, 0, 1)
        periodicity_score = np.clip(periodicity, 0, 1)
        
        # Weighted combination
        final_score = 0.6 * snr_score + 0.4 * periodicity_score
        
        return float(final_score)
