import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_ultimate_blend():
    print("Loading Champion and Hedge submissions...")
    try:
        champ = pd.read_csv('winning_submission.csv')
        hedge = pd.read_csv('hedge_submission.csv')
    except FileNotFoundError:
        print("CRITICAL ERROR: Make sure both 'winning_submission.csv' and 'hedge_submission.csv' are in the current folder.")
        return

    # Ensure IDs match perfectly
    if not all(champ['id'] == hedge['id']):
        raise ValueError("CRITICAL ERROR: The IDs in the two submission files do not match!")

    print("Blending predictions (80% Champion, 20% Hedge)...")
    final = champ.copy()
    
    # The Safer Grandmaster Hedge Math
    champ_weight = 0.80
    hedge_weight = 0.20
    
    final['target'] = (champ['target'] * champ_weight) + (hedge['target'] * hedge_weight)
    
    # Format cleanly for submission
    final['target'] = np.round(final['target'], 3)

    final.to_csv('ultimate_blend.csv', index=False, lineterminator='\n')
    print("✅ File 'ultimate_blend.csv' created successfully! Ready for your FINAL submission.")

if __name__ == "__main__":
    create_ultimate_blend()
