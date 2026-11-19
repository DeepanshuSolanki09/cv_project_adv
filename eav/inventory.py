"""EAV dataset inventory (STATUS: written against the official file layout of
https://github.com/nubcico/EAV; NOT RUN in this project because the data could not be
obtained).  Usage: python eav/inventory.py --root /path/to/EAV
Writes results/tables/eav_inventory.csv with ACTUAL counts found on disk."""
import argparse, csv, glob, os
import scipy.io

EMO5 = ["Neutral", "Anger", "Happiness", "Sadness", "Calmness"]

def scan(root):
    rows = []
    for subj in sorted(glob.glob(os.path.join(root, "subject*"))):
        s = os.path.basename(subj).rstrip("_")
        eeg = os.path.join(subj, "EEG", f"{s}_eeg.mat")
        n_eeg, shape = 0, ""
        if os.path.exists(eeg):
            m = scipy.io.loadmat(eeg)
            x = m.get("seg1", m.get("seg"))
            shape = "x".join(map(str, x.shape)); n_eeg = x.shape[-1]
        wavs = glob.glob(os.path.join(subj, "Audio", "*.wav"))
        mp4 = glob.glob(os.path.join(subj, "Video", "*.mp4"))
        rows.append({"participant": s, "eeg_trials": n_eeg, "eeg_shape": shape,
                     "audio_files": len(wavs), "video_files": len(mp4),
                     "video_speaking": sum("Speaking" in os.path.basename(v) for v in mp4)})
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    rows = scan(a.root)
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/eav_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} participants;", sum(r['eeg_trials'] for r in rows), "EEG trials;",
          sum(r['audio_files'] for r in rows), "audio;", sum(r['video_files'] for r in rows), "video")
