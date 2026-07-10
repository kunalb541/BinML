"""BinML quick start: fetch a real OGLE event and classify it, then plot the evolution."""
import binml

clf = binml.Classifier()  # fine-tuned model, CPU
print(f"loaded BinML model={clf.model_name} seq_len={clf.seq_len}")

# OGLE-2014-BLG-0289 -- a caustic-crossing binary
t, mag, err = binml.surveys.fetch_ogle_ews(2014, 289)
pred = clf.predict(t, mag, err)
print(pred)
print("  is_microlensing:", round(pred.is_microlensing, 3),
      " is_anomalous:", round(pred.is_anomalous, 3))

# probability evolution -> PNG (needs matplotlib: pip install "binml[plot]")
try:
    evo = clf.predict_evolution(t, mag, err)
    binml.plot_evolution(evo, title="OGLE-2014-BLG-0289").savefig(
        "ogle_2014_blg_0289_evolution.png", dpi=130, bbox_inches="tight")
    print("  saved ogle_2014_blg_0289_evolution.png")
except ImportError:
    print("  (install matplotlib for plots)")
