# MODEL_LOADING_FIX_IT_PLAN.md

**Issue (short bullets)**

* You dropped `class_mapping_dicts`, so openWakeWord returns scores keyed by the model *basename* (e.g., `Hay--compUta_v_lrg`), but your downstream code still expects the legacy mapped label (e.g., `hay_computa`), so detections never match. ([GitHub][1], [GitHub][2])
* `wakeword_model_paths` was renamed to `wakeword_models` (a compatibility shim still maps the old name), so the rename itself isn't the break; the missing mapping is. ([GitHub][1])
* Supplying `inference_framework='tflite'` is redundant because TFLite is already the default; harmless but noisy. ([GitHub][1], [GitHub][2])
* Custom training (e.g., HA/Colab flow) yields a `.tflite` model file whose filename becomes the user-visible wake-word name unless you provide an explicit mapping. Pick a clean filename or pass `class_mapping_dicts`. ([Home Assistant][3], [GitHub][1])

---

**Fix (numbered, for a competent dev)**

1. Import and instantiate cleanly: `import openwakeword; model = openwakeword.Model(...)` (namespace clarity). ([GitHub][2])
2. Restore label mapping **or** adjust downstream logic to use the model key:

   ```python
   model = openwakeword.Model(
       wakeword_models=[custom_model_path],
       class_mapping_dicts=[{0: "hay_computa"}],
       vad_threshold=0.5,
   )
   ```

   (If single-class, you can skip mapping and just watch for `"Hay--compUta_v_lrg"`.) ([GitHub][1])
3. Drop the unnecessary `inference_framework='tflite'` unless you're actually switching frameworks; default is TFLite. ([GitHub][1], [GitHub][2])
4. Update your pipeline test to compare against whichever label you chose in step 2 (mapped label or basename) before counting detections. ([GitHub][1])
5. (Optional cleanup) Re-export/retrain your wake-word via the HA/Colab workflow and give the file the canonical name you want surfaced; then you can omit `class_mapping_dicts` entirely. ([Home Assistant][3], [GitHub][1])

[1]: https://raw.githubusercontent.com/dscripka/openWakeWord/main/openwakeword/model.py "raw.githubusercontent.com"
[2]: https://github.com/dscripka/openWakeWord/blob/main/README.md "openWakeWord/README.md at main · dscripka/openWakeWord · GitHub"
[3]: https://www.home-assistant.io/voice_control/create_wake_word/ "Wake words for Assist - Home Assistant"