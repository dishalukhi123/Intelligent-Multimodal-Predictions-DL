# Fixed Animal Voice Detection project

## What was fixed
The error `Unsupported URL scheme in metadata URL: ./my_model/metadata.json` happens because this version of `speech-commands` requires an absolute HTTP(S) model URL. The HTML now builds the absolute URL from `window.location.origin`.

## Run
1. Extract the ZIP.
2. Keep `index.html` next to the `my_model` folder.
3. Open PowerShell in that folder.
4. Run `python -m http.server 8000`
5. Visit http://localhost:8000 in Chrome.
6. Click Start and allow microphone access.

Files in `my_model`: `model.json`, `metadata.json`, `weights.bin`.
Internet is needed for the TensorFlow.js libraries loaded from the CDN.
