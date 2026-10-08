# Public release checklist

- [ ] All five authors approved the public code release.
- [ ] The authors selected and added a software license.
- [ ] `python -m pytest` passes in a clean Python 3.10 environment.
- [ ] `python -m shiftroute.synthetic` passes on CPU.
- [ ] The final staging tree contains no `.env`, key, token, signed URL, dataset,
      checkpoint, feature bank, private log, or internal review document.
- [ ] The repository URL is added to the manuscript and submission metadata.
- [ ] `CITATION.cff` is updated with the repository URL and, when available, DOI.
- [ ] A version tag (for example `v0.1.0`) is created and the commit hash is
      retained with the submitted manuscript.
- [ ] Large reproducibility artifacts are placed in a versioned archival service
      rather than committed to ordinary Git history.

