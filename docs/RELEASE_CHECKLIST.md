# Public release checklist

- [ ] All six authors approved the public code release and author order:
      Shijiong Shen, Kexin Jia, Wenting Lu, Xinrong Zhu, Xin Sui, and
      Ta-Wen KUAN.
- [x] The MIT software license is included in `LICENSE`.
- [ ] `python -m pytest` passes in a clean Python 3.10 environment.
      (Release preparation also recorded 28/28 passing tests on Python 3.12.8.)
- [x] `python -m shiftroute.synthetic` passes on CPU.
- [x] The tracked release tree contains no `.env`, key, token, signed URL, dataset,
      checkpoint, feature bank, private log, or internal review document.
- [x] The repository URL is present in the code and submission metadata.
- [x] `CITATION.cff` records the repository URL and v1.0.0 metadata; article and
      archive DOIs remain to be added when available.
- [x] The annotated `v1.0.0` version tag is created and the commit hash is
      retained with the submitted manuscript.
- [ ] Large reproducibility artifacts are placed in a versioned archival service
      rather than committed to ordinary Git history.
