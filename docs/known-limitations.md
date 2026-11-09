# Known limitations

- **Synthetic records.** Names, typos, nicknames, moves and look-alikes are generated, so the truth is
  known. Real data has error patterns this generator doesn't (OCR errors, transliteration, data-entry
  shortcuts), and absolute scores will differ.
- **People only.** Companies need different features: legal suffixes, trading names, registration
  numbers and parent-subsidiary links.
- **Batch resolution.** Every run resolves from scratch. Incremental resolution of a new record
  against existing clusters, and keeping cluster IDs stable over time, are not implemented.
- **The nickname list is short and English-centric.** Names from other languages need their own
  variant lists or a learned name-similarity model.
- **Labelled pairs are assumed.** Here they come from the generator; in practice they come from
  review queues, and active learning would make the labelling cheaper.
- **The threshold is fixed at 0.5.** It is not tuned per source or per field-completeness pattern.
