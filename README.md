# ceng778-termproject
* [ ] IStella indir
* [ ] See Istella 21 mpaper for example methods and use them
* [ ] try LTR and neural IR approaches
  * [ ] 𝜆-Mart -> LTR
  * [ ] monoT5(transfer) -> Neural model, transfer learning, zero shot -> DENİZ
  * [ ] monoT5(tuned) -> Neural model, tuned on istella22 -> DENİZ
  * [ ] 𝜆-Mart_{monoT5} -> LTR with monoT5 output added as a feature
* [ ] some possible other LTR methods:
  * [ ] Feature selection with GAS(P4)
    * similarity by concordoncy
    * decrease by similarity after each selection
  * [ ] MY 3 GAS alternatives
    * Instead of selecting best score select Best(score-(max or average or min) similarity)  
  * [ ]  Multi-Stage Retrieval Pipeline with LCE loss(P6 - ours)
* [ ] evaluate above aproaches
* [ ] Decide on combining stratagies(see the classical paper(https://ciir-publications.cs.umass.edu/getpdf.php?id=63)
* [ ] some ranking aggregation approaches are
  * [ ] combnz
  * [ ] combsum
  * [ ] borda voting
  * [ ] some supervised learning approach  
